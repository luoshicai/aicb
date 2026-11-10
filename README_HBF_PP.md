# HBF 推理策略搜索：PP、microbatch 和双池容量

实现属于 **SimAI 的 AICB 子仓库**。Riven 开发路径为 `/home/lsc/HBF/SimAI/aicb`，R760 部署路径为 `/home/lsc/HBF-LLM/SimAI/aicb`；实验输入、日志和结果放在仓库外的 `experiments` 中。

上游基线与旧改动的对应关系见父仓库的 `README_HBF_PORT.md`，可用
`python3 ../scripts/check_hbf_upstream.py --check-remote --github-ssh` 复核。
旧 `HBF_inference_workload_with_computeCache.sh` 入口保留外部计算输入路径，
恢复了 HBF 默认配置；缺失、非正或非有限的计算时间会报错。
外部 AIOB 的 `time_gpu_avg` 使用 μs，经原提取器转换为 workload 的 ns。
旧入口采用其历史负载口径；本文件下述固定全局 batch 与 PP 前向调度，
使用 `build_strategy_body_workload.py` / `search_inference_pipeline.py`。

## 能力与边界

- `inference_strategy_contract.py`：固定全局 batch，TP/EP/PP/DP 分组、microbatch、逐流水级 HBM/HBF 容量与放置。复用原 `estimate_prefill_memory.py` 的层与首尾权重分配。PP4 支持 94 层的 24/24/23/23 分配；KV 覆盖完整副本批次，工作区按 microbatch，额外预留全部 microbatch 的边界缓冲。
- `capacity_strategy_search.py`：枚举候选、分池容量与计算标签覆盖；不把专家微基准当作完整层时间。
- `build_strategy_body_workload.py`：校验精确完整层 profile，生成各流水级的单 microbatch SimAI workload 和 JSON bundle。保留原 PP1、单 microbatch 的文本入口，并接入已有 TP 归约审计。
- `inference_pipeline.py`：前向流水线事件调度；PP 传输占用相邻两级，不与这两级计算重叠。不同流水级可以并行计算，缓冲区与容量预算一致。
- `evaluate_inference_pipeline.py`：实际调用 SimAI analytical 得到各级 TP/EP 服务时间；再结合显式 PP 传输代价进行前向调度，不使用后端的训练式 PP 气泡公式。
- `search_inference_pipeline.py`：完整候选筛选、精确 profile 查找、逐级评估、有限域排名和缺失采样清单。默认每级选择满足容量的最小 q；缺数据保留 `UNKNOWN`，预测 profile 不覆盖 reference，只有部分候选完成时仅给 `best_completed`。

当前支持 BF16 Qwen3-MoE、均衡受控路由、顺序专家实现、94 层主体 prefill。并行分组采用本项目约定 `G=TP×EP×PP×DP`，不代表所有 MoE 框架。`rank_batch/rank_tokens/replica_tokens/M` 是**单个 microbatch** 的计算形状；`replica_batch` 是整个 DP 副本的请求总数。

PP 级间通信可输入具有 workload 指纹的独立代价文件，或显式指定有效带宽与延迟。有效带宽是**每副本、每边界聚合 GB/s**，不是每卡带宽，也不是实测 HBF 参数。当前评估假设各级 TP/EP 服务隔离、DP 副本对称，不模拟它们之间共享 NIC 的争用。结果明确输出 `physical_pp_validation=false`；不能把软件联调结果当成物理 PP 性能已验证。

计算时间仍来自 Accel-Sim 生成的完整层 profile。embedding/末尾 head 计入容量，尚未计入主体时间，因此输出不是完整 TTFT/TPOT。已有冻结实验目录及其代码不改动。

## 容量与覆盖检查

在 AICB 目录执行，结果目录必须是新目录或空目录：

```bash
python3 capacity_strategy_search.py \
  --config /path/to/qwen_config.json \
  --world-size 32 --global-batch 32 --seq 256 \
  --candidate-tp 1,2,4 --candidate-ep 4,8,16,32 --candidate-pp 1,2,4 \
  --microbatch-sizes 8,16,32 --hbm-gib 40 --hbf-gib 0 \
  --output /path/to/new_capacity_result
```

`--microbatch-sizes` 省略时每副本一个 microbatch。实际支持的模型维度和负载必须可整除，拒绝原因会保存；纯 HBM 使用 `--hbf-gib 0`。可选 `--profiles` 读取既有 V2 微基准 CSV，仅报告覆盖；该 CSV 仍限定原采样模型与实现。

容量结果以 `stages` 为准。`q_min_by_stage` 是逐级放置，`feasible` 表示各级均存在可行放置；旧 `q_min/feasible_q` 表示所有级使用同一个 q。顶层其他旧容量字段描述 all-HBM 峰值最大的那一级，不可当作各级总和。

## 自动搜索

profile 目录可存现有带 `profile` 字段的完整层 manifest，或直接存 profile JSON；精确匹配模型哈希、TP、M、E、rank batch、seq、q 和延迟。PP 所在级的层数通过 workload 展开，局部 profile 的缓存/执行上下文仍需用户审计为等价。TP>1 还需要可读取且哈希一致的归约审计文件。

```bash
python3 search_inference_pipeline.py \
  --config /path/to/qwen_config.json --profile-dir /path/to/audited_profiles \
  --world-size 32 --global-batch 32 --seq 256 \
  --candidate-pp 1,2,4 --microbatch-sizes 8,16,32 \
  --hbm-gib 40 --hbf-gib 128 --hbf-latency 500 \
  --output /path/to/new_search --dry-run
```

PP>1 缺少显式网络参数时报告 UNKNOWN；实际评估时移除 `--dry-run`，指定 `--simai-root`，并提供经过选择的 `--pp-effective-bandwidth-gbytes-per-sec` 与 `--pp-boundary-latency-us`。两者均需显式提供，没有偷偷取零延迟的默认值。结果保存候选状态、排名范围及 `missing_profiles.json`；不按 FLOPs 为缺失点填时间。

## 单个 PP 候选

```bash
python3 build_strategy_body_workload.py \
  --config /path/to/qwen_config.json --profile /path/to/stage_profiles.json \
  --world-size 32 --global-batch 32 --seq 256 --tp 1 --ep 8 --pp 4 \
  --microbatch-size 8 --q 0 0 0 0 --hbf-latency 100 \
  --output /path/to/new_pipeline.bundle.json

python3 evaluate_inference_pipeline.py \
  --bundle /path/to/new_pipeline.bundle.json \
  --simai-root /home/lsc/HBF-LLM/SimAI \
  --pp-costs /path/to/audited_pp_costs.json \
  --output /path/to/new_pipeline_result
```

`stage_profiles.json` 使用 `{"stage_profiles": [profile0, profile1, ...]}`；所有级 q/形状相同时可只给一个 profile，由入口重复匹配。`--q` 可给一个所有级相同的值，或恰好 PP 个值。

`pp_costs.json` 必须包含 `schema="aicb_pp_costs_v1"`、`source`、`workload_sha256`、`time_us`。哈希由 `evaluate_inference_pipeline.workload_signature(bundle)` 生成；`time_us` 长度为 PP−1，每项为一次 microbatch 在对应边界的传输时间。也可通过 `effective_bandwidth_costs()` 生成显式解析模型。零传输只应用于标明用途的调度诊断；正式性能标签需要可信的通信参数。

## 验证与复现

```bash
python3 -m unittest -v test_inference_strategy_contract test_strategy_body_workload test_inference_pipeline
```

通过 `HBF_PP1_CONTRACT` 指定旧冻结契约，可校验15个PP1容量点；通过 `HBF_PROFILE_FIXTURE` 指定原 `simulation_results.csv`，可校验缺失数据不会变成零时间。未提供这些外部文件时对应测试会明确 skip。

单位：profile 为 μs，SimAI workload 为 ns，CSV 汇总为 μs且可能取整；最终流水线时间来自关键路径。各级服务时间与空闲时间之和不能当作可直接相加的系统时延。每次后端调用使用唯一输出前缀并保存输入、二进制、日志和输出哈希。
