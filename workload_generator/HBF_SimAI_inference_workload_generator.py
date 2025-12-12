# 导入与依赖
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
import workload_generator.mocked_model.MockedModel
import workload_generator.mocked_model.HBF_models.HBF_MockedQwen3 as MockedQwen3
from utils.utils import CommType, get_params, get_comp_out, extract_inference_averages
import os
from typing import List, Tuple
from collections import deque
import dataclasses
from enum import Enum

try:
    import torch
except ImportError as e:
    torch = None
    print("Failed to import 'torch'.")
import math
import re

# 数据结构定义
# Work_Item 数据类
@dataclasses.dataclass
class Work_Item:
    name: str = dataclasses.field(default="none")
    placeholder: int = dataclasses.field(default=-1)
    forward_compute_time: int = dataclasses.field(default=0)
    forward_comm: str = dataclasses.field(default="NONE")
    forward_comm_size: str = dataclasses.field(default="0")
    backward_compute_time: int = dataclasses.field(default=0)
    backward_comm: str = dataclasses.field(default="NONE")
    backward_comm_size: int = dataclasses.field(default=0)
    dp_compute_time: int = dataclasses.field(default=0)
    dp_comm: str = dataclasses.field(default="NONE")
    dp_comm_size: int = dataclasses.field(default=0)
    process_time: int = dataclasses.field(default=100)


# 作用：从性能缓存中查找某一层（stage）的前向计算耗时（单位可能是微秒）。
def _get_aiob_compute_time(compute_cache, forward_or_backward, stage, dowarn=True, aiob_enable=True):
    default_compute_time = 1
    if not aiob_enable:
        return default_compute_time
    compute_time_map = compute_cache
    if stage == "shared_experts" or stage == "dense_mlp":
        prefix = "mlp"
    else:
        prefix = stage

    for key, value in compute_time_map.items():
        if prefix == key:

            compute_time = compute_time_map.get(key)
            return compute_time

    if dowarn:
        print(f"[warn] can't match any stage with {stage}, using default_compute_time {default_compute_time}")
    return default_compute_time

# 简单封装每层的 ID 和名称，便于遍历模型结构。
class LayerInfo:
    def __init__(self, layer_id, layer_name):
        self.layer_id = layer_id
        self.layer_name = layer_name

# 这是整个脚本的核心类，负责生成 workload
class SimAIWorkload():
    def __init__(self, model, args, compue_cache=None):
        self.model = model
        self.args = args
        self.compute_cache = compute_cache
        self.workload = []
        self.seq_len = args.seq_length
        self.tp = args.tensor_model_parallel_size
        self.mbs = args.micro_batch
        if args.moe_enable:
            self.expert_model_parallel_size = args.expert_model_parallel_size
            self.num_experts = args.num_experts
            #如果有moe_router_topk则用，否则用num_experts_per_tok
            self.topk = args.moe_router_topk if hasattr(args,"moe_router_topk") else args.num_experts_per_tok

    # 递归遍历模型的所有子模块，收集所有关键层的信息（Attention、MLP、MoE 路由/专家等）。
    def get_model_details(self):
        layers = []
        visited = set()

        def traverse_model(model, child_id=0):
            if id(model) in visited:
                return
            visited.add(id(model))
            if (
                    isinstance(model, MockedDeepSeek.DeepSeekAttention)
                    or isinstance(model, MockedDeepSeek.DeepSeekMLP)
                    # or isinstance(model, MockedDeepSeek.DeepSeekMOE)
                    # or isinstance(model, MockedDeepSeek.DeepSeekColumnLinear)
                    # or isinstance(model, MockedDeepSeek.DeepSeekRowLinear)
                    or isinstance(model, MockedQwen3.Qwen3MoeRMSNorm)
                    or isinstance(model, MockedQwen3.Qwen3MoeAttention)
                    or isinstance(model, MockedQwen3.Qwen3MoeRoute)
                    or isinstance(model, MockedQwen3.Qwen3MoeExpert)
                ):
                    layers.append(LayerInfo(model.layer_id, model.name))
            for child in model.child_modules():
                traverse_model(child, child_id+1)

        traverse_model(model)

        return layers

    # 核心逻辑
    # 根据模型层（layers）信息，生成一个按层顺序的 self.workload 列表（每个元素是 Work_Item），
    # 每个 Work_Item 描述该层的前向计算耗时、前向/后向/数据并行时的通信类型与通信量等信息。这个 workload 会被上层调度器用于模拟调度与性能评估。 
    def workload_generate_aiob(self):
        default_compute_time = 1
        compute_time = 0

        # 计算 TP 通信量（AllReduce）
        tp_comm_size = 2 * self.mbs * self.seq_len * self.args.hidden_size
        layers = self.get_model_details()
        print("layer length:" , len(layers))

        for layer in layers:
            name = layer.layer_name
            forward_comm = "NONE"
            forward_comm_size = tp_comm_size
            #  EP（Expert Parallel）通信量
            ep_dispatch_size = tp_comm_size * self.topk // self.tp
            ep_combine_size = tp_comm_size * self.topk // self.tp

            # DeepSeek 特殊优化：FP8 量化因子
            if self.args.frame == "DeepSeek" or self.args.frame == "Qwen3-Moe":
                # for DeepEP based on https://github.com/parthpower/DeepEP/commit/50aee15f592bc22142eb04b7d718296b19613ae9
                ep_dispatch_size = int(ep_dispatch_size * MockedDeepSeek.FP8_FACTOR)

            if args.tensor_model_parallel_size == 1 :
                forward_comm = "NONE"
            else:
                forward_comm = "ALLREDUCE"

            # 归一化层 / 共享专家层（仅计算）
            if "norm" in name or "shared_expert" in name: # comp only layers
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name=name,
                        forward_compute_time=compute_time,
                        forward_comm="NONE",
                        forward_comm_size="0",
                        backward_compute_time=0,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=0,
                        dp_comm="NONE",
                        dp_comm_size=0,
                    )
                )
            # Attention / Dense MLP 层（计算 + TP AllReduce）
            elif "attention" in name or "dense_mlp" in name: # comp & ALLREDUCE comm layers
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name=name,
                        forward_compute_time=compute_time,
                        forward_comm=forward_comm,
                        forward_comm_size=forward_comm_size,
                        backward_compute_time=0,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=0,
                        dp_comm="NONE",
                        dp_comm_size=0,
                    )
                )
            # MoE 路由层（moe_route）
            elif "moe_route" in name: # moe route
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name="moe_route",
                        forward_compute_time=compute_time,
                        forward_comm="ALLTOALL_EP",
                        # forward_comm_size = "2*" + str(token_num) + "*" + str(self.args.moe_router_topk) + "*" + str(self.args.expert_dim),
                        forward_comm_size = ep_dispatch_size,
                        backward_compute_time=default_compute_time,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=default_compute_time,
                        dp_comm="NONE",
                        dp_comm_size=0
                    )
                )
            #  MoE 专家层（moe_expert）
            elif "moe_expert" in name: # moe experrt
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )

                if self.args.frame == "DeepSeek":
                    #TODO currently AiobDeepSeek doesn't support moe_route, should be fixed.
                    self.workload.append(
                        Work_Item(
                            name="moe_route",
                            forward_compute_time=default_compute_time,
                            forward_comm="ALLTOALL_EP",
                            # forward_comm_size = "2*" + str(token_num) + "*" + str(self.args.moe_router_topk) + "*" + str(self.args.expert_dim),
                            forward_comm_size = ep_dispatch_size,
                            backward_compute_time=default_compute_time,
                            backward_comm="NONE",
                            backward_comm_size=0,
                            dp_compute_time=default_compute_time,
                            dp_comm="NONE",
                            dp_comm_size=0
                        )
                    )
                self.workload.append(
                    Work_Item(
                        name="moe_expert",
                        forward_compute_time=compute_time,
                        forward_comm="ALLTOALL_EP",
                        forward_comm_size = ep_combine_size,
                        backward_compute_time=default_compute_time,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=default_compute_time,
                        dp_comm="NONE",
                        dp_comm_size=0
                    )
                )

    # dump_file() 方法：输出 workload 文件
    def dump_file(self, filename):
        filename = filename + ".txt"

        pp_comm_value = 2 * self.mbs * self.seq_len * self.args.hidden_size * (1 if self.args.pipeline_model_parallel > 1 else 0)

        pp_comm = (
            f"pp_comm: {pp_comm_value}"
        )
        with open(filename, "w") as f:
            f.write((
                f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "
                f"ep: {self.args.expert_model_parallel_size} "
                f"pp: {self.args.pipeline_model_parallel} "
                f"all_gpus: {self.args.world_size} "
                f"mode: 1 " # inference
                f"vpp: 1 ga: 1 checkpoints: 0 checkpoint_initiates: 0 "
            ) + pp_comm + "\n")

            f.write(str(len(self.workload)) + "\n")
            for item in self.workload:
                f.write(
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                    + "\n"
                )


# 主程序逻辑（if __name__ == "__main__":）
if __name__ == "__main__":
    import sys

    # args = get_params()
    # print(args)
    # Check if a config file is provided as a command line argument
    config_file = None
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print("Usage: python workload_generator.py <model_name> [config_file]")
        sys.exit(1)
    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    # 加载模型配置和实例化
    if "Qwen3-Moe" in model_name:
        args = MockedQwen3.Qwen3MoeParams(config_file)
        model = MockedQwen3.Qwen3MoeModel(args)
    elif "DeepSeek" in model_name:
        args = MockedDeepSeek.DeepSeekParams(config_file)
        model = MockedDeepSeek.DeepSeekModel(args)
    else:
        print(f"Invalid model name: {model_name}")
        sys.exit(1)
    # args = MockedDeepSeek.DeepSeekParams(config_file)
    # model = MockedDeepSeek.DeepSeekModel(args)

    # 创建结果目录
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-bs{args.micro_batch}-seq{args.seq_length}"

    # 获取 compute_cache（计算耗时）
    if args.aiob_enable:
        # 调用 Aiob 模型运行一次，获取性能数据
        if "Qwen3-Moe" in model_name:
            import workload_generator.mocked_model.HBF_models.HBF_AiobQwen3 as AiobQwen3
            aiob_model = AiobQwen3.Qwen3MoeModel(args)
            aiob_output_filepath = aiob_model()
        elif "DeepSeek" in model_name:
            import workload_generator.mocked_model.inference.AiobDeepSeek as AiobDeepSeek
            aiob_model = AiobDeepSeek.DeepSeekModel(args)
            aiob_output_filepath = aiob_model()
        else:
            print(f"Invalid model name: {model_name}")
            sys.exit(1)

    else:
        # 读取已有的 aiob 输出文件作为缓存
        # 这里的逻辑是如果不启用aiob，则会读取已有的output.txt文件，如果也没有这个output.txt文件，则会全0输出
        aiob_dir = "results/aiob_outputs"
        aiob_output_filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-bpg{args.micro_batch}-seq{args.seq_length}.txt"
        aiob_output_filepath = os.path.join(aiob_dir,aiob_output_filename)
        if not os.path.exists(aiob_output_filepath):
            print(f"aiob not enabled, and {aiob_output_filepath} not found. Using default compute time.")
            aiob_output_filepath = ""
        else:
            print(f"aiob not enabled, using existing file {aiob_output_filepath}.")
    compute_cache = extract_inference_averages(aiob_output_filepath,args)
    print("compute_cache = {")
    for key, value in compute_cache.items():
        print(f"    '{key}' : {value},")
    print("}")

    # 生成 workload 并保存
    work = SimAIWorkload(
        model, args,compute_cache
    )
    name_layers = work.workload_generate_aiob()
    # set comm_size = 0 for any comm_type == NONE
    for i in range(len(work.workload)):
        if work.workload[i].forward_comm == "NONE":
            work.workload[i].forward_comm_size = 0
        if work.workload[i].backward_comm == "NONE":
            work.workload[i].backward_comm_size = 0

    filepath = os.path.join(result_dir, filename)
    work.dump_file(filepath)
    print("workload save in :", filepath)
    print("Finish Model initialization")