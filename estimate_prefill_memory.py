#!/usr/bin/env python3

"""Estimate per-GPU prefill memory for HBF/SimAI inference workloads.

Current scope:
- Prefill only
- Includes final KV cache when use_cache=true
- Qwen3-Moe configs used by the HBF workload generator

The estimator follows the existing SimAI workload split assumptions:
- Attention weights are tensor-parallel sharded.
- MoE expert weights are expert-parallel sharded and tensor-parallel sharded.
- Router weights are replicated.
- Parameters are resident for the whole run.
- Prefill activation peak is approximated as the maximum of per-layer peaks.

This is an estimator, not a runtime allocator trace.

Role in the HBF workflow:
- this file is a constraint model
- it filters infeasible strategies by per-GPU memory
- it does not generate compute time or communication workload
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
HBF_AIOB_ROOT = SCRIPT_DIR / "results" / "aiob_outputs" / "HBF"
HBF_MEM_ROOT = SCRIPT_DIR / "results" / "aiob_outputs" / "HBF_MEM"


DTYPE_TO_BYTES = {
    "float8": 1,
    "fp8": 1,
    "int8": 1,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
}


@dataclass
class WorkloadSummary:
    tp: int
    ep: int
    pp: int
    item_count: int
    stage_counts: Dict[str, int]


@dataclass
class StageEstimate:
    pp_rank: int
    local_layers: int
    parameter_bytes: int
    activation_peak_bytes: int
    activation_peak_with_kv_bytes: int
    kv_cache_bytes: int
    prefill_end_resident_bytes: int
    total_peak_bytes: int
    parameter_breakdown_bytes: Dict[str, int]
    activation_breakdown_bytes: Dict[str, int]
    kv_cache_breakdown_bytes: Dict[str, int]


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dtype_bytes(dtype_name: Optional[str], default: int) -> int:
    if dtype_name is None:
        return default
    normalized = str(dtype_name).strip().lower()
    if normalized not in DTYPE_TO_BYTES:
        return default
    return DTYPE_TO_BYTES[normalized]


def parse_workload(path: Path) -> WorkloadSummary:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if len(lines) < 2:
        raise ValueError(f"workload file is too short: {path}")

    header = lines[0]
    count = int(lines[1].strip())
    match = re.search(
        r"model_parallel_NPU_group:\s*(\d+)\s+ep:\s*(\d+)\s+pp:\s*(\d+)",
        header,
    )
    if not match:
        raise ValueError(f"failed to parse workload header: {header}")

    stage_counts: Dict[str, int] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        stage_name = line.split("\t", 1)[0]
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1

    return WorkloadSummary(
        tp=int(match.group(1)),
        ep=int(match.group(2)),
        pp=int(match.group(3)),
        item_count=count,
        stage_counts=stage_counts,
    )


def split_layers(num_layers: int, pipeline_parallel: int) -> List[int]:
    base, remainder = divmod(num_layers, pipeline_parallel)
    return [base + (1 if rank < remainder else 0) for rank in range(pipeline_parallel)]


def get_data_parallel_size(config: Dict) -> int:
    return config["world_size"] // (
        config["tensor_model_parallel_size"]
        * config["expert_model_parallel_size"]
        * config["pipeline_model_parallel"]
    )


def build_parallel_tag(config: Dict) -> str:
    return (
        f"tp{config['tensor_model_parallel_size']}_"
        f"pp{config['pipeline_model_parallel']}_"
        f"ep{config['expert_model_parallel_size']}_"
        f"dp{get_data_parallel_size(config)}"
    )


def build_fallback_subdir(config_path: Path) -> Path:
    parts = config_path.parts
    if "compute_cache" in parts:
        idx = parts.index("compute_cache")
        if idx + 1 < len(parts) - 1:
            return Path(f"accel_sim_{parts[idx + 1]}")
    return Path("from_config")


def derive_output_file_for_source(source_file: Path, config: Dict, output_root: Path) -> Path:
    parallel_tag = build_parallel_tag(config)
    try:
        relative_path = source_file.resolve().relative_to(HBF_AIOB_ROOT.resolve())
        return output_root / parallel_tag / relative_path
    except ValueError:
        return output_root / parallel_tag / source_file.name


def derive_default_output_file(config_path: Path, config: Dict, output_root: Path) -> Path:
    parallel_tag = build_parallel_tag(config)
    return output_root / parallel_tag / build_fallback_subdir(config_path) / f"{config_path.stem}.txt"


def render_text_report(
    config: Dict,
    config_path: Path,
    model_name: str,
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
    workload_summary: Optional[WorkloadSummary],
    estimates: List[StageEstimate],
) -> str:
    peak_stage = max(estimates, key=lambda item: item.total_peak_bytes)
    dp = get_data_parallel_size(config)

    lines = [
        f"Config: {config_path}",
        f"Model: {model_name}",
        (
            "Parallelism: "
            f"tp={config['tensor_model_parallel_size']} "
            f"ep={config['expert_model_parallel_size']} "
            f"pp={config['pipeline_model_parallel']} "
            f"dp={dp}"
        ),
        (
            "Shape: "
            f"micro_batch={config['micro_batch']} "
            f"seq_length={config['seq_length']} "
            f"layers={config['num_hidden_layers']}"
        ),
        (
            "DType bytes: "
            f"weight={weight_bytes} "
            f"activation={activation_bytes} "
            f"moe_load_balance_factor={moe_load_balance_factor} "
            f"use_cache={bool(config.get('use_cache', True))}"
        ),
    ]
    if workload_summary:
        lines.append(
            "Workload validation: "
            f"items={workload_summary.item_count} "
            f"stage_counts={workload_summary.stage_counts}"
        )
    lines.append("")
    for stage in estimates:
        lines.append(
            f"PP stage {stage.pp_rank}: "
            f"layers={stage.local_layers}, "
            f"params={human_bytes(stage.parameter_bytes)}, "
            f"prefill_peak_tmp_no_kv={human_bytes(stage.activation_peak_bytes)}, "
            f"final_kv_cache={human_bytes(stage.kv_cache_bytes)}, "
            f"prefill_end_resident={human_bytes(stage.prefill_end_resident_bytes)}, "
            f"prefill_peak_with_kv={human_bytes(stage.total_peak_bytes)}"
        )
    lines.append("")
    lines.append(
        f"Peak stage: pp_rank={peak_stage.pp_rank}, "
        f"estimated per-GPU peak with kv={human_bytes(peak_stage.total_peak_bytes)}"
    )
    lines.append("Breakdown for peak stage:")
    for key, value in peak_stage.parameter_breakdown_bytes.items():
        if value == 0:
            continue
        lines.append(f"  param.{key}: {human_bytes(value)}")
    for key, value in peak_stage.activation_breakdown_bytes.items():
        if key == "peak_component":
            continue
        lines.append(f"  activation.{key}: {human_bytes(value)}")
    for key, value in peak_stage.kv_cache_breakdown_bytes.items():
        if value == 0:
            continue
        lines.append(f"  kv_cache.{key}: {human_bytes(value)}")
    lines.append(
        f"  peak.prefill_tmp_no_kv: {human_bytes(peak_stage.activation_peak_bytes)}"
    )
    lines.append(
        f"  peak.prefill_tmp_with_kv: {human_bytes(peak_stage.activation_peak_with_kv_bytes)}"
    )
    lines.append(
        f"  peak.prefill_end_resident: {human_bytes(peak_stage.prefill_end_resident_bytes)}"
    )
    return "\n".join(lines) + "\n"


def render_json_payload(
    config: Dict,
    config_path: Path,
    model_name: str,
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
    workload_summary: Optional[WorkloadSummary],
    estimates: List[StageEstimate],
) -> Dict:
    peak_stage = max(estimates, key=lambda item: item.total_peak_bytes)
    return {
        "config_file": str(config_path),
        "model_name": model_name,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "moe_load_balance_factor": moe_load_balance_factor,
        "parallel_tag": build_parallel_tag(config),
        "workload_summary": asdict(workload_summary) if workload_summary else None,
        "peak_stage_pp_rank": peak_stage.pp_rank,
        "peak_stage_prefill_tmp_no_kv_bytes": peak_stage.activation_peak_bytes,
        "peak_stage_prefill_tmp_with_kv_bytes": peak_stage.activation_peak_with_kv_bytes,
        "peak_stage_kv_cache_bytes": peak_stage.kv_cache_bytes,
        "peak_stage_prefill_end_resident_bytes": peak_stage.prefill_end_resident_bytes,
        "peak_stage_total_peak_bytes": peak_stage.total_peak_bytes,
        "stages": [asdict(stage) for stage in estimates],
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_qwen3_config(config: Dict) -> None:
    required_keys = [
        "model_name",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "micro_batch",
        "seq_length",
        "tensor_model_parallel_size",
        "expert_model_parallel_size",
        "pipeline_model_parallel",
        "vocab_size",
        "world_size",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"config is missing required keys: {', '.join(missing)}")

    tp = config["tensor_model_parallel_size"]
    ep = config["expert_model_parallel_size"]
    pp = config["pipeline_model_parallel"]
    if config["num_attention_heads"] % tp != 0:
        raise ValueError("num_attention_heads must be divisible by tp")
    if config["num_key_value_heads"] % tp != 0:
        raise ValueError("num_key_value_heads must be divisible by tp")
    if config["num_experts"] % ep != 0:
        raise ValueError("num_experts must be divisible by ep")
    if config["world_size"] % (tp * ep * pp) != 0:
        raise ValueError("world_size must be divisible by tp * ep * pp")


def qwen3_param_breakdown(
    config: Dict,
    local_layers: int,
    pp_rank: int,
    weight_bytes: int,
) -> Dict[str, int]:
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    tp = config["tensor_model_parallel_size"]
    ep = config["expert_model_parallel_size"]
    pp = config["pipeline_model_parallel"]
    q_proj_dim = config["num_attention_heads"] * config["head_dim"]
    kv_proj_dim = config["num_key_value_heads"] * config["head_dim"]
    local_experts = config["num_experts"] // ep
    moe_intermediate_size = config["moe_intermediate_size"]
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    vocab_rows_per_rank = ceil_div(vocab_size, tp)
    local_intermediate = ceil_div(moe_intermediate_size, tp)

    embedding_params = 0
    final_norm_params = 0
    lm_head_params = 0
    if pp_rank == 0:
        embedding_params = vocab_rows_per_rank * hidden_size
    if pp_rank == pp - 1:
        final_norm_params = hidden_size
        if not tie_word_embeddings:
            lm_head_params = vocab_rows_per_rank * hidden_size

    per_layer_norm = 2 * hidden_size
    per_layer_attention = (
        hidden_size * (q_proj_dim + 2 * kv_proj_dim) // tp
        + q_proj_dim * hidden_size // tp
    )
    per_layer_router = hidden_size * config["num_experts"]
    per_layer_experts = local_experts * (3 * hidden_size * local_intermediate)

    params = {
        "embedding": embedding_params * weight_bytes,
        "transformer_norm": local_layers * per_layer_norm * weight_bytes,
        "transformer_attention": local_layers * per_layer_attention * weight_bytes,
        "transformer_router": local_layers * per_layer_router * weight_bytes,
        "transformer_experts": local_layers * per_layer_experts * weight_bytes,
        "final_norm": final_norm_params * weight_bytes,
        "lm_head": lm_head_params * weight_bytes,
    }
    return params


def qwen3_activation_breakdown(
    config: Dict,
    local_layers: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
) -> Dict[str, int]:
    if local_layers == 0:
        return {
            "hidden_buffer": 0,
            "attention_peak": 0,
            "moe_peak": 0,
            "peak_component": 0,
            "attention_q": 0,
            "attention_k": 0,
            "attention_v": 0,
            "attention_qkv": 0,
            "attention_current_layer_kv": 0,
            "attention_peak_without_current_kv_overlap": 0,
            "attention_output": 0,
            "router_logits": 0,
            "dispatch_duplicate_buffer": 0,
            "expert_up_intermediate": 0,
            "expert_gate_intermediate": 0,
            "expert_output": 0,
        }

    tokens = config["micro_batch"] * config["seq_length"]
    hidden_size = config["hidden_size"]
    tp = config["tensor_model_parallel_size"]
    ep = config["expert_model_parallel_size"]
    topk = config["num_experts_per_tok"]
    num_experts = config["num_experts"]
    moe_intermediate_size = config["moe_intermediate_size"]
    local_q_proj_dim = config["num_attention_heads"] * config["head_dim"] // tp
    local_kv_proj_dim = config["num_key_value_heads"] * config["head_dim"] // tp
    local_hidden = ceil_div(hidden_size, tp)
    local_intermediate = ceil_div(moe_intermediate_size, tp)

    hidden = tokens * hidden_size * activation_bytes
    residual = hidden
    q = tokens * local_q_proj_dim * activation_bytes
    k = tokens * local_kv_proj_dim * activation_bytes
    v = tokens * local_kv_proj_dim * activation_bytes
    attn_out = tokens * local_q_proj_dim * activation_bytes
    attn_lse = tokens * (config["num_attention_heads"] // tp) * 4

    router_logits = tokens * num_experts * activation_bytes
    topk_weights = tokens * topk * 4
    topk_indices = tokens * topk * 4

    duplicated_dispatch = int(
        tokens * topk * local_hidden * activation_bytes * moe_load_balance_factor
    )
    local_routed_tokens = math.ceil(tokens * topk * moe_load_balance_factor / ep)
    expert_up = local_routed_tokens * (2 * local_intermediate) * activation_bytes
    expert_gate = local_routed_tokens * local_intermediate * activation_bytes
    expert_down = local_routed_tokens * hidden_size * activation_bytes

    attention_peak = hidden + residual + q + k + v + attn_out + attn_lse
    current_layer_kv = k + v
    moe_peak = (
        hidden
        + residual
        + router_logits
        + topk_weights
        + topk_indices
        + duplicated_dispatch
        + expert_up
        + expert_gate
        + expert_down
    )

    return {
        "hidden_buffer": hidden,
        "attention_peak": attention_peak,
        "moe_peak": moe_peak,
        "peak_component": max(attention_peak, moe_peak),
        "attention_q": q,
        "attention_k": k,
        "attention_v": v,
        "attention_qkv": q + k + v,
        "attention_current_layer_kv": current_layer_kv,
        "attention_peak_without_current_kv_overlap": attention_peak - current_layer_kv,
        "attention_output": attn_out,
        "router_logits": router_logits,
        "dispatch_duplicate_buffer": duplicated_dispatch,
        "expert_up_intermediate": expert_up,
        "expert_gate_intermediate": expert_gate,
        "expert_output": expert_down,
    }


def qwen3_kv_cache_breakdown(
    config: Dict,
    local_layers: int,
    activation_bytes: int,
) -> Dict[str, int]:
    if local_layers == 0 or not bool(config.get("use_cache", True)):
        return {
            "per_layer": 0,
            "final": 0,
        }

    tokens = config["micro_batch"] * config["seq_length"]
    tp = config["tensor_model_parallel_size"]
    local_kv_proj_dim = config["num_key_value_heads"] * config["head_dim"] // tp
    kv_per_layer = tokens * local_kv_proj_dim * 2 * activation_bytes
    return {
        "per_layer": kv_per_layer,
        "final": local_layers * kv_per_layer,
    }


def estimate_qwen3_stages(
    config: Dict,
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
) -> List[StageEstimate]:
    pp = config["pipeline_model_parallel"]
    layers_per_stage = split_layers(config["num_hidden_layers"], pp)
    estimates: List[StageEstimate] = []
    for pp_rank, local_layers in enumerate(layers_per_stage):
        param_breakdown = qwen3_param_breakdown(config, local_layers, pp_rank, weight_bytes)
        act_breakdown = qwen3_activation_breakdown(
            config,
            local_layers,
            activation_bytes,
            moe_load_balance_factor,
        )
        kv_breakdown = qwen3_kv_cache_breakdown(config, local_layers, activation_bytes)
        parameter_bytes = sum(param_breakdown.values())
        activation_peak_bytes = act_breakdown["peak_component"]
        if kv_breakdown["final"] > 0:
            activation_peak_with_kv_bytes = max(
                act_breakdown["moe_peak"],
                act_breakdown["attention_peak_without_current_kv_overlap"],
            )
        else:
            activation_peak_with_kv_bytes = activation_peak_bytes
        kv_cache_bytes = kv_breakdown["final"]
        prefill_end_resident_bytes = parameter_bytes + kv_cache_bytes
        estimates.append(
            StageEstimate(
                pp_rank=pp_rank,
                local_layers=local_layers,
                parameter_bytes=parameter_bytes,
                activation_peak_bytes=activation_peak_bytes,
                activation_peak_with_kv_bytes=activation_peak_with_kv_bytes,
                kv_cache_bytes=kv_cache_bytes,
                prefill_end_resident_bytes=prefill_end_resident_bytes,
                total_peak_bytes=prefill_end_resident_bytes + activation_peak_with_kv_bytes,
                parameter_breakdown_bytes=param_breakdown,
                activation_breakdown_bytes=act_breakdown,
                kv_cache_breakdown_bytes=kv_breakdown,
            )
        )
    return estimates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-GPU prefill memory, including final KV cache when use_cache=true, "
            "from HBF/SimAI Qwen3-Moe configs."
        )
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the workload-generation config JSON.",
    )
    parser.add_argument(
        "--workload-file",
        help="Optional workload .txt file. Used only for split/header validation.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        help=(
            "Override data parallel size. "
            "When set, world_size is recomputed as tp * ep * pp * dp."
        ),
    )
    parser.add_argument(
        "--source-aiob-path",
        help=(
            "Optional HBF AIOB source file or directory. "
            "If provided, result files are mirrored under results/aiob_outputs/HBF_MEM."
        ),
    )
    parser.add_argument(
        "--output-file",
        help="Explicit output report path. Use this only when writing a single result file.",
    )
    parser.add_argument(
        "--weight-bytes",
        type=int,
        default=None,
        help="Override bytes per parameter. Default comes from torch_dtype.",
    )
    parser.add_argument(
        "--activation-bytes",
        type=int,
        default=None,
        help="Override bytes per activation element. Default comes from torch_dtype.",
    )
    parser.add_argument(
        "--moe-load-balance-factor",
        type=float,
        default=1.0,
        help="Scale local routed tokens to model MoE imbalance. 1.0 means perfectly balanced.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable report.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print the report to stdout and do not write result files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    config = load_json(config_path)
    if args.dp_size is not None:
        if args.dp_size < 1:
            raise ValueError("--dp-size must be >= 1")
        config["world_size"] = (
            config["tensor_model_parallel_size"]
            * config["expert_model_parallel_size"]
            * config["pipeline_model_parallel"]
            * args.dp_size
        )
    validate_qwen3_config(config)

    model_name = str(config.get("model_name", ""))
    if "Qwen3" not in model_name and config.get("model_type") != "qwen3_moe":
        raise ValueError(
            "this estimator currently supports only the Qwen3-Moe HBF configs"
        )

    inferred_weight_bytes = dtype_bytes(config.get("torch_dtype"), default=2)
    weight_bytes = args.weight_bytes if args.weight_bytes is not None else inferred_weight_bytes
    activation_bytes = (
        args.activation_bytes if args.activation_bytes is not None else inferred_weight_bytes
    )

    workload_summary = None
    if args.workload_file:
        workload_summary = parse_workload(Path(args.workload_file).resolve())
        if workload_summary.tp != config["tensor_model_parallel_size"]:
            raise ValueError("workload tp does not match config tp")
        if workload_summary.ep != config["expert_model_parallel_size"]:
            raise ValueError("workload ep does not match config ep")
        if workload_summary.pp != config["pipeline_model_parallel"]:
            raise ValueError("workload pp does not match config pp")

    estimates = estimate_qwen3_stages(
        config,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        moe_load_balance_factor=args.moe_load_balance_factor,
    )

    if args.output_file and args.source_aiob_path:
        source_path = Path(args.source_aiob_path).resolve()
        if source_path.is_dir():
            raise ValueError("--output-file cannot be used together with a directory source-aiob-path")

    payload = render_json_payload(
        config,
        config_path,
        model_name,
        weight_bytes,
        activation_bytes,
        args.moe_load_balance_factor,
        workload_summary,
        estimates,
    )
    text_report = render_text_report(
        config,
        config_path,
        model_name,
        weight_bytes,
        activation_bytes,
        args.moe_load_balance_factor,
        workload_summary,
        estimates,
    )

    if args.json:
        stdout_content = json.dumps(payload, indent=2)
    else:
        stdout_content = text_report

    print(stdout_content)

    if args.no_save:
        return 0

    output_paths: List[Path] = []
    if args.output_file:
        output_paths = [Path(args.output_file).resolve()]
    elif args.source_aiob_path:
        source_path = Path(args.source_aiob_path).resolve()
        if source_path.is_file():
            output_paths = [derive_output_file_for_source(source_path, config, HBF_MEM_ROOT)]
        elif source_path.is_dir():
            source_files = sorted(path for path in source_path.rglob("*.txt") if path.is_file())
            if not source_files:
                raise ValueError(f"no .txt files found under source-aiob-path directory: {source_path}")
            output_paths = [
                derive_output_file_for_source(source_file, config, HBF_MEM_ROOT)
                for source_file in source_files
            ]
        else:
            raise ValueError(f"source-aiob-path does not exist: {source_path}")
    else:
        output_paths = [derive_default_output_file(config_path, config, HBF_MEM_ROOT)]

    file_content = json.dumps(payload, indent=2) if args.json else text_report
    for output_path in output_paths:
        write_text(output_path, file_content)

    if len(output_paths) == 1:
        print(f"Saved result to: {output_paths[0]}")
    else:
        print(f"Saved {len(output_paths)} result files under: {output_paths[0].parent.parent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
