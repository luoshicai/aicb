#!/usr/bin/env python3

"""Reusable memory-estimation APIs for HBF/SimAI inference strategy search."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from estimate_decode_memory import (
    DecodeStageEstimate,
    LiveTokensScenario,
    build_file_scenarios,
    estimate_decode_scenarios,
)
from estimate_prefill_memory import (
    HBF_AIOB_ROOT,
    HBF_MEM_ROOT,
    SCRIPT_DIR,
    StageEstimate,
    WorkloadSummary,
    build_parallel_tag,
    derive_default_output_file,
    derive_output_file_for_source,
    dtype_bytes,
    get_data_parallel_size,
    human_bytes,
    load_json,
    parse_workload,
    validate_qwen3_config,
    estimate_qwen3_stages,
)


def ensure_supported_qwen3_config(config: Dict) -> None:
    validate_qwen3_config(config)
    model_name = str(config.get("model_name", ""))
    if "Qwen3" not in model_name and config.get("model_type") != "qwen3_moe":
        raise ValueError(
            "this estimator currently supports only the Qwen3-Moe HBF configs"
        )


def apply_parallelism(
    config: Dict,
    *,
    tp: int,
    ep: int,
    pp: int,
    dp: int,
) -> Dict:
    if min(tp, ep, pp, dp) < 1:
        raise ValueError("tp/ep/pp/dp must all be >= 1")
    updated = dict(config)
    updated["tensor_model_parallel_size"] = tp
    updated["expert_model_parallel_size"] = ep
    updated["pipeline_model_parallel"] = pp
    updated["world_size"] = tp * ep * pp * dp
    ensure_supported_qwen3_config(updated)
    return updated


def load_and_prepare_config(config_path: Path, dp_size: Optional[int] = None) -> Dict:
    config = load_json(config_path)
    if dp_size is not None:
        if dp_size < 1:
            raise ValueError("--dp-size must be >= 1")
        config["world_size"] = (
            config["tensor_model_parallel_size"]
            * config["expert_model_parallel_size"]
            * config["pipeline_model_parallel"]
            * dp_size
        )
    ensure_supported_qwen3_config(config)
    return config


def infer_precision_bytes(
    config: Dict,
    *,
    weight_bytes: Optional[int] = None,
    activation_bytes: Optional[int] = None,
) -> Dict[str, int]:
    inferred_weight_bytes = dtype_bytes(config.get("torch_dtype"), default=2)
    resolved_weight_bytes = (
        weight_bytes if weight_bytes is not None else inferred_weight_bytes
    )
    resolved_activation_bytes = (
        activation_bytes if activation_bytes is not None else inferred_weight_bytes
    )
    return {
        "weight_bytes": resolved_weight_bytes,
        "activation_bytes": resolved_activation_bytes,
    }


def estimate_prefill_memory(
    config: Dict,
    *,
    weight_bytes: Optional[int] = None,
    activation_bytes: Optional[int] = None,
    moe_load_balance_factor: float = 1.0,
) -> Dict:
    ensure_supported_qwen3_config(config)
    precisions = infer_precision_bytes(
        config,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
    )
    stages = estimate_qwen3_stages(
        config,
        weight_bytes=precisions["weight_bytes"],
        activation_bytes=precisions["activation_bytes"],
        moe_load_balance_factor=moe_load_balance_factor,
    )
    peak_stage = max(stages, key=lambda item: item.total_peak_bytes)
    return {
        "config": config,
        "parallel_tag": build_parallel_tag(config),
        "weight_bytes": precisions["weight_bytes"],
        "activation_bytes": precisions["activation_bytes"],
        "moe_load_balance_factor": moe_load_balance_factor,
        "stages": stages,
        "peak_stage": peak_stage,
        "peak_stage_pp_rank": peak_stage.pp_rank,
        "peak_stage_total_peak_bytes": peak_stage.total_peak_bytes,
        "peak_stage_prefill_end_resident_bytes": peak_stage.prefill_end_resident_bytes,
        "peak_stage_kv_cache_bytes": peak_stage.kv_cache_bytes,
    }


def build_decode_scenarios(
    *,
    active_seqs: int,
    avg_live_tokens: Optional[float] = None,
    p90_live_tokens: Optional[float] = None,
    p99_live_tokens: Optional[float] = None,
    max_live_tokens: Optional[float] = None,
    live_tokens_file: Optional[Path] = None,
) -> List[LiveTokensScenario]:
    if active_seqs < 1:
        raise ValueError("active_seqs must be >= 1")

    scenarios: List[LiveTokensScenario] = []
    if live_tokens_file is not None:
        scenarios.extend(build_file_scenarios(live_tokens_file))

    cli_metrics = [
        ("avg", avg_live_tokens),
        ("p90", p90_live_tokens),
        ("p99", p99_live_tokens),
        ("max_bound", max_live_tokens),
    ]
    for name, per_seq_value in cli_metrics:
        if per_seq_value is None:
            continue
        if per_seq_value < 0:
            raise ValueError(f"{name} live tokens must be >= 0")
        total_live_tokens = int(round(active_seqs * per_seq_value))
        scenarios.append(
            LiveTokensScenario(
                name=name,
                active_seqs=active_seqs,
                total_live_tokens=total_live_tokens,
                avg_live_tokens_per_seq=float(per_seq_value),
                source="api",
            )
        )
    return scenarios


def estimate_decode_memory(
    config: Dict,
    *,
    scenarios: Sequence[LiveTokensScenario],
    weight_bytes: Optional[int] = None,
    activation_bytes: Optional[int] = None,
    moe_load_balance_factor: float = 1.0,
    step_seq_length: int = 1,
    workspace_bytes: int = 0,
) -> Dict:
    ensure_supported_qwen3_config(config)
    if step_seq_length < 1:
        raise ValueError("step_seq_length must be >= 1")
    if workspace_bytes < 0:
        raise ValueError("workspace_bytes must be >= 0")
    if not scenarios:
        raise ValueError("at least one decode scenario is required")

    precisions = infer_precision_bytes(
        config,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
    )
    estimates_by_scenario = estimate_decode_scenarios(
        config,
        scenarios,
        weight_bytes=precisions["weight_bytes"],
        activation_bytes=precisions["activation_bytes"],
        moe_load_balance_factor=moe_load_balance_factor,
        step_seq_length=step_seq_length,
        workspace_bytes=workspace_bytes,
    )
    scenario_lookup = {scenario.name: scenario for scenario in scenarios}
    scenario_peak_stages = {
        scenario_name: max(stage_estimates, key=lambda item: item.total_peak_bytes)
        for scenario_name, stage_estimates in estimates_by_scenario.items()
    }
    overall_peak_scenario_name = max(
        scenario_peak_stages,
        key=lambda name: scenario_peak_stages[name].total_peak_bytes,
    )
    overall_peak_stage = scenario_peak_stages[overall_peak_scenario_name]
    return {
        "config": config,
        "parallel_tag": build_parallel_tag(config),
        "weight_bytes": precisions["weight_bytes"],
        "activation_bytes": precisions["activation_bytes"],
        "moe_load_balance_factor": moe_load_balance_factor,
        "step_seq_length": step_seq_length,
        "workspace_bytes": workspace_bytes,
        "scenarios": list(scenarios),
        "estimates_by_scenario": estimates_by_scenario,
        "scenario_peak_stages": scenario_peak_stages,
        "overall_peak_scenario": scenario_lookup[overall_peak_scenario_name],
        "overall_peak_stage": overall_peak_stage,
        "overall_peak_total_peak_bytes": overall_peak_stage.total_peak_bytes,
    }


__all__ = [
    "DecodeStageEstimate",
    "HBF_AIOB_ROOT",
    "HBF_MEM_ROOT",
    "LiveTokensScenario",
    "SCRIPT_DIR",
    "StageEstimate",
    "WorkloadSummary",
    "apply_parallelism",
    "build_decode_scenarios",
    "build_parallel_tag",
    "derive_default_output_file",
    "derive_output_file_for_source",
    "ensure_supported_qwen3_config",
    "estimate_decode_memory",
    "estimate_prefill_memory",
    "get_data_parallel_size",
    "human_bytes",
    "infer_precision_bytes",
    "load_and_prepare_config",
    "load_json",
    "parse_workload",
]
