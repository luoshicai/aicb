#!/usr/bin/env python3

"""Estimate per-GPU decode memory for HBF/SimAI inference workloads.

Current scope:
- Decode only
- Separates resident live KV cache from one-step decode tmp
- Supports percentile-style live-token scenarios and exact live-token snapshots
- Qwen3-Moe configs used by the HBF workload generator

This is an estimator, not a runtime allocator trace.

Role in the HBF workflow:
- this file is a decode-side memory constraint model
- it complements prefill memory estimation during strategy search
- it does not decide operator compute time or communication behavior
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from estimate_prefill_memory import (
    HBF_MEM_ROOT,
    SCRIPT_DIR,
    build_parallel_tag,
    dtype_bytes,
    get_data_parallel_size,
    human_bytes,
    load_json,
    qwen3_activation_breakdown,
    qwen3_param_breakdown,
    split_layers,
    validate_qwen3_config,
    write_text,
)


@dataclass
class LiveTokensScenario:
    name: str
    active_seqs: int
    total_live_tokens: int
    avg_live_tokens_per_seq: float
    source: str


@dataclass
class DecodeStageEstimate:
    scenario_name: str
    scenario_source: str
    pp_rank: int
    local_layers: int
    active_seqs: int
    total_live_tokens: int
    parameter_bytes: int
    kv_live_before_step_bytes: int
    kv_step_append_bytes: int
    kv_after_step_bytes: int
    workspace_bytes: int
    activation_peak_bytes: int
    activation_peak_with_kv_bytes: int
    decode_end_resident_bytes: int
    total_peak_bytes: int
    parameter_breakdown_bytes: Dict[str, int]
    activation_breakdown_bytes: Dict[str, int]
    kv_cache_breakdown_bytes: Dict[str, int]


def build_default_output_file(config_path: Path, config: Dict) -> Path:
    parts = config_path.parts
    if "compute_cache" in parts:
        idx = parts.index("compute_cache")
        if idx + 1 < len(parts) - 1:
            return HBF_MEM_ROOT / parts[idx + 1] / f"{build_parallel_tag(config)}.txt"
    return HBF_MEM_ROOT / "from_config" / f"{build_parallel_tag(config)}.txt"


def percentile_index(length: int, percentile: float) -> int:
    if length < 1:
        raise ValueError("cannot compute percentile for an empty sequence")
    if percentile <= 0:
        return 0
    if percentile >= 100:
        return length - 1
    return max(0, min(length - 1, math.ceil(percentile / 100.0 * length) - 1))


def scenario_from_snapshot(name: str, values: Sequence[int], source: str) -> LiveTokensScenario:
    if not values:
        raise ValueError(f"{name}: live-token snapshot is empty")
    total_live_tokens = sum(values)
    active_seqs = len(values)
    return LiveTokensScenario(
        name=name,
        active_seqs=active_seqs,
        total_live_tokens=total_live_tokens,
        avg_live_tokens_per_seq=total_live_tokens / active_seqs,
        source=source,
    )


def nearest_snapshot_to_total(
    name: str,
    snapshots: Sequence[Sequence[int]],
    target_total_live_tokens: float,
    source_prefix: str,
) -> LiveTokensScenario:
    ranked = sorted(
        (
            (
                abs(sum(snapshot) - target_total_live_tokens),
                abs((sum(snapshot) / len(snapshot)) - (target_total_live_tokens / len(snapshot))),
                idx,
            )
            for idx, snapshot in enumerate(snapshots)
            if snapshot
        )
    )
    if not ranked:
        raise ValueError("no non-empty snapshots available")
    snapshot_idx = ranked[0][2]
    return scenario_from_snapshot(
        name,
        snapshots[snapshot_idx],
        f"{source_prefix}:snapshot[{snapshot_idx}]",
    )


def normalize_int_list(values: Iterable[object], source: str) -> List[int]:
    normalized: List[int] = []
    for item in values:
        value = int(item)
        if value < 0:
            raise ValueError(f"{source}: live tokens must be >= 0, got {value}")
        normalized.append(value)
    return normalized


def parse_snapshot_line(line: str) -> List[int]:
    tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
    if not tokens:
        return []
    return normalize_int_list(tokens, source="text snapshot")


def load_live_token_snapshots(path: Path) -> List[List[int]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"live-tokens file is empty: {path}")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        if isinstance(payload, dict):
            if "snapshots" in payload:
                return [
                    normalize_int_list(snapshot, source=f"{path}:snapshots")
                    for snapshot in payload["snapshots"]
                ]
            if "live_tokens" in payload:
                return [normalize_int_list(payload["live_tokens"], source=f"{path}:live_tokens")]
            raise ValueError(
                f"unsupported JSON format in {path}; expected 'snapshots' or 'live_tokens'"
            )
        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"live-tokens JSON array is empty: {path}")
            if all(not isinstance(item, list) for item in payload):
                return [normalize_int_list(payload, source=str(path))]
            return [
                normalize_int_list(snapshot, source=f"{path}:snapshot[{idx}]")
                for idx, snapshot in enumerate(payload)
            ]
        raise ValueError(f"unsupported JSON payload in {path}")

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    rows = [parse_snapshot_line(line) for line in lines]
    if not rows:
        raise ValueError(f"no valid live-token rows found in {path}")
    if all(len(row) == 1 for row in rows):
        return [[row[0] for row in rows]]
    return rows


def build_file_scenarios(path: Path) -> List[LiveTokensScenario]:
    snapshots = load_live_token_snapshots(path)
    if len(snapshots) == 1:
        return [scenario_from_snapshot("trace_exact", snapshots[0], str(path))]

    totals = sorted(sum(snapshot) for snapshot in snapshots if snapshot)
    if not totals:
        raise ValueError(f"no non-empty snapshots found in {path}")

    scenarios = [
        nearest_snapshot_to_total("trace_mean", snapshots, sum(totals) / len(totals), str(path)),
        nearest_snapshot_to_total(
            "trace_p50", snapshots, totals[percentile_index(len(totals), 50)], str(path)
        ),
        nearest_snapshot_to_total(
            "trace_p90", snapshots, totals[percentile_index(len(totals), 90)], str(path)
        ),
        nearest_snapshot_to_total(
            "trace_p99", snapshots, totals[percentile_index(len(totals), 99)], str(path)
        ),
        nearest_snapshot_to_total("trace_max", snapshots, totals[-1], str(path)),
    ]

    deduped: List[LiveTokensScenario] = []
    seen = set()
    for scenario in scenarios:
        key = (scenario.source, scenario.total_live_tokens, scenario.active_seqs)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(scenario)
    return deduped


def build_cli_scenarios(config: Dict, args: argparse.Namespace) -> List[LiveTokensScenario]:
    active_seqs = args.active_seqs if args.active_seqs is not None else config["micro_batch"]
    if active_seqs < 1:
        raise ValueError("--active-seqs must be >= 1")

    scenarios: List[LiveTokensScenario] = []
    metrics = [
        ("avg", args.avg_live_tokens),
        ("p90", args.p90_live_tokens),
        ("p99", args.p99_live_tokens),
        ("max_bound", args.max_live_tokens),
    ]
    for name, live_tokens_per_seq in metrics:
        if live_tokens_per_seq is None:
            continue
        if live_tokens_per_seq < 0:
            raise ValueError(f"{name} live tokens must be >= 0")
        total_live_tokens = int(round(active_seqs * live_tokens_per_seq))
        scenarios.append(
            LiveTokensScenario(
                name=name,
                active_seqs=active_seqs,
                total_live_tokens=total_live_tokens,
                avg_live_tokens_per_seq=float(live_tokens_per_seq),
                source="cli",
            )
        )
    return scenarios


def qwen3_kv_bytes_for_tokens(
    config: Dict,
    local_layers: int,
    token_count: int,
    activation_bytes: int,
) -> int:
    if local_layers == 0 or token_count <= 0 or not bool(config.get("use_cache", True)):
        return 0
    tp = config["tensor_model_parallel_size"]
    local_kv_proj_dim = config["num_key_value_heads"] * config["head_dim"] // tp
    return local_layers * token_count * local_kv_proj_dim * 2 * activation_bytes


def make_decode_step_config(config: Dict, active_seqs: int, step_seq_length: int) -> Dict:
    step_config = dict(config)
    step_config["micro_batch"] = active_seqs
    step_config["seq_length"] = step_seq_length
    return step_config


def estimate_decode_scenarios(
    config: Dict,
    scenarios: Sequence[LiveTokensScenario],
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
    step_seq_length: int,
    workspace_bytes: int,
) -> Dict[str, List[DecodeStageEstimate]]:
    pp = config["pipeline_model_parallel"]
    layers_per_stage = split_layers(config["num_hidden_layers"], pp)
    estimates_by_scenario: Dict[str, List[DecodeStageEstimate]] = {}

    for scenario in scenarios:
        scenario_estimates: List[DecodeStageEstimate] = []
        decode_step_config = make_decode_step_config(config, scenario.active_seqs, step_seq_length)
        for pp_rank, local_layers in enumerate(layers_per_stage):
            param_breakdown = qwen3_param_breakdown(config, local_layers, pp_rank, weight_bytes)
            act_breakdown = qwen3_activation_breakdown(
                decode_step_config,
                local_layers,
                activation_bytes,
                moe_load_balance_factor,
            )
            parameter_bytes = sum(param_breakdown.values())
            activation_peak_bytes = act_breakdown["peak_component"]

            kv_live_before_step_bytes = qwen3_kv_bytes_for_tokens(
                config,
                local_layers,
                scenario.total_live_tokens,
                activation_bytes,
            )
            kv_step_append_bytes = qwen3_kv_bytes_for_tokens(
                config,
                local_layers,
                scenario.active_seqs * step_seq_length,
                activation_bytes,
            )
            kv_after_step_bytes = kv_live_before_step_bytes + kv_step_append_bytes

            if kv_step_append_bytes > 0:
                activation_peak_with_kv_bytes = max(
                    act_breakdown["moe_peak"],
                    act_breakdown["attention_peak_without_current_kv_overlap"],
                )
            else:
                activation_peak_with_kv_bytes = activation_peak_bytes

            decode_end_resident_bytes = (
                parameter_bytes + kv_after_step_bytes + workspace_bytes
            )
            kv_cache_breakdown = {
                "live_before_step": kv_live_before_step_bytes,
                "step_append": kv_step_append_bytes,
                "after_step": kv_after_step_bytes,
                "workspace": workspace_bytes,
            }
            scenario_estimates.append(
                DecodeStageEstimate(
                    scenario_name=scenario.name,
                    scenario_source=scenario.source,
                    pp_rank=pp_rank,
                    local_layers=local_layers,
                    active_seqs=scenario.active_seqs,
                    total_live_tokens=scenario.total_live_tokens,
                    parameter_bytes=parameter_bytes,
                    kv_live_before_step_bytes=kv_live_before_step_bytes,
                    kv_step_append_bytes=kv_step_append_bytes,
                    kv_after_step_bytes=kv_after_step_bytes,
                    workspace_bytes=workspace_bytes,
                    activation_peak_bytes=activation_peak_bytes,
                    activation_peak_with_kv_bytes=activation_peak_with_kv_bytes,
                    decode_end_resident_bytes=decode_end_resident_bytes,
                    total_peak_bytes=decode_end_resident_bytes + activation_peak_with_kv_bytes,
                    parameter_breakdown_bytes=param_breakdown,
                    activation_breakdown_bytes=act_breakdown,
                    kv_cache_breakdown_bytes=kv_cache_breakdown,
                )
            )
        estimates_by_scenario[scenario.name] = scenario_estimates

    return estimates_by_scenario


def render_text_report(
    config: Dict,
    config_path: Path,
    model_name: str,
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
    step_seq_length: int,
    workspace_bytes: int,
    scenarios: Sequence[LiveTokensScenario],
    estimates_by_scenario: Dict[str, List[DecodeStageEstimate]],
) -> str:
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
            "Decode step: "
            f"step_seq_length={step_seq_length} "
            f"use_cache={bool(config.get('use_cache', True))}"
        ),
        (
            "DType bytes: "
            f"weight={weight_bytes} "
            f"activation={activation_bytes} "
            f"moe_load_balance_factor={moe_load_balance_factor} "
            f"workspace={human_bytes(workspace_bytes)}"
        ),
        "",
    ]

    for scenario in scenarios:
        estimates = estimates_by_scenario[scenario.name]
        peak_stage = max(estimates, key=lambda item: item.total_peak_bytes)
        lines.append(
            f"Scenario {scenario.name}: "
            f"active_seqs={scenario.active_seqs}, "
            f"avg_live_tokens_per_seq={scenario.avg_live_tokens_per_seq:.2f}, "
            f"total_live_tokens={scenario.total_live_tokens}, "
            f"source={scenario.source}"
        )
        for stage in estimates:
            lines.append(
                f"  PP stage {stage.pp_rank}: "
                f"layers={stage.local_layers}, "
                f"params={human_bytes(stage.parameter_bytes)}, "
                f"kv_live_before={human_bytes(stage.kv_live_before_step_bytes)}, "
                f"kv_step_append={human_bytes(stage.kv_step_append_bytes)}, "
                f"workspace={human_bytes(stage.workspace_bytes)}, "
                f"decode_tmp_no_kv={human_bytes(stage.activation_peak_bytes)}, "
                f"decode_end_resident={human_bytes(stage.decode_end_resident_bytes)}, "
                f"decode_peak_with_kv={human_bytes(stage.total_peak_bytes)}"
            )
        lines.append(
            f"  Peak stage: pp_rank={peak_stage.pp_rank}, "
            f"estimated per-GPU decode peak={human_bytes(peak_stage.total_peak_bytes)}"
        )
        lines.append("  Breakdown for peak stage:")
        for key, value in peak_stage.parameter_breakdown_bytes.items():
            if value == 0:
                continue
            lines.append(f"    param.{key}: {human_bytes(value)}")
        for key, value in peak_stage.activation_breakdown_bytes.items():
            if key == "peak_component":
                continue
            lines.append(f"    activation.{key}: {human_bytes(value)}")
        for key, value in peak_stage.kv_cache_breakdown_bytes.items():
            if value == 0:
                continue
            lines.append(f"    kv_cache.{key}: {human_bytes(value)}")
        lines.append(
            f"    peak.decode_tmp_no_kv: {human_bytes(peak_stage.activation_peak_bytes)}"
        )
        lines.append(
            f"    peak.decode_tmp_with_kv: {human_bytes(peak_stage.activation_peak_with_kv_bytes)}"
        )
        lines.append(
            f"    peak.decode_end_resident: {human_bytes(peak_stage.decode_end_resident_bytes)}"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_json_payload(
    config: Dict,
    config_path: Path,
    model_name: str,
    weight_bytes: int,
    activation_bytes: int,
    moe_load_balance_factor: float,
    step_seq_length: int,
    workspace_bytes: int,
    scenarios: Sequence[LiveTokensScenario],
    estimates_by_scenario: Dict[str, List[DecodeStageEstimate]],
) -> Dict:
    scenario_payloads = []
    for scenario in scenarios:
        estimates = estimates_by_scenario[scenario.name]
        peak_stage = max(estimates, key=lambda item: item.total_peak_bytes)
        scenario_payloads.append(
            {
                **asdict(scenario),
                "peak_stage_pp_rank": peak_stage.pp_rank,
                "peak_stage_total_peak_bytes": peak_stage.total_peak_bytes,
                "peak_stage_decode_end_resident_bytes": peak_stage.decode_end_resident_bytes,
                "peak_stage_kv_live_before_step_bytes": peak_stage.kv_live_before_step_bytes,
                "peak_stage_kv_step_append_bytes": peak_stage.kv_step_append_bytes,
                "peak_stage_activation_peak_bytes": peak_stage.activation_peak_bytes,
                "peak_stage_activation_peak_with_kv_bytes": peak_stage.activation_peak_with_kv_bytes,
                "stages": [asdict(stage) for stage in estimates],
            }
        )
    return {
        "config_file": str(config_path),
        "model_name": model_name,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "moe_load_balance_factor": moe_load_balance_factor,
        "step_seq_length": step_seq_length,
        "workspace_bytes": workspace_bytes,
        "parallel_tag": build_parallel_tag(config),
        "scenarios": scenario_payloads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-GPU decode memory from HBF/SimAI Qwen3-Moe configs, "
            "including live KV cache scenarios."
        )
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the model/workload config JSON.",
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
        "--active-seqs",
        type=int,
        help="Concurrent active decode sequences on one GPU. Defaults to config micro_batch.",
    )
    parser.add_argument(
        "--avg-live-tokens",
        type=float,
        help="Average live tokens per active sequence before the next decode step.",
    )
    parser.add_argument(
        "--p90-live-tokens",
        type=float,
        help="P90 live tokens per active sequence before the next decode step.",
    )
    parser.add_argument(
        "--p99-live-tokens",
        type=float,
        help="P99 live tokens per active sequence before the next decode step.",
    )
    parser.add_argument(
        "--max-live-tokens",
        type=float,
        help="Upper-bound live tokens per active sequence before the next decode step.",
    )
    parser.add_argument(
        "--live-tokens-file",
        help=(
            "Optional file with exact live-token snapshots. "
            "Supported formats: JSON array, JSON snapshots, or text rows."
        ),
    )
    parser.add_argument(
        "--step-seq-length",
        type=int,
        default=1,
        help="Decode tokens processed per step. Default is 1.",
    )
    parser.add_argument(
        "--paged-kv-workspace-mib",
        type=float,
        default=0.0,
        help=(
            "Optional runtime workspace to reserve per GPU for paged KV kernels. "
            "Set to 128 to mimic the current HBF mock's flashinfer buffer."
        ),
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
        "--output-file",
        help="Explicit output report path.",
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

    if args.step_seq_length < 1:
        raise ValueError("--step-seq-length must be >= 1")

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
    workspace_bytes = int(round(args.paged_kv_workspace_mib * 1024 * 1024))

    scenarios: List[LiveTokensScenario] = []
    if args.live_tokens_file:
        scenarios.extend(build_file_scenarios(Path(args.live_tokens_file).resolve()))
    scenarios.extend(build_cli_scenarios(config, args))
    if not scenarios:
        raise ValueError(
            "provide at least one of --avg-live-tokens/--p90-live-tokens/--p99-live-tokens/"
            "--max-live-tokens or --live-tokens-file"
        )

    estimates_by_scenario = estimate_decode_scenarios(
        config,
        scenarios,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        moe_load_balance_factor=args.moe_load_balance_factor,
        step_seq_length=args.step_seq_length,
        workspace_bytes=workspace_bytes,
    )

    payload = render_json_payload(
        config,
        config_path,
        model_name,
        weight_bytes,
        activation_bytes,
        args.moe_load_balance_factor,
        args.step_seq_length,
        workspace_bytes,
        scenarios,
        estimates_by_scenario,
    )
    text_report = render_text_report(
        config,
        config_path,
        model_name,
        weight_bytes,
        activation_bytes,
        args.moe_load_balance_factor,
        args.step_seq_length,
        workspace_bytes,
        scenarios,
        estimates_by_scenario,
    )

    stdout_content = json.dumps(payload, indent=2) if args.json else text_report
    print(stdout_content)

    if args.no_save:
        return 0

    output_path = (
        Path(args.output_file).resolve()
        if args.output_file
        else build_default_output_file(config_path, config)
    )
    write_text(output_path, json.dumps(payload, indent=2) if args.json else text_report)
    print(f"Saved result to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
