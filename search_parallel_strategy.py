#!/usr/bin/env python3

"""Search feasible HBF/SimAI inference parallel strategies for one workload phase.

This script sits above workload generation and memory estimation:
- it enumerates candidate tp/ep/pp choices
- it uses memory models as feasibility filters
- it prepares candidate sets for later SimAI validation

It is intentionally not the workload builder itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from memory_model import (
    apply_parallelism,
    build_decode_scenarios,
    build_parallel_tag,
    estimate_decode_memory,
    estimate_prefill_memory,
    load_and_prepare_config,
)


GIB = 1024**3


def parse_int_list(raw_value: Optional[str], flag_name: str) -> Optional[List[int]]:
    if raw_value is None:
        return None
    values: List[int] = []
    for token in raw_value.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"{flag_name} values must be >= 1")
        values.append(value)
    if not values:
        raise ValueError(f"{flag_name} did not contain any valid integers")
    return sorted(set(values))


def positive_divisors(limit: int, *, divisible_of: int) -> List[int]:
    values: List[int] = []
    for candidate in range(1, limit + 1):
        if divisible_of % candidate == 0:
            values.append(candidate)
    return values


def auto_tp_candidates(config: Dict, total_gpus: int) -> List[int]:
    max_candidate = min(total_gpus, config["num_attention_heads"], config["num_key_value_heads"])
    return [
        value
        for value in positive_divisors(max_candidate, divisible_of=max_candidate)
        if total_gpus % value == 0
        and config["num_attention_heads"] % value == 0
        and config["num_key_value_heads"] % value == 0
    ]


def auto_ep_candidates(config: Dict, total_gpus: int) -> List[int]:
    max_candidate = min(total_gpus, config["num_experts"])
    return [
        value
        for value in positive_divisors(max_candidate, divisible_of=max_candidate)
        if total_gpus % value == 0 and config["num_experts"] % value == 0
    ]


def auto_pp_candidates(config: Dict, total_gpus: int) -> List[int]:
    max_candidate = min(total_gpus, config["num_hidden_layers"])
    return [
        value
        for value in positive_divisors(max_candidate, divisible_of=max_candidate)
        if total_gpus % value == 0
    ]


def log2_penalty(value: int, coefficient: float) -> float:
    if value <= 1:
        return 0.0
    return coefficient * math.log2(value)


def coarse_latency_penalty(tp: int, ep: int, pp: int, gpus_per_node: int) -> float:
    penalty = 1.0
    penalty += log2_penalty(tp, 0.12)
    penalty += log2_penalty(ep, 0.08)
    penalty += log2_penalty(pp, 0.15)
    penalty += 0.25 * max(0, math.ceil(tp / gpus_per_node) - 1)
    penalty += 0.20 * max(0, math.ceil(ep / gpus_per_node) - 1)
    penalty += 0.15 * max(0, math.ceil((tp * ep) / gpus_per_node) - 1)
    return penalty


def infer_inference_phase(config: Dict, requested_phase: str) -> str:
    seq_length = int(config["seq_length"])
    if seq_length < 1:
        raise ValueError("seq_length must be >= 1")

    inferred_phase = "decode" if seq_length == 1 else "prefill"
    if requested_phase == "auto":
        return inferred_phase
    if requested_phase != inferred_phase:
        raise ValueError(
            f"config seq_length={seq_length} implies {inferred_phase}, "
            f"but --inference-phase={requested_phase}"
        )
    return requested_phase


def load_latency_table(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError(f"latency table does not exist: {path}")

    suffix = path.suffix.lower()
    entries: Dict[str, Dict[str, float]] = {}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "strategies" in payload:
            rows = payload["strategies"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = [
                {"parallel_tag": parallel_tag, **value}
                for parallel_tag, value in payload.items()
            ]
        else:
            raise ValueError(f"unsupported latency JSON format: {path}")
        for row in rows:
            parallel_tag = str(row["parallel_tag"])
            entries[parallel_tag] = {
                key: float(value)
                for key, value in row.items()
                if key != "parallel_tag" and value is not None
            }
        return entries

    if suffix != ".csv":
        raise ValueError(f"unsupported latency table format: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parallel_tag = str(row["parallel_tag"]).strip()
            if not parallel_tag:
                continue
            normalized: Dict[str, float] = {}
            for key, value in row.items():
                if key == "parallel_tag" or value is None or value == "":
                    continue
                normalized[key] = float(value)
            entries[parallel_tag] = normalized
    return entries


def lookup_latency_ms(row: Optional[Dict[str, float]]) -> Optional[float]:
    if row is None:
        return None
    for key in ("latency_ms", "time_ms", "latency", "time"):
        if key in row:
            return float(row[key])
    return None


def candidate_record(candidate: Dict) -> Dict:
    return {
        "parallel_tag": candidate["parallel_tag"],
        "phase": candidate["phase"],
        "tp": candidate["tp"],
        "ep": candidate["ep"],
        "pp": candidate["pp"],
        "dp": candidate["dp"],
        "peak_gib": round(candidate["peak_bytes"] / GIB, 4),
        "usable_memory_gib": round(candidate["usable_memory_bytes"] / GIB, 4),
        "memory_margin_gib": round(candidate["memory_margin_bytes"] / GIB, 4),
        "latency_ms": candidate["latency_ms"],
        "coarse_latency_penalty": round(candidate["coarse_latency_penalty"], 6),
        "coarse_throughput_score": round(candidate["coarse_throughput_score"], 6),
        "cluster_tokens_per_s": (
            round(candidate["cluster_tokens_per_s"], 6)
            if candidate["cluster_tokens_per_s"] is not None
            else None
        ),
        "selected_decode_scenario": candidate["selected_decode_scenario"],
        "selected_decode_active_seqs": candidate["selected_decode_active_seqs"],
        "selected_decode_total_live_tokens": candidate["selected_decode_total_live_tokens"],
        "memory_feasible": candidate["memory_feasible"],
        "latency_feasible": candidate["latency_feasible"],
        "coarse_penalty_feasible": candidate["coarse_penalty_feasible"],
        "feasible": candidate["feasible"],
        "reject_reason": ";".join(candidate["reject_reasons"]),
    }


def ranking_score(candidate: Dict) -> float:
    if candidate["cluster_tokens_per_s"] is not None:
        return candidate["cluster_tokens_per_s"]
    return candidate["coarse_throughput_score"]


def render_text_report(payload: Dict, topk: int) -> str:
    lines = [
        f"Config: {payload['config_file']}",
        (
            "Phase: "
            f"{payload['inference_phase']} "
            f"(micro_batch={payload['micro_batch']}, seq_length={payload['seq_length']})"
        ),
        (
            "Hardware: "
            f"nodes={payload['num_nodes']} "
            f"gpus_per_node={payload['gpus_per_node']} "
            f"total_gpus={payload['total_gpus']}"
        ),
        (
            "Memory budget per GPU: "
            f"physical={payload['gpu_memory_gib']:.2f} GiB "
            f"usable={payload['usable_memory_gib']:.2f} GiB "
            f"ratio={payload['memory_utilization_ratio']:.3f}"
        ),
        (
            "Search summary: "
            f"enumerated={payload['enumerated_candidate_count']} "
            f"evaluated={payload['evaluated_candidate_count']} "
            f"feasible={payload['feasible_candidate_count']}"
        ),
        "",
        "Top feasible strategies:",
    ]

    feasible_records = [record for record in payload["candidates"] if record["feasible"]]
    if not feasible_records:
        lines.append("  None")
        return "\n".join(lines) + "\n"

    header = (
        "  rank  parallel_tag            score        "
        "peak(GiB)  margin(GiB)  penalty"
    )
    lines.append(header)
    for index, record in enumerate(feasible_records[:topk], start=1):
        score = (
            record["cluster_tokens_per_s"]
            if record["cluster_tokens_per_s"] is not None
            else record["coarse_throughput_score"]
        )
        lines.append(
            f"  {index:<4}  {record['parallel_tag']:<22} "
            f"{score:>10.2f}  "
            f"{record['peak_gib']:>9.2f}  "
            f"{record['memory_margin_gib']:>11.2f}  "
            f"{record['coarse_latency_penalty']:>7.3f}"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: Sequence[Dict]) -> None:
    if not records:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search tp/ep/pp/dp strategies under memory constraints for one HBF/SimAI "
            "inference workload. seq_length == 1 is treated as decode; otherwise prefill."
        )
    )
    parser.add_argument("--config-file", required=True, help="Workload config JSON.")
    parser.add_argument(
        "--inference-phase",
        choices=("auto", "prefill", "decode"),
        default="auto",
        help="Use auto to infer from seq_length. Default is auto.",
    )
    parser.add_argument("--num-nodes", required=True, type=int, help="Number of machines.")
    parser.add_argument(
        "--gpus-per-node",
        required=True,
        type=int,
        help="Number of GPUs on each machine.",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        required=True,
        type=float,
        help="Physical GPU memory capacity in GiB.",
    )
    parser.add_argument(
        "--memory-utilization-ratio",
        type=float,
        default=0.9,
        help="Usable memory ratio for capacity filtering. Default is 0.9.",
    )
    parser.add_argument(
        "--candidate-tp",
        help="Optional comma-separated tp candidates. Default is auto enumeration.",
    )
    parser.add_argument(
        "--candidate-ep",
        help="Optional comma-separated ep candidates. Default is auto enumeration.",
    )
    parser.add_argument(
        "--candidate-pp",
        help="Optional comma-separated pp candidates. Default is auto enumeration.",
    )
    parser.add_argument(
        "--active-seqs",
        "--decode-active-seqs",
        dest="active_seqs",
        type=int,
        help="Concurrent active decode sequences on one GPU. Only used for decode.",
    )
    parser.add_argument("--avg-live-tokens", "--decode-avg-live-tokens", dest="avg_live_tokens", type=float)
    parser.add_argument("--p90-live-tokens", "--decode-p90-live-tokens", dest="p90_live_tokens", type=float)
    parser.add_argument("--p99-live-tokens", "--decode-p99-live-tokens", dest="p99_live_tokens", type=float)
    parser.add_argument("--max-live-tokens", "--decode-max-live-tokens", dest="max_live_tokens", type=float)
    parser.add_argument(
        "--live-tokens-file",
        "--decode-live-tokens-file",
        dest="live_tokens_file",
        help="Optional decode live-token snapshots JSON/text file.",
    )
    parser.add_argument(
        "--step-seq-length",
        "--decode-step-seq-length",
        dest="step_seq_length",
        type=int,
        default=1,
        help="Decode tokens processed per step. Default is 1.",
    )
    parser.add_argument(
        "--paged-kv-workspace-mib",
        type=float,
        default=0.0,
        help="Optional paged-KV runtime workspace to reserve per GPU. Only used for decode.",
    )
    parser.add_argument("--weight-bytes", type=int, default=None)
    parser.add_argument("--activation-bytes", type=int, default=None)
    parser.add_argument(
        "--moe-load-balance-factor",
        type=float,
        default=1.0,
        help="Scale local routed tokens to model MoE imbalance.",
    )
    parser.add_argument(
        "--latency-table",
        help="Optional CSV/JSON table keyed by parallel_tag with phase latency in ms.",
    )
    parser.add_argument("--latency-budget-ms", type=float)
    parser.add_argument(
        "--max-coarse-latency-penalty",
        type=float,
        help="Optional coarse penalty threshold used as a simple latency guardrail.",
    )
    parser.add_argument("--topk", type=int, default=20, help="How many top feasible rows to show.")
    parser.add_argument(
        "--output-file",
        help=(
            "Optional output file. "
            "Use .json for structured output, .csv for flat rows, otherwise text."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_nodes < 1 or args.gpus_per_node < 1:
        raise ValueError("num-nodes and gpus-per-node must be >= 1")
    if not (0.0 < args.memory_utilization_ratio <= 1.0):
        raise ValueError("memory-utilization-ratio must be in (0, 1]")
    if args.step_seq_length < 1:
        raise ValueError("step-seq-length must be >= 1")

    total_gpus = args.num_nodes * args.gpus_per_node
    usable_memory_bytes = int(round(args.gpu_memory_gib * GIB * args.memory_utilization_ratio))
    config_path = Path(args.config_file).resolve()
    base_config = load_and_prepare_config(config_path)
    inference_phase = infer_inference_phase(base_config, args.inference_phase)

    tp_candidates = parse_int_list(args.candidate_tp, "--candidate-tp")
    ep_candidates = parse_int_list(args.candidate_ep, "--candidate-ep")
    pp_candidates = parse_int_list(args.candidate_pp, "--candidate-pp")
    if tp_candidates is None:
        tp_candidates = auto_tp_candidates(base_config, total_gpus)
    if ep_candidates is None:
        ep_candidates = auto_ep_candidates(base_config, total_gpus)
    if pp_candidates is None:
        pp_candidates = auto_pp_candidates(base_config, total_gpus)

    latency_table = load_latency_table(
        Path(args.latency_table).resolve() if args.latency_table else None
    )

    decode_scenarios = []
    decode_active_seqs: Optional[int] = None
    workspace_bytes = int(round(args.paged_kv_workspace_mib * 1024 * 1024))
    if inference_phase == "decode":
        decode_active_seqs = args.active_seqs or base_config["micro_batch"]
        decode_scenarios = build_decode_scenarios(
            active_seqs=decode_active_seqs,
            avg_live_tokens=args.avg_live_tokens,
            p90_live_tokens=args.p90_live_tokens,
            p99_live_tokens=args.p99_live_tokens,
            max_live_tokens=args.max_live_tokens,
            live_tokens_file=(
                Path(args.live_tokens_file).resolve() if args.live_tokens_file else None
            ),
        )
        if not decode_scenarios:
            raise ValueError(
                "decode search requires live-token scenarios. "
                "Use --avg-live-tokens/--p90-live-tokens/--p99-live-tokens/"
                "--max-live-tokens or --live-tokens-file."
            )

    candidates: List[Dict] = []
    enumerated_candidate_count = 0
    evaluated_candidate_count = 0

    for tp in tp_candidates:
        for ep in ep_candidates:
            for pp in pp_candidates:
                enumerated_candidate_count += 1
                if total_gpus % (tp * ep * pp) != 0:
                    continue
                dp = total_gpus // (tp * ep * pp)

                try:
                    config = apply_parallelism(
                        base_config,
                        tp=tp,
                        ep=ep,
                        pp=pp,
                        dp=dp,
                    )
                except ValueError:
                    continue

                evaluated_candidate_count += 1
                parallel_tag = build_parallel_tag(config)
                selected_decode_scenario: Optional[str] = None
                selected_decode_active_seqs: Optional[int] = None
                selected_decode_total_live_tokens: Optional[int] = None

                if inference_phase == "prefill":
                    phase_result = estimate_prefill_memory(
                        config,
                        weight_bytes=args.weight_bytes,
                        activation_bytes=args.activation_bytes,
                        moe_load_balance_factor=args.moe_load_balance_factor,
                    )
                    peak_bytes = phase_result["peak_stage_total_peak_bytes"]
                else:
                    phase_result = estimate_decode_memory(
                        config,
                        scenarios=decode_scenarios,
                        weight_bytes=args.weight_bytes,
                        activation_bytes=args.activation_bytes,
                        moe_load_balance_factor=args.moe_load_balance_factor,
                        step_seq_length=args.step_seq_length,
                        workspace_bytes=workspace_bytes,
                    )
                    peak_bytes = phase_result["overall_peak_total_peak_bytes"]
                    selected_decode_scenario = phase_result["overall_peak_scenario"].name
                    selected_decode_active_seqs = phase_result["overall_peak_scenario"].active_seqs
                    selected_decode_total_live_tokens = (
                        phase_result["overall_peak_scenario"].total_live_tokens
                    )

                memory_feasible = peak_bytes <= usable_memory_bytes

                coarse_penalty = coarse_latency_penalty(tp, ep, pp, args.gpus_per_node)
                coarse_penalty_feasible = True
                if (
                    args.max_coarse_latency_penalty is not None
                    and coarse_penalty > args.max_coarse_latency_penalty
                ):
                    coarse_penalty_feasible = False

                latency_row = latency_table.get(parallel_tag)
                latency_ms = lookup_latency_ms(latency_row)

                reject_reasons: List[str] = []
                if not memory_feasible:
                    reject_reasons.append("memory")
                if not coarse_penalty_feasible:
                    reject_reasons.append("coarse_latency_penalty")

                latency_feasible = True
                if args.latency_budget_ms is not None:
                    if latency_ms is None:
                        latency_feasible = False
                        reject_reasons.append("missing_latency")
                    elif latency_ms > args.latency_budget_ms:
                        latency_feasible = False
                        reject_reasons.append("latency")

                if inference_phase == "prefill":
                    tokens_per_replica = config["micro_batch"] * config["seq_length"]
                else:
                    tokens_per_replica = decode_active_seqs * args.step_seq_length
                coarse_throughput_score = dp * tokens_per_replica / coarse_penalty
                cluster_tokens_per_s = (
                    dp * tokens_per_replica / (latency_ms / 1000.0)
                    if latency_ms not in (None, 0.0)
                    else None
                )

                memory_margin_bytes = usable_memory_bytes - peak_bytes
                candidate = {
                    "parallel_tag": parallel_tag,
                    "phase": inference_phase,
                    "tp": tp,
                    "ep": ep,
                    "pp": pp,
                    "dp": dp,
                    "peak_bytes": peak_bytes,
                    "usable_memory_bytes": usable_memory_bytes,
                    "memory_margin_bytes": memory_margin_bytes,
                    "latency_ms": latency_ms,
                    "coarse_latency_penalty": coarse_penalty,
                    "coarse_throughput_score": coarse_throughput_score,
                    "cluster_tokens_per_s": cluster_tokens_per_s,
                    "selected_decode_scenario": selected_decode_scenario,
                    "selected_decode_active_seqs": selected_decode_active_seqs,
                    "selected_decode_total_live_tokens": selected_decode_total_live_tokens,
                    "memory_feasible": memory_feasible,
                    "latency_feasible": latency_feasible,
                    "coarse_penalty_feasible": coarse_penalty_feasible,
                    "reject_reasons": reject_reasons,
                }
                candidate["feasible"] = (
                    candidate["memory_feasible"]
                    and candidate["latency_feasible"]
                    and candidate["coarse_penalty_feasible"]
                )
                candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            not candidate["feasible"],
            -ranking_score(candidate),
            candidate["coarse_latency_penalty"],
            -candidate["memory_margin_bytes"],
        )
    )
    candidate_rows = [candidate_record(candidate) for candidate in candidates]
    feasible_candidate_count = sum(1 for row in candidate_rows if row["feasible"])

    payload = {
        "config_file": str(config_path),
        "inference_phase": inference_phase,
        "micro_batch": base_config["micro_batch"],
        "seq_length": base_config["seq_length"],
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "total_gpus": total_gpus,
        "gpu_memory_gib": args.gpu_memory_gib,
        "memory_utilization_ratio": args.memory_utilization_ratio,
        "usable_memory_gib": usable_memory_bytes / GIB,
        "decode_scenarios": [
            {
                "name": scenario.name,
                "active_seqs": scenario.active_seqs,
                "total_live_tokens": scenario.total_live_tokens,
                "avg_live_tokens_per_seq": scenario.avg_live_tokens_per_seq,
                "source": scenario.source,
            }
            for scenario in decode_scenarios
        ],
        "enumerated_candidate_count": enumerated_candidate_count,
        "evaluated_candidate_count": evaluated_candidate_count,
        "feasible_candidate_count": feasible_candidate_count,
        "candidates": candidate_rows,
    }
    text_report = render_text_report(payload, topk=args.topk)
    stdout_content = json.dumps(payload, indent=2) if args.json else text_report
    print(stdout_content)

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif suffix == ".csv":
            write_csv(output_path, candidate_rows)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text_report, encoding="utf-8")
        print(f"Saved result to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
