#!/usr/bin/env python3

"""Search stage-level strategies for one fixed per-replica HBF/SimAI workload.

This script bridges:
- search outputs or feasible candidate sets
- workload generation for each candidate strategy
- SimAI analytical execution
- result aggregation into a summary table

Stage time excludes request queueing, so it is not a TTFT or TPOT estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from memory_model import apply_parallelism, load_and_prepare_config


SCRIPT_DIR = Path(__file__).resolve().parent
SIMAI_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = (
    SCRIPT_DIR
    / "scripts"
    / "inference_configs"
    / "compute_cache"
    / "bs32_seq256"
    / "qwen3moe_235B_A22B_tp1_ep32_bs32_seq256.json"
)
DEFAULT_OUTPUT_ROOT = SIMAI_ROOT / "results" / "outputs" / "HBF_topk_validation"


def input_fingerprint(paths: Sequence[Path], settings: Dict) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in sorted(paths):
        digest.update(str(path.name).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def shape_from_config(config: Dict) -> str:
    return f"bs{config['micro_batch']}_seq{config['seq_length']}"


def parse_latency_dir_name(path: Path) -> int:
    match = re.search(r"latency(\d+)", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"failed to parse latency from: {path}")
    return int(match.group(1))


def parse_leading_number(raw_value: str) -> float:
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", raw_value)
    if not match:
        raise ValueError(f"failed to parse numeric value from: {raw_value!r}")
    return float(match.group(1))


def parse_int_list(raw_value: Optional[str], flag_name: str) -> Optional[List[int]]:
    if raw_value is None:
        return None
    values: List[int] = []
    for token in raw_value.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"{flag_name} values must be >= 0")
        values.append(value)
    if not values:
        raise ValueError(f"{flag_name} did not contain any valid integers")
    return sorted(set(values))


def ensure_prefill_shape(config: Dict) -> None:
    if int(config["seq_length"]) <= 1:
        raise ValueError(
            "this bridge currently targets prefill workloads only; "
            f"got seq_length={config['seq_length']}"
        )


def run_command(command: Sequence[str], *, cwd: Path, log_file: Optional[Path] = None) -> None:
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                list(command),
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        result = subprocess.run(list(command), cwd=str(cwd), text=True)
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"command failed with exit code {result.returncode}: {joined}")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_search_payload(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_topk_feasible_candidates(payload: Dict, topk: int) -> List[Dict]:
    feasible_rows = [row for row in payload["candidates"] if row["feasible"]]
    return feasible_rows[:topk]


def pick_all_feasible_candidates(payload: Dict) -> List[Dict]:
    return [row for row in payload["candidates"] if row["feasible"]]


def candidate_config_filename(base_config_path: Path, parallel_tag: str, shape: str) -> str:
    stem = base_config_path.stem
    stem = re.sub(r"_tp\d+_ep\d+_", "_", stem)
    return f"{stem}_{parallel_tag}_{shape}.json"


def materialize_candidate_configs(
    *,
    base_config_path: Path,
    base_config: Dict,
    candidates: Sequence[Dict],
    config_output_dir: Path,
    workload_output_root: Path,
) -> Dict[str, Path]:
    config_output_dir.mkdir(parents=True, exist_ok=True)
    shape = shape_from_config(base_config)
    output_paths: Dict[str, Path] = {}
    for candidate in candidates:
        config = apply_parallelism(
            base_config,
            tp=int(candidate["tp"]),
            ep=int(candidate["ep"]),
            pp=int(candidate["pp"]),
            dp=int(candidate["dp"]),
        )
        config["result_dir"] = str(workload_output_root / candidate["parallel_tag"])
        output_path = config_output_dir / candidate_config_filename(
            base_config_path,
            candidate["parallel_tag"],
            shape,
        )
        output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        output_paths[candidate["parallel_tag"]] = output_path
    return output_paths


def stage_filtered_aiob_inputs(
    *,
    source_dir: Path,
    staged_dir: Path,
    latency_values: Optional[Sequence[int]],
) -> Path:
    if latency_values is None:
        return source_dir

    staged_dir.mkdir(parents=True, exist_ok=True)
    for existing in staged_dir.glob("*.txt"):
        existing.unlink()

    copied = 0
    for source_file in sorted(source_dir.glob("*.txt")):
        try:
            latency = parse_latency_dir_name(Path(source_file.stem))
        except ValueError:
            match = re.search(r"Latency(\d+)", source_file.name, re.IGNORECASE)
            if not match:
                continue
            latency = int(match.group(1))
        if latency not in latency_values:
            continue
        shutil.copy2(source_file, staged_dir / source_file.name)
        copied += 1

    if copied == 0:
        raise RuntimeError(
            f"no AIOB input files matched latency_values={list(latency_values)} in {source_dir}"
        )
    return staged_dir


def workload_dir_ready(path: Path) -> bool:
    return any(path.glob("latency*/*.txt"))


def simai_dir_ready(path: Path) -> bool:
    return any(path.glob("latency*/*_EndToEnd.csv"))


def run_search(
    *,
    config_file: Path,
    num_nodes: int,
    gpus_per_node: int,
    gpu_memory_gib: float,
    memory_utilization_ratio: float,
    candidate_tp: Optional[str],
    candidate_ep: Optional[str],
    candidate_pp: Optional[str],
    latency_table: Optional[Path],
    latency_budget_ms: Optional[float],
    max_coarse_latency_penalty: Optional[float],
    output_json: Path,
    log_file: Path,
) -> Dict:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "search_parallel_strategy.py"),
        "--config-file",
        str(config_file),
        "--num-nodes",
        str(num_nodes),
        "--gpus-per-node",
        str(gpus_per_node),
        "--gpu-memory-gib",
        str(gpu_memory_gib),
        "--memory-utilization-ratio",
        str(memory_utilization_ratio),
        "--output-file",
        str(output_json),
    ]
    if candidate_tp:
        command.extend(["--candidate-tp", candidate_tp])
    if candidate_ep:
        command.extend(["--candidate-ep", candidate_ep])
    if candidate_pp:
        command.extend(["--candidate-pp", candidate_pp])
    if latency_table:
        command.extend(["--latency-table", str(latency_table)])
    if latency_budget_ms is not None:
        command.extend(["--latency-budget-ms", str(latency_budget_ms)])
    if max_coarse_latency_penalty is not None:
        command.extend(["--max-coarse-latency-penalty", str(max_coarse_latency_penalty)])
    command.append("--json")
    run_command(command, cwd=SCRIPT_DIR, log_file=log_file)
    return load_search_payload(output_json)


def run_workload_generation(
    *,
    aiob_input_dir: Path,
    config_file: Path,
    output_dir: Path,
    log_file: Path,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "batch_generate_workload.py"),
        "--input_dir",
        str(aiob_input_dir),
        "--config_file",
        str(config_file),
        "--model_output_prefix",
        "Qwen3-Moe-235B",
        "--output_base_dir",
        str(output_dir),
    ]
    run_command(command, cwd=SCRIPT_DIR, log_file=log_file)


def run_simai_analytical(
    *,
    workload_dir: Path,
    output_dir: Path,
    log_file: Path,
    num_gpus: int,
    gpus_per_node: int,
    nvlink_bandwidth: float,
    nic_bandwidth: float,
    nodes_per_switch: int,
) -> None:
    command = [
        sys.executable,
        str(SIMAI_ROOT / "run_simai_batch.py"),
        "--workLoadInput_dir",
        str(workload_dir),
        "--output_dir",
        str(output_dir),
        "--num-gpus", str(num_gpus),
        "--gpus-per-node", str(gpus_per_node),
        "--nvlink-bandwidth", str(nvlink_bandwidth),
        "--nic-bandwidth", str(nic_bandwidth),
        "--nodes-per-switch", str(nodes_per_switch),
    ]
    run_command(command, cwd=SIMAI_ROOT, log_file=log_file)


def parse_end_to_end_csv(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        headers = [item.strip() for item in next(reader)]
        values = [item.strip() for item in next(reader)]
    row = {header: value for header, value in zip(headers, values)}
    return {
        "expose_dp_comm": parse_leading_number(row["Expose DP comm"]),
        "expose_dp_ep_comm": parse_leading_number(row["Expose DP_EP comm"]),
        "expose_tp_comm": parse_leading_number(row["Expose TP comm"]),
        "expose_ep_comm": parse_leading_number(row["Expose_EP_comm"]),
        "expose_pp_comm": parse_leading_number(row["Expose_PP_comm"]),
        "bubble_time": parse_leading_number(row["bubble time"]),
        "total_comp": parse_leading_number(row["total comp"]),
        "total_exposed_comm": parse_leading_number(row["total exposed comm"]),
        "total_time": parse_leading_number(row["Total time"]),
    }


def collect_candidate_latency_rows(
    *,
    simai_output_dir: Path,
    candidate: Dict,
    tokens_per_replica: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for csv_file in sorted(simai_output_dir.glob("latency*/*_EndToEnd.csv")):
        latency = parse_latency_dir_name(csv_file.parent)
        metrics = parse_end_to_end_csv(csv_file)
        total_time = metrics["total_time"]
        rows.append(
            {
                "parallel_tag": candidate["parallel_tag"],
                "tp": candidate["tp"],
                "ep": candidate["ep"],
                "pp": candidate["pp"],
                "dp": candidate["dp"],
                "search_rank": candidate["search_rank"],
                "search_peak_gib": candidate["peak_gib"],
                "search_margin_gib": candidate["memory_margin_gib"],
                "search_score": candidate["coarse_throughput_score"],
                "latency_tag": latency,
                "total_comp": metrics["total_comp"],
                "total_exposed_comm": metrics["total_exposed_comm"],
                "total_time": total_time,
                "expose_ep_comm": metrics["expose_ep_comm"],
                "cluster_tokens_per_time_unit": (
                    int(candidate["dp"]) * tokens_per_replica / total_time if total_time else 0.0
                ),
                "end_to_end_csv": str(csv_file),
            }
        )
    return rows


def build_candidate_summary(latency_rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in latency_rows:
        grouped.setdefault(str(row["parallel_tag"]), []).append(row)

    summaries: List[Dict] = []
    for parallel_tag, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda item: int(item["latency_tag"]))
        total_times = [float(item["total_time"]) for item in sorted_rows]
        throughputs = [float(item["cluster_tokens_per_time_unit"]) for item in sorted_rows]
        first = sorted_rows[0]
        summaries.append(
            {
                "parallel_tag": parallel_tag,
                "tp": first["tp"],
                "ep": first["ep"],
                "pp": first["pp"],
                "dp": first["dp"],
                "search_rank": first["search_rank"],
                "search_peak_gib": first["search_peak_gib"],
                "search_margin_gib": first["search_margin_gib"],
                "latency_count": len(sorted_rows),
                "best_total_time": min(total_times),
                "mean_total_time": mean(total_times),
                "worst_total_time": max(total_times),
                "best_cluster_tokens_per_time_unit": max(throughputs),
                "mean_cluster_tokens_per_time_unit": mean(throughputs),
            }
        )

    return summaries


def rank_candidate_summary(
    summaries: Sequence[Dict],
    *,
    throughput_stat: str,
    latency_stat: str,
) -> List[Dict]:
    throughput_key = f"{throughput_stat}_cluster_tokens_per_time_unit"
    latency_key = f"{latency_stat}_total_time"

    ranked = [dict(row) for row in summaries]
    ranked.sort(
        key=lambda row: (
            -float(row[throughput_key]),
            float(row[latency_key]),
            int(row["search_rank"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_throughput_stat"] = throughput_stat
        row["selection_latency_stat"] = latency_stat
        row["selection_throughput"] = float(row[throughput_key])
        row["selection_latency"] = float(row[latency_key])
        row["selection_rank"] = rank
    return ranked


def select_by_objective(
    latency_rows: Sequence[Dict], *, objective: str,
    max_stage_time: Optional[float],
) -> List[Dict]:
    """Select independently for each hardware latency configuration."""
    selected: List[Dict] = []
    for latency in sorted({row["latency_tag"] for row in latency_rows}):
        eligible = [dict(row) for row in latency_rows
                    if row["latency_tag"] == latency and
                    (max_stage_time is None or row["total_time"] <= max_stage_time)]
        if objective == "max_throughput":
            eligible.sort(key=lambda row: (-row["cluster_tokens_per_time_unit"],
                                           row["total_time"], row["parallel_tag"]))
        else:
            eligible.sort(key=lambda row: (row["total_time"],
                                           -row["cluster_tokens_per_time_unit"],
                                           row["parallel_tag"]))
        for rank, row in enumerate(eligible, 1):
            row["selection_rank"] = rank
            row["objective"] = objective
            selected.append(row)
    return selected


def render_topn_report(
    summaries: Sequence[Dict],
    *,
    throughput_stat: str,
    latency_stat: str,
    topn: int,
) -> str:
    lines = [
        "Top strategies by throughput-first, latency-second:",
        (
            f"  throughput_stat={throughput_stat} "
            f"latency_stat={latency_stat} "
            f"topn={topn}"
        ),
    ]
    if not summaries:
        lines.append("  None")
        return "\n".join(lines)

    lines.append(
        "  rank  parallel_tag            throughput        latency        tp  ep  pp  dp"
    )
    for row in summaries[:topn]:
        lines.append(
            "  "
            f"{int(row['selection_rank']):>4}  "
            f"{str(row['parallel_tag']):<22}  "
            f"{float(row['selection_throughput']):>14.6f}  "
            f"{float(row['selection_latency']):>12.2f}  "
            f"{int(row['tp']):>2}  "
            f"{int(row['ep']):>2}  "
            f"{int(row['pp']):>2}  "
            f"{int(row['dp']):>2}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run topK candidate validation for one HBF/SimAI workload by chaining "
            "search -> config materialization -> workload generation -> SimAI analytical."
        )
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG),
        help="Base workload config JSON. Default is bs32_seq256 ep32.",
    )
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--gpu-memory-gib", type=float, default=40.0)
    parser.add_argument("--memory-utilization-ratio", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=0,
                        help="Coarse candidate cap; 0 (default) evaluates every capacity-feasible strategy.")
    parser.add_argument(
        "--all-feasible",
        action="store_true",
        help="Validate every feasible candidate instead of truncating to topk.",
    )
    parser.add_argument("--candidate-tp")
    parser.add_argument("--candidate-ep")
    parser.add_argument("--candidate-pp")
    parser.add_argument("--objective", choices=("max_throughput", "min_latency"),
                        default="max_throughput",
                        help="Stage-level objective; micro_batch is fixed per DP replica.")
    parser.add_argument("--max-stage-time", type=float,
                        help="Maximum SimAI stage time, in simulator time units, per latency scenario.")
    parser.add_argument("--nvlink-bandwidth", type=float, default=360)
    parser.add_argument("--nic-bandwidth", type=float, default=48.5)
    parser.add_argument("--nodes-per-switch", type=int, default=8)
    parser.add_argument("--latency-table")
    parser.add_argument("--latency-budget-ms", type=float)
    parser.add_argument("--max-coarse-latency-penalty", type=float)
    parser.add_argument(
        "--aiob-input-dir",
        help="Optional AIOB input dir. Default is inferred from bs/seq shape.",
    )
    parser.add_argument(
        "--latency-values",
        help="Optional comma-separated AIOB latency tags to keep, e.g. 100 or 100,150.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional validation output root. Default is under SimAI/results/outputs/HBF_topk_validation.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run only search and config materialization, skip workload/SimAI stages.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip workload or SimAI stages if their outputs already exist.",
    )
    parser.add_argument(
        "--selection-topn",
        type=int,
        default=3,
        help="How many final strategies to report after full validation. Default is 3.",
    )
    parser.add_argument(
        "--selection-throughput-stat",
        choices=("best", "mean"),
        default="mean",
        help="Which throughput summary statistic to optimize first. Default is mean.",
    )
    parser.add_argument(
        "--selection-latency-stat",
        choices=("best", "mean", "worst"),
        default="mean",
        help="Which latency summary statistic to use as the tiebreaker. Default is mean.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.topk < 0 or args.selection_topn < 1:
        parser.error("topk must be nonnegative and selection-topn must be positive")
    if args.max_stage_time is not None and args.max_stage_time <= 0:
        parser.error("max-stage-time must be positive")

    base_config_path = Path(args.config_file).resolve()
    base_config = load_and_prepare_config(base_config_path)
    ensure_prefill_shape(base_config)
    shape = shape_from_config(base_config)

    aiob_input_dir = (
        Path(args.aiob_input_dir).resolve()
        if args.aiob_input_dir
        else SCRIPT_DIR / "results" / "aiob_outputs" / "HBF" / f"accel_sim_{shape}"
    )
    if not aiob_input_dir.is_dir():
        raise FileNotFoundError(f"AIOB input dir not found: {aiob_input_dir}")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else DEFAULT_OUTPUT_ROOT
        / shape
        / f"nodes{args.num_nodes}_gpus{args.gpus_per_node}_A100_{int(args.gpu_memory_gib)}GB"
    )
    search_dir = output_root / "search"
    config_dir = output_root / "configs"
    workload_root = output_root / "workloads"
    simai_root = output_root / "simai"
    summary_dir = output_root / "summary"
    log_dir = output_root / "logs"
    staged_aiob_root = output_root / "staged_aiob_inputs"

    latency_values = parse_int_list(args.latency_values, "--latency-values")
    effective_aiob_input_dir = stage_filtered_aiob_inputs(
        source_dir=aiob_input_dir,
        staged_dir=staged_aiob_root,
        latency_values=latency_values,
    )
    fingerprint = input_fingerprint(
        [base_config_path, Path(__file__).resolve(),
         SCRIPT_DIR / "batch_generate_workload.py",
         SCRIPT_DIR / "search_parallel_strategy.py", SCRIPT_DIR / "memory_model.py",
         SCRIPT_DIR / "estimate_prefill_memory.py",
         SCRIPT_DIR / "workload_generator" / "HBF_SimAI_inference_workload_generator.py",
         SIMAI_ROOT / "run_simai_batch.py", SIMAI_ROOT / "bin" / "SimAI_analytical",
         *effective_aiob_input_dir.glob("*.txt")],
        {"num_nodes": args.num_nodes, "gpus_per_node": args.gpus_per_node,
         "gpu_memory_gib": args.gpu_memory_gib,
         "memory_utilization_ratio": args.memory_utilization_ratio,
         "candidate_tp": args.candidate_tp, "candidate_ep": args.candidate_ep,
         "candidate_pp": args.candidate_pp,
         "nvlink_bandwidth": args.nvlink_bandwidth,
         "nic_bandwidth": args.nic_bandwidth,
         "nodes_per_switch": args.nodes_per_switch,
         "latency_values": latency_values},
    )
    manifest_path = summary_dir / "run_manifest.json"
    if args.skip_existing and simai_root.exists():
        if not manifest_path.exists():
            raise ValueError("--skip-existing requires a matching run manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("input_fingerprint") != fingerprint:
            raise ValueError("--skip-existing inputs changed; use a new output root")

    search_output_json = search_dir / "search_results.json"
    search_payload = run_search(
        config_file=base_config_path,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_gib=args.gpu_memory_gib,
        memory_utilization_ratio=args.memory_utilization_ratio,
        candidate_tp=args.candidate_tp,
        candidate_ep=args.candidate_ep,
        candidate_pp=args.candidate_pp,
        latency_table=Path(args.latency_table).resolve() if args.latency_table else None,
        latency_budget_ms=args.latency_budget_ms,
        max_coarse_latency_penalty=args.max_coarse_latency_penalty,
        output_json=search_output_json,
        log_file=log_dir / "search.log",
    )

    if args.all_feasible or args.topk == 0:
        topk_candidates = pick_all_feasible_candidates(search_payload)
    else:
        topk_candidates = pick_topk_feasible_candidates(search_payload, args.topk)
    if not topk_candidates:
        raise RuntimeError("search produced no feasible candidates")

    for search_rank, candidate in enumerate(topk_candidates, start=1):
        candidate["search_rank"] = search_rank

    topk_path = search_dir / "topk_candidates.json"
    write_json(topk_path, {"candidates": topk_candidates})
    write_csv(search_dir / "topk_candidates.csv", topk_candidates)

    candidate_config_paths = materialize_candidate_configs(
        base_config_path=base_config_path,
        base_config=base_config,
        candidates=topk_candidates,
        config_output_dir=config_dir,
        workload_output_root=workload_root,
    )

    if args.search_only:
        print(f"Search finished. TopK candidates saved under: {search_dir}")
        print(f"Candidate configs saved under: {config_dir}")
        return 0

    tokens_per_replica = int(base_config["micro_batch"]) * int(base_config["seq_length"])
    latency_rows: List[Dict] = []
    for candidate in topk_candidates:
        parallel_tag = str(candidate["parallel_tag"])
        candidate_workload_dir = workload_root / parallel_tag
        candidate_simai_dir = simai_root / parallel_tag

        if not (args.skip_existing and workload_dir_ready(candidate_workload_dir)):
            run_workload_generation(
                aiob_input_dir=effective_aiob_input_dir,
                config_file=candidate_config_paths[parallel_tag],
                output_dir=candidate_workload_dir,
                log_file=log_dir / f"workload_{parallel_tag}.log",
            )

        if not (args.skip_existing and simai_dir_ready(candidate_simai_dir)):
            run_simai_analytical(
                workload_dir=candidate_workload_dir,
                output_dir=candidate_simai_dir,
                log_file=log_dir / f"simai_{parallel_tag}.log",
                num_gpus=args.num_nodes * args.gpus_per_node,
                gpus_per_node=args.gpus_per_node,
                nvlink_bandwidth=args.nvlink_bandwidth,
                nic_bandwidth=args.nic_bandwidth,
                nodes_per_switch=args.nodes_per_switch,
            )

        latency_rows.extend(
            collect_candidate_latency_rows(
                simai_output_dir=candidate_simai_dir,
                candidate=candidate,
                tokens_per_replica=tokens_per_replica,
            )
        )

    candidate_summary = build_candidate_summary(latency_rows)
    ranked_summary = rank_candidate_summary(
        candidate_summary,
        throughput_stat=args.selection_throughput_stat,
        latency_stat=args.selection_latency_stat,
    )
    for row in latency_rows:
        row["stage_time_feasible"] = (
            args.max_stage_time is None or row["total_time"] <= args.max_stage_time
        )
    objective_rankings = select_by_objective(
        latency_rows, objective=args.objective,
        max_stage_time=args.max_stage_time,
    )
    selected_topn = [row for row in objective_rankings
                     if row["selection_rank"] <= args.selection_topn]
    write_csv(summary_dir / "candidate_latency_rows.csv", latency_rows)
    write_csv(summary_dir / "objective_rankings.csv", objective_rankings)
    write_csv(summary_dir / "candidate_summary.csv", ranked_summary)
    write_csv(summary_dir / "selected_top_strategies.csv", selected_topn)
    write_json(
        summary_dir / "run_manifest.json",
        {
            "config_file": str(base_config_path),
            "input_fingerprint": fingerprint,
            "shape": shape,
            "num_nodes": args.num_nodes,
            "gpus_per_node": args.gpus_per_node,
            "gpu_memory_gib": args.gpu_memory_gib,
            "aiob_input_dir": str(aiob_input_dir),
            "effective_aiob_input_dir": str(effective_aiob_input_dir),
            "latency_values": latency_values,
            "output_root": str(output_root),
            "topk": args.topk,
            "all_feasible": args.all_feasible,
            "objective": args.objective,
            "objective_scope": "stage_execution_no_queueing",
            "workload_semantics": "fixed_micro_batch_per_dp_replica",
            "max_stage_time": args.max_stage_time,
            "nvlink_bandwidth": args.nvlink_bandwidth,
            "nic_bandwidth": args.nic_bandwidth,
            "nodes_per_switch": args.nodes_per_switch,
            "candidate_evaluation_exact": args.all_feasible or args.topk == 0,
            "selection_topn": args.selection_topn,
            "selection_throughput_stat": args.selection_throughput_stat,
            "selection_latency_stat": args.selection_latency_stat,
            "topk_candidates": topk_candidates,
            "selected_top_strategies": selected_topn,
        },
    )

    print(f"Validation finished. Output root: {output_root}")
    print(f"TopK candidates: {len(topk_candidates)}")
    print(f"Latency rows: {len(latency_rows)}")
    print(f"Summary file: {summary_dir / 'candidate_summary.csv'}")
    print(f"Selected top strategies: {summary_dir / 'selected_top_strategies.csv'}")
    for row in selected_topn:
        print(f"latency={row['latency_tag']} rank={row['selection_rank']} "
              f"strategy={row['parallel_tag']} stage_time={row['total_time']} "
              f"throughput={row['cluster_tokens_per_time_unit']:.8f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
