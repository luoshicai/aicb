"""CPU regression checks for the migrated HBF external-compute entry point."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("legacy AICB model imports require CPU PyTorch") from exc

from workload_generator.HBF_SimAI_inference_workload_generator import (
    SimAIWorkload, _get_aiob_compute_time,
)
from workload_generator.mocked_model.HBF_models.HBF_MockedQwen3 import (
    Qwen3MoeModel, Qwen3MoeParams,
)

ROOT = Path(__file__).resolve().parent
CACHE = dict(attention_norm=11, attention_layer=23, moe_norm=7,
             moe_route=13, moe_expert=47)


def workload(tp=1, layers=2):
    args = Qwen3MoeParams(str(ROOT / "scripts/inference_configs/HBF_qwen3_default.json"))
    args.num_hidden_layers = layers
    args.tensor_model_parallel_size = tp
    return SimAIWorkload(Qwen3MoeModel(args), args, dict(CACHE))


class LegacyWorkloadTests(unittest.TestCase):
    def test_independent_instances_without_cli_globals(self):
        a, b = workload(1, 2), workload(2, 3)
        a.workload_generate_aiob()
        b.workload_generate_aiob()
        self.assertEqual(len(a.workload), 10)
        self.assertEqual(len(b.workload), 15)
        self.assertEqual(a.workload[1].forward_comm, "NONE")
        self.assertEqual(b.workload[1].forward_comm, "ALLREDUCE")
        self.assertEqual(sum(x.forward_compute_time for x in a.workload), 2 * sum(CACHE.values()))

    def test_missing_and_invalid_timings_fail(self):
        for cache in [{}, dict(CACHE, moe_expert=0), dict(CACHE, moe_expert=-1),
                      dict(CACHE, moe_expert=float("nan")), dict(CACHE, moe_expert=True)]:
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                w = workload()
                w.compute_cache = cache
                w.workload_generate_aiob()
        with self.assertRaises(ValueError):
            SimAIWorkload(None, None)

    def test_config_aliases_retained(self):
        for name in ["HBF_qwen3_default.json", "HBF_qwen3_default_aiob.json"]:
            config = json.loads((ROOT / "scripts/inference_configs" / name).read_text())
            self.assertEqual(config["num_hidden_layers"], 94)
            self.assertEqual(config["torch_dtype"], "bfloat16")

    def test_shell_paths_and_missing_input(self):
        with tempfile.TemporaryDirectory(prefix="hbf input ") as tmp:
            tmp = Path(tmp)
            config = json.loads((ROOT / "scripts/inference_configs/HBF_qwen3_default.json").read_text())
            config.update(num_hidden_layers=2, result_dir=str(tmp / "output"))
            (tmp / "config file.json").write_text(json.dumps(config))
            timings = tmp / "time $(literal).txt"
            timings.write_text("".join(f"{stage}:\n    time_gpu_avg: {value}\n" for stage, value in
                [("atten_norm", 11), ("atten_core", 23), ("moe_norm", 7),
                 ("moe_route", 13), ("moe_expert", 47)]))
            command = ["bash", str(ROOT / "scripts/HBF_inference_workload_with_computeCache.sh"),
                       "-m", "qwen3-235B", "-c", "config file.json", "-a", timings.name]
            run = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = list((tmp / "output").glob("*.txt"))
            self.assertEqual(len(result), 1)
            rows = result[0].read_text().splitlines()
            self.assertEqual(int(rows[1]), 10)
            # External input is in microseconds; the workload stores nanoseconds.
            self.assertEqual(sum(int(r.split()[2]) for r in rows[2:]), 202000)
            command[-1] = "absent.txt"
            run = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
