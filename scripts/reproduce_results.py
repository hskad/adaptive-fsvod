"""Evaluate all public heads with both scores and compare with the published AP."""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from adaptive_fsvod.data import checksum, read_json, write_json


def reproduce(root, threads=4, weights=None, tolerance=1e-6):
    repo = Path(__file__).resolve().parents[1]
    protocol = repo / "results/score_calibration/confirmation_protocol"
    base = root / "score_calibration/confirmation"
    for name in ("confirmation.json", "episodes.json"):
        if checksum(base / name) != checksum(protocol / name):
            raise ValueError("Published data or episodes differ")
    heads = read_json(repo / "results/score_calibration/inference_bundle.manifest.json")
    rows = []
    for seed in (42, 43, 44):
        checkpoint = (weights or root / "weights") / f"shared_seed{seed}.pt"
        if weights is None:
            row = next(x for x in heads["models"] if x["seed"] == seed)
            if checksum(checkpoint) != row["sha256"]:
                raise ValueError("Public checkpoint differs")
        for method, beta in (("baseline", 0), ("objectness", 1)):
            output = root / "reproduced" / f"{method}_seed{seed}"
            # Limit threads before the module runs, on Windows and Linux.
            code = "import runpy, torch; torch.set_num_threads(int(__import__('sys').argv.pop(1))); runpy.run_module('adaptive_fsvod.evaluate',run_name='__main__')"
            command = [sys.executable, "-u", "-c", code, str(threads),
                       "--manifest", str(base / "confirmation.json"), "--cache", str(base / "cache"),
                       "--episodes", str(base / "episodes.json"), "--checkpoint", str(checkpoint),
                       "--objectness-beta", str(beta), "--output", str(output), "--device", "cpu"]
            subprocess.run(command, check=True)
            metrics = read_json(output / "metrics.json")
            original = read_json(repo / f"results/score_calibration/evaluations/{method}_seed{seed}/metrics.json")
            errors = {key: metrics[key] - original[key] for key in ("AP", "AP50", "AP75")}
            rows.append({"method": method, "seed": seed, "checkpoint_sha256": checksum(checkpoint),
                         **{k: metrics[k] for k in ("AP", "AP50", "AP75")}, "metric_differences": errors})
            print("REPRODUCED", method, seed, metrics["AP"], "differences", errors, flush=True)
    summary = {method: {key: statistics.mean(r[key] for r in rows if r["method"] == method)
                        for key in ("AP", "AP50", "AP75")} for method in ("baseline", "objectness")}
    exact = all(abs(d) <= tolerance for r in rows for d in r["metric_differences"].values())
    record = {"rows": rows, "summary": summary, "matches_published_within_tolerance": exact,
              "AP_point_tolerance": tolerance, "source_summary_sha256": checksum(repo / "results/score_calibration/verified_summary.json"),
              "note": "Saved-feature CPU evaluation. Raw image extraction is a separate check."}
    write_json(root / "reproduced/summary.json", record)
    print("MEAN", summary, "MATCHES", exact, flush=True)
    if not exact:
        raise SystemExit("Scores differ from the published result; inspect reproduced/summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weights-dir", type=Path, help="Optional shared_seed*.pt files from fresh training")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Allowed AP-point difference")
    args = parser.parse_args()
    reproduce(args.experiment_dir.resolve(), args.threads, args.weights_dir, args.tolerance)
