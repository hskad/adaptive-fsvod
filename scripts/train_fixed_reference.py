"""Reproduce only the previously selected 300-update fixed reference models."""

import argparse
import shutil
from pathlib import Path

import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from run_revision import training_run


def reproduce(root):
    record_path = root / "revision_v2/replications.json"
    records = read_json(record_path) if record_path.exists() else []
    source = Path(__file__).resolve().parents[1] / "results/revision_v2/configs/fixed_tau060.json"
    target = root / "revision_v2/configs/fixed_tau060.json"
    if target.exists() and checksum(target) != checksum(source):
        raise ValueError("Existing reference configuration differs from the published selection")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    changed = False
    for seed in (42, 43, 44):
        existing = next((r for r in records if r["method"] == "fixed" and r["seed"] == seed), None)
        if existing is not None:
            if checksum(existing["checkpoint"]) != existing["checkpoint_sha256"]:
                raise ValueError("Existing reference checkpoint changed")
            print("Keep existing fixed reference, seed", seed, flush=True)
            continue
        row = training_run(root, target, "fixed_tau060", seed, study="revision_v2")
        records.append(dict(row, method="fixed"))
        changed = True
    if changed:
        write_json(record_path, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reproduce(args.experiment_dir.resolve())
