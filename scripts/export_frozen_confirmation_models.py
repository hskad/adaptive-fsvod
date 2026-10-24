"""Package the nine frozen heads for inference, with verified tensor round-trips."""

import argparse
import shutil
import zipfile
from pathlib import Path

import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.evaluate import load_detector
from run_larger_confirmation import STUDY, verify


def export(root):
    verify(root)
    directory = root / "exports/larger_confirmation_inference"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for row in read_json(root / STUDY / "models.json"):
        original, checkpoint = load_detector(row["checkpoint"])
        output = directory / f"{row['method']}_seed{row['seed']}.pt"
        inference = {k: v for k, v in checkpoint.items()
                     if k not in ("optimizer", "scheduler", "torch_rng", "python_rng", "cuda_rng")}
        inference["artifact_type"] = "inference-only; optimizer resume requires the full training checkpoint"
        inference["source_checkpoint_sha256"] = row["checkpoint_sha256"]
        torch.save(inference, output)
        restored, _ = load_detector(output)
        for key, value in original.state_dict().items():
            if not torch.equal(value, restored.state_dict()[key]):
                raise ValueError("Export changed a model tensor")
        records.append({"method": row["method"], "seed": row["seed"], "file": output.name,
                        "bytes": output.stat().st_size, "sha256": checksum(output),
                        "source_checkpoint_sha256": row["checkpoint_sha256"],
                        "training_budget": row["training_budget"], "checkpoint_step": checkpoint["step"],
                        "model_config": checkpoint["model_config"]})
        print("VERIFIED EXPORT", output.name, flush=True)
    write_json(directory / "manifest.json", {"models": records,
               "source_protocol_sha256": checksum(root / STUDY / "selection.json"),
               "note": "Head weights include the frozen training feature center when enabled. Download the pinned public OWLv2 encoder separately. These files support inference, not optimizer resume."})
    for name in ("confirmation.json", "episodes.json", "support_pool.json", "selection.json"):
        shutil.copyfile(root / STUDY / name, directory / name)
    (directory / "README.txt").write_text(
        "Nine inference checkpoints for https://github.com/hskad/adaptive-fsvod\n"
        "fixed_50 and centered_50: exactly 50 updates, schedule horizon 300, threshold 0.8.\n"
        "fixed_300: 300-update training budget, threshold 0.6, validation-selected checkpoint.\n"
        "Use python -m adaptive_fsvod.evaluate with a checkpoint, manifest, cache, episodes, and output.\n"
        "The frozen OWLv2 encoder is not bundled. All original model tensors were checked after reload.\n"
        "Optimizer state is omitted. Train from scratch to reproduce or use full source checkpoints to resume.\n",
        encoding="utf-8")
    archive = directory.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    result = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": checksum(archive),
              "models": records}
    write_json(archive.with_suffix(".manifest.json"), result)
    print("INFERENCE BUNDLE", archive, result["bytes"], result["sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export(args.experiment_dir.resolve())
