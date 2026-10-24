"""Export shared inference heads and the two explicit detection-score presets."""

import argparse
import shutil
import zipfile
from pathlib import Path

import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.evaluate import load_detector
from run_score_calibration import STUDY, verify


def export(root):
    protocol = verify(root)
    selection = read_json(root / STUDY / "selection.json")
    directory = root / "exports/score_calibration_inference"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for row in protocol["models"]:
        model, checkpoint = load_detector(row["checkpoint"])
        keep = {key: value for key, value in checkpoint.items()
                if key not in ("optimizer", "scheduler", "torch_rng", "python_rng", "cuda_rng")}
        keep["artifact_type"] = "inference-only shared head; choose the score preset explicitly"
        keep["source_checkpoint_sha256"] = row["checkpoint_sha256"]
        path = directory / f"shared_seed{row['seed']}.pt"
        torch.save(keep, path)
        restored, _ = load_detector(path)
        for name, value in model.state_dict().items():
            if not torch.equal(value, restored.state_dict()[name]):
                raise ValueError("Export changed a model tensor")
        records.append({"seed": row["seed"], "file": path.name, "sha256": checksum(path),
                        "bytes": path.stat().st_size, "source_checkpoint_sha256": row["checkpoint_sha256"],
                        "training_budget": 300, "checkpoint_step": checkpoint["step"],
                        "model_config": checkpoint["model_config"]})
        print("VERIFIED SHARED HEAD", path.name, flush=True)
    presets = {"baseline": {"objectness_beta": 0., "score_floor": .001, "nms_iou": .5, "max_detections": 100},
               "objectness": {"objectness_beta": selection["selected_beta"], "score_floor": .001, "nms_iou": .5, "max_detections": 100}}
    for name, value in presets.items():
        write_json(directory / f"{name}_postprocessing.json", value)
    manifest = {"models": records, "postprocessing_presets": presets,
                "study_protocol_sha256": checksum(root / STUDY / "protocol.json"),
                "selection_sha256": checksum(root / STUDY / "selection.json"),
                "note": "Both methods share all trained tensors. Set --objectness-beta explicitly: 0 baseline, selected beta for the proposed score. Encoder and optimizer state are excluded."}
    write_json(directory / "manifest.json", manifest)
    shutil.copyfile(root / STUDY / "protocol.json", directory / "study_protocol.json")
    (directory / "README.txt").write_text(
        "Inference heads for https://github.com/hskad/adaptive-fsvod\n"
        "All model tensors were verified after export and reload. Both methods use the same heads.\n"
        "Run from the installed repository, with E set to the experiment directory:\n"
        "python -m adaptive_fsvod.evaluate --manifest E/score_calibration/confirmation/confirmation.json --cache E/score_calibration/confirmation/cache --episodes E/score_calibration/confirmation/episodes.json --checkpoint shared_seed42.pt --objectness-beta 0 --output E/eval_baseline_seed42 --device cpu\n"
        "Change --objectness-beta to 1 for the selected score; repeat seeds 43 and 44.\n"
        "For raw-image inference, use python -m adaptive_fsvod.infer with --frames, --support, --checkpoint, --output and --objectness-beta 1.\n"
        "The pinned public OWLv2 encoder downloads separately. These exports cannot resume an optimizer.\n",
        encoding="utf-8")
    archive = directory.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    record = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": checksum(archive), **manifest}
    write_json(archive.with_suffix(".manifest.json"), record)
    print("SHARED INFERENCE BUNDLE", archive, record["bytes"], record["sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export(args.experiment_dir.resolve())
