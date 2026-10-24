"""Score the frozen beta=0/beta=selected pair on the untouched new classes."""

import argparse
import statistics
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, checksum, read_json, write_json
from adaptive_fsvod.evaluate import cache_signature, evaluate, load_detector
from adaptive_fsvod.train import environment
from prepare_confirmation import immutable
from prepare_score_confirmation import verify_fresh
from run_score_calibration import STUDY


def score(root):
    selection = verify_fresh(root)
    directory = root / STUDY / "confirmation"
    frozen = read_json(directory / "cache_artifact_sha256.json")
    if frozen["selection_sha256"] != checksum(directory / "selection.json"):
        raise ValueError("Frozen confirmation selection changed")
    for name, digest in frozen["files"].items():
        if checksum(directory / "cache" / name) != digest:
            raise ValueError("Frozen confirmation feature changed")
    for name, key in (("metadata.json", "metadata_sha256"), ("support_pool.json", "support_pool_sha256")):
        if checksum(directory / "cache" / name) != frozen[key]:
            raise ValueError("Frozen cache metadata changed")
    protocol = read_json(root / STUDY / "protocol.json")
    execution = {"environment": environment(), "threads": torch.get_num_threads(),
                 "feature_manifest_sha256": checksum(directory / "cache_artifact_sha256.json")}
    environment_path = root / STUDY / "confirmation_environment.json"
    if not environment_path.exists():
        immutable(environment_path, execution)
    cached = CachedEpisodes(VideoManifest(directory / "confirmation.json"), directory / "cache")
    episodes_path = directory / "episodes.json"
    episodes = read_json(episodes_path)
    records = []
    for source in sorted(protocol["models"], key=lambda r: r["seed"]):
        model, checkpoint = load_detector(source["checkpoint"])
        if checkpoint["cache_signature"] != cache_signature(cached.metadata):
            raise ValueError("Encoder settings differ from training")
        seed_rows = []
        for method, beta in (("baseline", 0.), ("objectness", selection["selected_beta"])):
            output = root / STUDY / "evaluations" / f"{method}_seed{source['seed']}"
            expected = {"checkpoint_sha256": source["checkpoint_sha256"], "objectness_beta": beta,
                        "manifest_sha256": checksum(directory / "confirmation.json"),
                        "episodes_sha256": checksum(episodes_path),
                        "feature_manifest_sha256": execution["feature_manifest_sha256"]}
            if all((output / f"{name}.json").exists() for name in ("metrics", "ground_truth", "predictions")):
                metrics = read_json(output / "metrics.json")
                if any(metrics.get(k) != v for k, v in expected.items()):
                    raise ValueError("Existing evaluation used a different protocol")
            else:
                metrics, truth, predictions = evaluate(model, cached, episodes, "cpu", objectness_beta=beta)
                metrics.update({**expected, "training_seed": source["seed"], "training_budget": 300,
                                "checkpoint_step": checkpoint["step"], "training_git": checkpoint["environment"]["git"],
                                "model_config": checkpoint["model_config"], "cache_signature": checkpoint["cache_signature"],
                                "encoder_attention_backend": cached.metadata.get("attention_backend", "eager"),
                                "study_protocol_sha256": checksum(root / STUDY / "protocol.json"),
                                "fresh_selection_sha256": checksum(directory / "selection.json"),
                                "evaluation_environment": execution})
                for name, value in (("metrics", metrics), ("ground_truth", truth), ("predictions", predictions)):
                    write_json(output / f"{name}.json", value)
            record = {"method": method, "seed": source["seed"], **metrics}
            records.append(record)
            seed_rows.append(record)
            print("FRESH RESULT", method, source["seed"],
                  {k: round(metrics[k], 4) for k in ("AP", "AP50", "AP75")}, flush=True)
        for key in ("mean_memory_tokens", "mean_threshold", "checkpoint_sha256"):
            if seed_rows[0][key] != seed_rows[1][key]:
                raise ValueError("Score calibration changed the detector or temporal computation")
    summary = {}
    for method in ("baseline", "objectness"):
        rows = [r for r in records if r["method"] == method]
        summary[method] = {key: {"mean": statistics.mean(r[key] for r in rows),
                                "sample_std": statistics.stdev(r[key] for r in rows)}
                           for key in ("AP", "AP50", "AP75")}
    write_json(root / STUDY / "confirmation_results.json", {"rows": records, "summary": summary})
    print("FRESH SUMMARY", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    score(args.experiment_dir.resolve())
