"""Create a paired result table and AP plot from measured evaluation files."""

import argparse
import csv
from pathlib import Path

from .data import read_json, write_json


def compare(fixed_path, adaptive_path, output):
    fixed, adaptive = read_json(fixed_path), read_json(adaptive_path)
    for key in ("manifest_sha256", "episodes_sha256", "score_floor", "nms_iou", "synthetic",
                "training_seed", "training_budget", "cache_signature"):
        if key not in fixed or key not in adaptive or fixed[key] != adaptive[key]:
            raise ValueError(f"Results do not use the same evaluation protocol: {key}")
    if fixed["model_config"]["memory_mode"] != "fixed" or adaptive["model_config"]["memory_mode"] != "adaptive":
        raise ValueError("Expected fixed and adaptive results, in that order")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = ["AP", "AP50", "AP75", "mean_memory_tokens", "cached_pipeline_fps"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *keys, "synthetic"])
        writer.writeheader()
        for name, result in (("fixed", fixed), ("adaptive", adaptive)):
            writer.writerow({"method": name, **{k: result[k] for k in keys}, "synthetic": result["synthetic"]})
    write_json(output / "comparison.json", {"fixed": fixed, "adaptive": adaptive,
        "difference_adaptive_minus_fixed": {k: adaptive[k] - fixed[k] for k in keys},
        "note": "Cached pipeline FPS excludes the image encoder. AP differences are percentage points."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(5.5, 3.2))
    for offset, (label, result) in enumerate((("Fixed", fixed), ("Adaptive", adaptive))):
        axes.bar([x + offset * .36 for x in range(3)], [result[k] for k in keys[:3]], width=.36, label=label)
    axes.set_xticks([.18, 1.18, 2.18], keys[:3])
    axes.set_ylabel("Box AP (%)")
    axes.set_ylim(0, 100)
    axes.set_title("SYNTHETIC SOFTWARE CHECK" if fixed["synthetic"] else "Paired video detection evaluation")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--adaptive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.fixed, args.adaptive, args.output)


if __name__ == "__main__":
    main()
