"""Plot the measured pilot and validation diagnostics from saved artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_fsvod.data import read_json


def plot(root, output):
    summary = read_json(root / "exports/pilot_summary.json")
    diagnostic = read_json(root / "exports/validation_memory_diagnostics_seed42.json")["methods"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    colors = {"fixed": "#2563eb", "adaptive": "#d97706"}
    for method in ("fixed", "adaptive"):
        rows = [r for r in summary["rows"] if r["method"] == method]
        axes[0, 0].plot([str(r["seed"]) for r in rows], [r["AP"] for r in rows],
                        marker="o", color=colors[method], label=method.capitalize())
        records = [json.loads(line) for line in
                   (root / f"runs/{method}_seed42/training.jsonl").read_text().splitlines()]
        records = [r for r in records if "validation" in r]
        axes[0, 1].plot([r["step"] for r in records], [r["validation"]["AP"] for r in records],
                        marker="o", color=colors[method], label=method.capitalize())
    axes[0, 0].set(title="Test AP: three paired training seeds", xlabel="Training seed", ylabel="Box AP (%)")
    axes[0, 1].set(title="Validation learning curves: seed 42", xlabel="Training step", ylabel="Box AP (%)")
    names = ["Fixed", "Adaptive"]
    agreements = [diagnostic[m]["token_agreement"]["token_agreement_fraction"] * 100 for m in colors]
    bars = axes[1, 0].bar(names, agreements, color=list(colors.values()), width=.5)
    axes[1, 0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1, 0].set(title="Validation memory token agreement: seed 42", ylabel="Correct class and IoU >= 0.50 (%)", ylim=(0, 50))
    axes[1, 0].text(.5, -.19, "Many tokens may match one object; this is not detection precision.",
                    ha="center", transform=axes[1, 0].transAxes, fontsize=8)
    for index, method in enumerate(colors):
        values = [diagnostic[method]["validation"]["AP"], diagnostic[method]["memory_off_validation"]["AP"]]
        bars = axes[1, 1].bar([index - .18, index + .18], values, width=.34,
                              color=[colors[method], colors[method]], alpha=.8)
        bars[1].set_hatch("///")
        axes[1, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1, 1].set(title="Validation: same weights, memory disabled", ylabel="Box AP (%)", ylim=(0, 72))
    axes[1, 1].set_xticks([0, 1], names)
    axes[1, 1].text(.5, -.19, "Solid: memory on. Hatched: memory off at inference only.",
                    ha="center", transform=axes[1, 1].transAxes, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.experiment_dir, args.output)
