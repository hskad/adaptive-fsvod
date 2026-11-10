"""Controlled CPU latency of the baseline (beta=0) and objectness score (beta=1).

Measures milliseconds per frame for the trainable heads plus decoding (score, floor,
class-wise NMS) on the fixed confirmation episodes. Features are preloaded so file IO
is excluded; the frozen encoder is excluded because both methods share it. The head
forward pass is identical for both methods; decoding is timed separately for each
beta, interleaved within every repeat. Medians over repeats, then means over seeds.
"""

import argparse
import statistics
import time
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, checksum, read_json, write_json
from adaptive_fsvod.evaluate import decode, load_detector
from adaptive_fsvod.train import environment

BETAS = (0.0, 1.0)


def benchmark(root, repeats=5, threads=4, weights=None):
    torch.set_num_threads(threads)
    base = root / "score_calibration/confirmation"
    cached = CachedEpisodes(VideoManifest(base / "confirmation.json"), base / "cache")
    episodes = read_json(base / "episodes.json")["episodes"]
    loaded = [(episode, cached.load(episode, "cpu")) for episode in episodes]
    n_frames = sum(len(item[1][0]) for item in loaded)
    per_seed = {}
    for seed in (42, 43, 44):
        checkpoint = (weights or root / "weights") / f"shared_seed{seed}.pt"
        model, _ = load_detector(checkpoint, "cpu")
        head_ms, decode_ms = [], {beta: [] for beta in BETAS}
        with torch.inference_mode():
            for _, (frames, support, _, _) in loaded[:5]:  # warm-up
                model(frames, support)
            for repeat in range(repeats):
                head_seconds, decode_seconds = 0.0, {beta: 0.0 for beta in BETAS}
                order = BETAS if repeat % 2 == 0 else tuple(reversed(BETAS))
                for episode, (frames, support, _, info) in loaded:
                    started = time.perf_counter()
                    outputs = model(frames, support)
                    head_seconds += time.perf_counter() - started
                    for beta in order:
                        started = time.perf_counter()
                        for output, frame, encoded in zip(outputs, info, frames):
                            decode(output, episode["classes"], frame["width"], frame["height"], 0.001, 0.5,
                                   objectness=encoded.get("objectness"), objectness_beta=beta)
                        decode_seconds[beta] += time.perf_counter() - started
                head_ms.append(1000 * head_seconds / n_frames)
                for beta in BETAS:
                    decode_ms[beta].append(1000 * decode_seconds[beta] / n_frames)
        head = statistics.median(head_ms)
        per_seed[seed] = {"checkpoint_sha256": checksum(checkpoint), "head_ms": head,
                          "decode_ms": {str(b): statistics.median(decode_ms[b]) for b in BETAS},
                          "total_ms": {str(b): head + statistics.median(decode_ms[b]) for b in BETAS}}
        print(seed, per_seed[seed]["total_ms"], flush=True)
    mean_total = {str(b): statistics.mean(r["total_ms"][str(b)] for r in per_seed.values()) for b in BETAS}
    return {"scope": "CPU ms/frame: trainable heads + decoding; features preloaded; encoder excluded",
            "frames": n_frames, "episodes": len(episodes), "repeats": repeats, "threads": threads,
            "per_seed": per_seed, "mean_total_ms": mean_total,
            "difference_ms": mean_total["1.0"] - mean_total["0.0"],
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "environment": environment()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/score_calibration/latency.json"))
    args = parser.parse_args()
    record = benchmark(args.experiment_dir, args.repeats, args.threads, args.weights_dir)
    write_json(args.output, record)
    print("MEAN_MS", record["mean_total_ms"], "DIFFERENCE_MS", record["difference_ms"])
