"""Train small temporal heads with paired, fixed few-shot episodes."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .cache import CachedEpisodes, fit_feature_center, save_tensor
from .data import VideoManifest, checksum, read_json, validate_episodes, validate_splits, write_json
from .evaluate import cache_signature, evaluate
from .loss import DetectionLoss
from .model import ModelConfig, TemporalDetector


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    packages = {}
    for package in ["torch", "transformers", "numpy", "scipy", "Pillow", "pycocotools"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def train(args):
    set_seed(args.seed)
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        raise ValueError("Run already exists. Use --resume or choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = VideoManifest(args.train_manifest), VideoManifest(args.val_manifest)
    validate_splits([train_manifest, val_manifest])
    if train_manifest.data["split"] != "train" or val_manifest.data["split"] != "val":
        raise ValueError("Training requires train and val splits. Test data is not used for selection.")
    train_cache = CachedEpisodes(train_manifest, args.train_cache)
    val_cache = CachedEpisodes(val_manifest, args.val_cache)
    if cache_signature(train_cache.metadata) != cache_signature(val_cache.metadata):
        raise ValueError("Train and validation caches use different encoder settings")
    training = read_json(args.train_episodes)
    validation = read_json(args.val_episodes)
    validate_episodes(train_manifest, training)
    validate_episodes(val_manifest, validation)
    config = read_json(args.config)
    model_config = ModelConfig(**config)
    if model_config.input_dim != train_cache.metadata["feature_dim"]:
        raise ValueError("Model input dimension differs from cached features")
    model = TemporalDetector(model_config).to(args.device)
    center_record = None
    if model_config.feature_centering:
        center = fit_feature_center(train_cache)
        model.feature_center.copy_(center.to(args.device))
        center_record = {"source": "class-balanced training support cache",
                         "support_pool_sha256": checksum(Path(args.train_cache) / "support_pool.json"),
                         "float32_sha256": hashlib.sha256(center.numpy().tobytes()).hexdigest()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.steps * 0.05))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step - warmup) / max(1, args.steps - warmup))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = DetectionLoss()
    provenance = {"arguments": vars(args), "environment": environment(), "model_config": config,
                  "cache_signature": cache_signature(train_cache.metadata),
                  "train_manifest_sha256": checksum(args.train_manifest),
                  "val_manifest_sha256": checksum(args.val_manifest),
                  "train_episodes_sha256": checksum(args.train_episodes),
                  "val_episodes_sha256": checksum(args.val_episodes)}
    if center_record is not None:
        provenance["feature_center"] = center_record
    first_step, best_ap = 1, -1.0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous.get("feature_center") != provenance.get("feature_center"):
            raise ValueError("Resume training feature center differs")
        for key in ("model_config", "cache_signature", "train_manifest_sha256", "val_manifest_sha256",
                    "train_episodes_sha256", "val_episodes_sha256"):
            if previous[key] != provenance[key]:
                raise ValueError(f"Resume settings differ: {key}")
        # A resumed schedule must retain the original total budget.
        for key in ("steps", "lr", "weight_decay", "seed"):
            if previous["arguments"][key] != vars(args)[key]:
                raise ValueError(f"Resume training argument differs: {key}")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        first_step, best_ap = previous["step"] + 1, previous["best_val_ap"]
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        if torch.cuda.is_available() and previous.get("cuda_rng"):
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    write_json(output / "config.json", provenance)
    logs = output / "training.jsonl"
    started = time.perf_counter()
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Training steps and evaluation interval must be positive")
    for step in range(first_step, args.steps + 1):
        model.train()
        episode = training["episodes"][(step - 1) % len(training["episodes"])]
        frames, support, targets, _ = train_cache.load(episode, args.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames, support)
        losses = [criterion(result, target) for result, target in zip(outputs, targets)]
        loss = torch.stack([item["total"] for item in losses]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
                  "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started,
                  "mean_memory_tokens": float(np.mean([o["memory"].shape[0] for o in outputs])),
                  "synthetic": train_cache.metadata.get("synthetic", False)}
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _, _ = evaluate(model, val_cache, validation, args.device)
            record["validation"] = metrics
            improved = metrics["AP"] > best_ap
            best_ap = max(best_ap, metrics["AP"])
            checkpoint = {**provenance, "step": step, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "best_val_ap": best_ap, "torch_rng": torch.get_rng_state(),
                          "python_rng": random.getstate(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            if improved:
                save_tensor(output / "best.pt", checkpoint)
            # Save best first. A disconnect must not leave last.pt claiming
            # a best checkpoint that has not yet been written.
            save_tensor(output / "last.pt", checkpoint)
            write_json(output / "validation_latest.json", metrics)
        with logs.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if step == 1 or step % 10 == 0 or "validation" in record:
            print(json.dumps(record), flush=True)
    return output / "best.pt"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("train-manifest", "val-manifest", "train-cache", "val-cache", "train-episodes", "val-episodes", "config", "output"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--steps", type=int, default=300)
    result.add_argument("--eval-every", type=int, default=50)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resume")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
