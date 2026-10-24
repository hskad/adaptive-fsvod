"""Detect support classes in one folder of ordered video frames."""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .cache import FrozenEncoder
from .data import write_json
from .evaluate import decode, load_detector


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="One video; frame names must sort in time order")
    parser.add_argument("--support", required=True, help="One subfolder of object crops per class")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    detector, checkpoint = load_detector(args.checkpoint, args.device)
    signature = checkpoint["cache_signature"]
    if signature.get("synthetic"):
        raise ValueError("Synthetic software-check weights cannot be used as research checkpoints")
    encoder = FrozenEncoder(signature["encoder"], args.device, signature["revision"], signature["candidates"])
    class_folders = sorted(p for p in Path(args.support).iterdir() if p.is_dir())
    if not class_folders:
        raise ValueError("No support class folders found")
    classes, support = [], []
    for folder in class_folders:
        vectors = []
        for path in image_paths(folder):
            with Image.open(path) as image:
                vectors.append(encoder.encode(image.convert("RGB"), support=True))
        if not vectors:
            raise ValueError(f"No support images in {folder}")
        classes.append(folder.name)
        support.append(torch.stack(vectors).to(args.device))
    prototypes = detector.prototypes(support)
    paths = image_paths(args.frames)
    if not paths:
        raise ValueError("No input frames found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    memory = None
    results, timings = [], []
    for path in paths:
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with Image.open(path) as source:
            image = source.convert("RGB")
        frame = encoder.encode(image)
        detected = detector.forward_frame(frame["features"].float().to(args.device),
                                           frame["boxes"].to(args.device), prototypes, memory)
        memory = detected["memory"]
        predictions = decode(detected, classes, *image.size, args.score_threshold)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        draw = ImageDraw.Draw(image)
        for item in predictions:
            draw.rectangle(item["bbox"], outline="red", width=2)
            x, y = item["bbox"][:2]
            draw.text((x, y), f"{item['category_id']} {item['score']:.2f}", fill="red")
        image.save(output / f"{path.stem}.png")
        results.append({"frame": path.name, "detections": predictions})
    write_json(output / "detections.json", results)
    # Exclude the first frame for the warm run metric. Keep all timings for review.
    measured = timings[1:] or timings
    write_json(output / "timing.json", {"seconds_per_frame": timings,
        "warm_fps": len(measured) / sum(measured), "encoder": signature,
        "timing_scope": "frame read, preprocessing, encoder, temporal heads, decode; support setup and output writing excluded"})
    print(f"Saved detections and annotated frames to {output}")


if __name__ == "__main__":
    main()
