"""Optional real-weight API check. Requires the public OWLv2 download."""

import argparse
import json
import time

import torch
from PIL import Image

from adaptive_fsvod.cache import FrozenEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(4)
    revision = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
    encoder = FrozenEncoder(device=args.device, revision=revision)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    started = time.perf_counter()
    features = encoder.encode(image)
    elapsed = time.perf_counter() - started
    assert features["features"].shape == (128, 512)
    assert torch.isfinite(features["features"]).all()
    assert torch.isfinite(features["boxes"]).all()
    print(json.dumps({"status": "PASS", "revision": encoder.revision,
                      "feature_shape": list(features["features"].shape),
                      "seconds": elapsed, "device": args.device,
                      "note": "Real pretrained feature extraction check; not an accuracy evaluation"}))


if __name__ == "__main__":
    main()
