"""Exercise the real Transformers API without downloading weights."""

import torch
from PIL import Image
from transformers import Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor

from adaptive_fsvod.cache import FrozenEncoder


def test_encoder_api_with_tiny_random_model():
    config = Owlv2Config(
        text_config={"vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8},
        vision_config={"image_size": 32, "patch_size": 16, "hidden_size": 32,
                       "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4},
        projection_dim=16,
    )
    model = Owlv2ForObjectDetection(config)
    processor = Owlv2ImageProcessor(size={"height": 32, "width": 32})
    encoder = FrozenEncoder("tiny-random-software-check", candidates=3, model=model, processor=processor)
    image = Image.new("RGB", (48, 24), "red")
    encoded = encoder.encode(image)
    support = encoder.encode(image, support=True)
    assert encoded["features"].shape == (3, 16)
    assert encoded["boxes"].shape == (3, 4)
    assert support.shape == (16,)
    assert torch.isfinite(encoded["features"]).all()
    assert ((encoded["boxes"] >= 0) & (encoded["boxes"] <= 1)).all()
    assert not any(p.requires_grad for p in model.parameters())
