import types

import pytest
import torch
from transformers.models.owlv2.configuration_owlv2 import Owlv2VisionConfig
from transformers.models.owlv2.modeling_owlv2 import Owlv2Attention

from adaptive_fsvod.encoder_attention import sdpa_forward


@pytest.mark.parametrize("mask_mode", ["none", "padding", "causal", "both"])
def test_sdpa_matches_native_attention_with_masks_and_fallback(mask_mode):
    torch.manual_seed(12)
    attention = Owlv2Attention(Owlv2VisionConfig(hidden_size=32, num_attention_heads=4)).eval()
    states = torch.randn(2, 17, 32)
    padding = torch.zeros(2, 1, 17, 17) if mask_mode in ("padding", "both") else None
    if padding is not None:
        padding[..., -3:] = torch.finfo(states.dtype).min
    causal = torch.full((2, 1, 17, 17), torch.finfo(states.dtype).min).triu(1) if mask_mode in ("causal", "both") else None
    reference = attention(states, padding, causal)
    original_state = {key: value.clone() for key, value in attention.state_dict().items()}
    attention._native_forward = attention.forward
    attention.forward = types.MethodType(sdpa_forward, attention)
    actual = attention(states, padding, causal)
    torch.testing.assert_close(actual[0], reference[0], atol=2e-7, rtol=2e-6)
    assert actual[1] is None
    explicit = attention(states, padding, causal, output_attentions=True)
    native_explicit = attention._native_forward(states, padding, causal, output_attentions=True)
    assert torch.equal(explicit[0], native_explicit[0])
    assert torch.equal(explicit[1], native_explicit[1])
    for key, value in original_state.items():
        assert torch.equal(value, attention.state_dict()[key])
