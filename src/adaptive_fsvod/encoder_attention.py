"""Optional memory-saving attention for the frozen OWLv2 vision encoder.

Preserve projections, scaling, masks, and weights. PyTorch SDPA changes the
floating-point kernel, not the attention equation. Native eager execution
remains the default and handles requests for explicit attention matrices.
"""

import types

from torch.nn import functional as F


def enable_sdpa(model):
    from transformers.models.owlv2.modeling_owlv2 import Owlv2Attention

    for layer in model.owlv2.vision_model.encoder.layers:
        attention = layer.self_attn
        if not isinstance(attention, Owlv2Attention):
            raise TypeError("SDPA adapter requires the pinned OWLv2 attention implementation")
        if hasattr(attention, "_native_forward"):
            raise ValueError("Attention adapter already installed")
        attention._native_forward = attention.forward
        attention.forward = types.MethodType(sdpa_forward, attention)


def sdpa_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
    if output_attentions or self.training:
        return self._native_forward(hidden_states, attention_mask, causal_attention_mask, output_attentions)
    batch, length, dim = hidden_states.shape
    shape = (batch, length, self.num_heads, self.head_dim)
    # Scale Q before the dot product, as in the native implementation.
    query = (self.q_proj(hidden_states) * self.scale).view(shape).transpose(1, 2)
    key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    mask = attention_mask
    for current in (attention_mask, causal_attention_mask):
        if current is not None and current.shape != (batch, 1, length, length):
            raise ValueError("Attention mask has an unexpected shape")
    if causal_attention_mask is not None:
        mask = causal_attention_mask if mask is None else mask + causal_attention_mask
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0., scale=1.)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    return self.out_proj(output), None
