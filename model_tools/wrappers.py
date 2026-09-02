"""ATC-friendly UniRec wrappers.

The decoder deliberately uses a fixed-capacity self-attention cache. The app
updates one cache slot after every decoding step. This removes the dynamic
past-sequence axis that is difficult to compile into a finite OM gear set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from model_tools.contract import DECODER_LAYERS

if TYPE_CHECKING:
    from openrec.modeling.unirec_modeling.modeling_unirec import (
        UniRecForConditionalGenerationNew,
    )


class EncoderForOm(nn.Module):
    """Run the vision encoder and precompute every decoder cross-attention K/V."""

    def __init__(self, model: UniRecForConditionalGenerationNew) -> None:
        super().__init__()
        self.encoder = model.model.encoder
        self.cross_attentions = nn.ModuleList(
            [layer.encoder_attn for layer in model.model.decoder.layers]
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(pixel_values, return_dict=False)[0]
        batch_size = hidden_states.shape[0]
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for attention in self.cross_attentions:
            keys.append(attention._shape(attention.k_proj(hidden_states), -1, batch_size))
            values.append(attention._shape(attention.v_proj(hidden_states), -1, batch_size))
        # Batch size is fixed to one. Concatenating that axis directly produces
        # [layers, heads, sequence, head_dim] without a rank-five intermediate.
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)


class DecoderStepForOm(nn.Module):
    """One autoregressive step with static cache capacity and dynamic visual K/V."""

    def __init__(self, model: UniRecForConditionalGenerationNew) -> None:
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head

    @staticmethod
    def _attention(
        attention: nn.Module,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, target_length, _ = hidden_states.shape
        query_states = attention._shape(
            attention.q_proj(hidden_states) * attention.scaling,
            target_length,
            batch_size,
        )
        weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        if attention_mask is not None:
            weights = weights + attention_mask
        probabilities = torch.softmax(weights, dim=-1)
        output = torch.matmul(probabilities, value_states)
        output = output.transpose(1, 2).reshape(batch_size, target_length, attention.embed_dim)
        return attention.out_proj(output)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.decoder.embed_tokens(input_ids)
        flat_positions = position_ids.reshape(-1)
        positions = self.decoder.embed_positions.weights.index_select(0, flat_positions)
        positions = positions.reshape(hidden_states.shape)
        hidden_states = hidden_states + positions

        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []

        for layer_index in range(DECODER_LAYERS):
            layer = self.decoder.layers[layer_index]

            residual = hidden_states
            normalized = layer.self_attn_layer_norm(hidden_states)
            current_key = layer.self_attn._shape(
                layer.self_attn.k_proj(normalized), 1, normalized.shape[0]
            )
            current_value = layer.self_attn._shape(
                layer.self_attn.v_proj(normalized), 1, normalized.shape[0]
            )
            self_key = torch.cat((past_keys[layer_index], current_key[0]), dim=1)
            self_value = torch.cat((past_values[layer_index], current_value[0]), dim=1)
            hidden_states = residual + self._attention(
                layer.self_attn,
                normalized,
                self_key,
                self_value,
                self_attention_mask,
            )

            residual = hidden_states
            normalized = layer.encoder_attn_layer_norm(hidden_states)
            hidden_states = residual + self._attention(
                layer.encoder_attn,
                normalized,
                cross_k[layer_index],
                cross_v[layer_index],
                None,
            )

            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = residual + layer.fc2(hidden_states)

            new_keys.append(current_key)
            new_values.append(current_value)

        hidden_states = self.decoder.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states[:, 0, :])
        return (
            logits,
            torch.cat(new_keys, dim=0),
            torch.cat(new_values, dim=0),
        )
