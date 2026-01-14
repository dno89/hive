"""Custom RLlib Torch model with communication-aware attention."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CommunicatingAgentTorchModel(TorchModelV2, nn.Module):
    """Model that fuses local state with neighbor messages via a transformer."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        state_dim = int(np.prod(obs_space["self"].shape))
        comm_dim = int(obs_space["comm"].shape[-1])
        hidden_sizes: List[int] = model_config.get("state_hidden", [128, 64])
        model_dim = model_config.get("comm_model_dim", 64)
        n_heads = model_config.get("comm_num_heads", 4)
        n_layers = model_config.get("comm_num_layers", 2)

        state_layers: List[nn.Module] = []
        last_size = state_dim
        for hidden in hidden_sizes:
            state_layers.append(nn.Linear(last_size, hidden))
            state_layers.append(nn.ReLU(inplace=True))
            last_size = hidden
        self._state_encoder = nn.Sequential(*state_layers)

        self._msg_input = nn.Linear(comm_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self._msg_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        fusion_dim = last_size + model_dim
        self._fusion = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(inplace=True))
        self._action_head = nn.Linear(128, num_outputs)
        self._value_head = nn.Linear(128, 1)
        self._value_out = None

    @staticmethod
    def default_model_config() -> dict:
        return {
            "state_hidden": [128, 64],
            "comm_model_dim": 64,
            "comm_num_heads": 4,
            "comm_num_layers": 2,
        }

    def forward(self, input_dict, state: List[torch.Tensor], seq_lens: torch.Tensor = None):
        obs = input_dict["obs"]
        self_obs = obs["self"].float()
        messages = obs["comm"].float()
        messages_mask = obs.get("comm_mask")

        state_features = self._state_encoder(self_obs)
        msg_embeddings = self._msg_input(messages)

        if messages_mask is None:
            valid_mask = msg_embeddings.abs().sum(dim=-1) > 0
        else:
            valid_mask = messages_mask.bool()
        attn_mask = ~valid_mask

        encoded_msgs = self._msg_encoder(msg_embeddings, src_key_padding_mask=attn_mask)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        msg_summary = (encoded_msgs * valid_mask.unsqueeze(-1)).sum(dim=1) / denom

        fused = self._fusion(torch.cat([state_features, msg_summary], dim=-1))
        self._value_out = self._value_head(fused).squeeze(-1)
        logits = self._action_head(fused)
        return logits, state

    def value_function(self) -> torch.Tensor:
        assert self._value_out is not None, "value_function called before forward"
        return self._value_out
