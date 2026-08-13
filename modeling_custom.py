# modeling_custom.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.utils import ModelOutput
try:
    from .utils import TOKEN_PATTERNS_BY_MODEL_TYPE, find_token_for_gating
except ImportError:  # Local source-tree execution.
    from utils import TOKEN_PATTERNS_BY_MODEL_TYPE, find_token_for_gating

class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 10,
                 logit_scale: float = 1., hidden_dim: int = 1024, n_hidden: int = 3, dropout: float = 0.0,
                 learnable_logit_scale: bool = False,
                 active_attribute_indices: Optional[List[int]] = None):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.logit_scale = nn.Parameter(
            torch.ones(1) * logit_scale, requires_grad=learnable_logit_scale
        )
        self.dropout_prob = dropout
        active_mask = torch.ones(out_features, dtype=torch.bool)
        if active_attribute_indices is not None:
            if not active_attribute_indices:
                raise ValueError("At least one attribute must be active.")
            if min(active_attribute_indices) < 0 or max(active_attribute_indices) >= out_features:
                raise ValueError("active_attribute_indices contains an out-of-range index.")
            active_mask.zero_()
            active_mask[list(active_attribute_indices)] = True
        # Derived from packaged config; omit from state_dict for legacy compatibility.
        self.register_buffer("active_attribute_mask", active_mask, persistent=False)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0 and self.training:
                    x = F.dropout(x, p=self.dropout_prob)
        logits = x / self.temperature
        mask = self.active_attribute_mask.to(device=logits.device)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        x = F.softmax(logits, dim=-1)
        return x * self.logit_scale

@dataclass
class CustomOutput(ModelOutput):
    rewards: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    prompt_embedding: Optional[torch.Tensor] = None
    gating_output: Optional[torch.Tensor] = None
    score: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

class RewardModelWithGating(PreTrainedModel):
    """Backbone-agnostic reward model with a prompt-conditioned gating network."""

    config_class = AutoConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = AutoModel.from_config(config)
        config_dict = config.to_dict()

        # Default objective count for this project.
        self.num_objectives = config_dict.get("num_objectives", 23)

        self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives, bias=False)
        self.post_init()

        # Avoid torch.eye to keep compatibility with BF16 training setups.
        I = torch.zeros(self.num_objectives, self.num_objectives)
        I[range(self.num_objectives), range(self.num_objectives)] = 1.
        self.reward_transform_matrix = nn.Parameter(I)
        self.reward_transform_matrix.requires_grad = False

        self.gating = GatingNetwork(config.hidden_size, self.num_objectives,
                                    temperature=config_dict.get("gating_temperature", 10),
                                    logit_scale=config_dict.get("gating_logit_scale", 1.0),
                                    hidden_dim=config_dict.get("gating_hidden_dim", 1024),
                                    n_hidden=config_dict.get("gating_n_hidden", 3),
                                    dropout=config_dict.get("gating_dropout", 0.0),
                                    learnable_logit_scale=config_dict.get("gating_learnable_logit_scale", False),
                                    active_attribute_indices=config_dict.get(
                                        "gating_active_attribute_indices"))

    def compute_gating(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute routing weights from a prompt-only token sequence.

        Callers should render the prompt with ``add_generation_prompt=True``.
        The final non-padding prompt token is causal and therefore cannot see
        either candidate response.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs[0]
        if attention_mask is None:
            positions = torch.full(
                (hidden.shape[0],), hidden.shape[1] - 1,
                dtype=torch.long, device=hidden.device,
            )
        else:
            token_positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).unsqueeze(0)
            positions = (attention_mask.long() * token_positions).argmax(dim=-1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.gating(hidden[rows, positions])

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            gating_output_override: Optional[torch.Tensor] = None,
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tokens_hidden_states = transformer_outputs[0]
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # If no pad token is found, modulo keeps ONNX-compatible indexing.
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(tokens_hidden_states.device)
            else:
                sequence_lengths = -1

        dummy_iterator = torch.arange(batch_size, device=tokens_hidden_states.device)
        hidden_states = tokens_hidden_states[dummy_iterator, sequence_lengths]
        assert hidden_states.shape == (batch_size, self.config.hidden_size)

        rewards = self.regression_layer(hidden_states)
        prompt_embedding = None
        if gating_output_override is not None:
            gating_output = gating_output_override.to(device=rewards.device, dtype=rewards.dtype)
            if gating_output.ndim == 1:
                gating_output = gating_output.unsqueeze(0)
            if gating_output.shape != rewards.shape:
                raise ValueError(
                    f"gating_output_override shape {tuple(gating_output.shape)} does not "
                    f"match rewards shape {tuple(rewards.shape)}"
                )
        elif getattr(self.config, "shared_prompt_gating", False):
            raise ValueError(
                "This shared-prompt-gating checkpoint requires gating_output_override. "
                "Compute one prompt-only gate with compute_gating() and reuse it for all candidates."
            )
        else:
            if input_ids is None:
                raise ValueError("input_ids is required to compute gating token positions.")
            model_type = getattr(self.config, "model_type", None)
            gating_token_positions = [
                find_token_for_gating(ids.detach().cpu().tolist(), model_type) for ids in input_ids
            ]
            prompt_embedding = tokens_hidden_states[dummy_iterator, gating_token_positions, :]
            gating_output = self.gating(prompt_embedding)
        rewards_adjusted = rewards @ self.reward_transform_matrix
        score = torch.sum(gating_output * rewards_adjusted, dim=1)

        return CustomOutput(
            rewards=rewards,
            hidden_state=hidden_states,
            prompt_embedding=prompt_embedding,
            gating_output=gating_output,
            score=score,
            logits=score,
        )


# Backward compatibility alias for existing imports/checkpoints.
LlamaForRewardModelWithGating = RewardModelWithGating
