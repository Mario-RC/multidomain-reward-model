# modeling_custom.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.utils import ModelOutput
from utils import TOKEN_PATTERNS_BY_MODEL_TYPE, find_token_for_gating

class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 10,
                 logit_scale: float = 1., hidden_dim: int = 1024, n_hidden: int = 3, dropout: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
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
        x = F.softmax(x / self.temperature, dim=-1)
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
                                    hidden_dim=config_dict.get("gating_hidden_dim", 1024),
                                    n_hidden=config_dict.get("gating_n_hidden", 3))

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