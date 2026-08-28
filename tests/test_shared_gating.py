from types import SimpleNamespace

import torch
from torch import nn

from attributes import ATTRIBUTES, DOMAIN_ATTRIBUTE_INDICES, DOMAIN_NAMES
from attributes import attribute_selection_suffix, resolve_active_attributes
from modeling_custom import GatingNetwork, RewardModelWithGating
from config_utils import apply_model_registry, apply_section_overrides
from utils import (
    _resolve_inference_model_path, _score_pair_shared_gate, _stable_int64_id,
    debiasing_checkpoint_suffix, shared_gate_checkpoint_filename,
    validate_shared_routing_config,
)


def test_domain_attribute_partition_is_complete_and_disjoint():
    indices = [
        i for domain in DOMAIN_NAMES for i in DOMAIN_ATTRIBUTE_INDICES[domain]
    ]
    assert sorted(indices) == list(range(len(ATTRIBUTES)))
    assert len(indices) == len(set(indices))


def test_gating_temperature_must_be_positive():
    try:
        GatingNetwork(4, len(ATTRIBUTES), temperature=0)
    except ValueError as error:
        assert "Temperature must be positive" in str(error)
    else:
        raise AssertionError("A non-positive gating temperature was accepted.")


def test_logit_scale_is_fixed_by_default():
    gate = GatingNetwork(4, len(ATTRIBUTES), hidden_dim=8, n_hidden=1, logit_scale=2.0)
    output = gate(torch.randn(3, 4))
    assert not gate.logit_scale.requires_grad
    torch.testing.assert_close(output.sum(-1), torch.full((3,), 2.0))


class _Backbone(nn.Module):
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        batch, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.float32)
        return (positions.view(1, length, 1).expand(batch, length, 2),)


def test_compute_gating_uses_last_non_padding_token_with_left_padding():
    model = RewardModelWithGating.__new__(RewardModelWithGating)
    nn.Module.__init__(model)
    model.model = _Backbone()
    model.gating = nn.Identity()
    input_ids = torch.tensor([[0, 0, 4, 5], [7, 8, 0, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    output = model.compute_gating(input_ids, attention_mask)
    torch.testing.assert_close(output[:, 0], torch.tensor([3.0, 1.0]))


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        suffix = "<assistant>" if add_generation_prompt else ""
        return "|".join(message["content"] for message in messages) + suffix

    def __call__(self, text, **kwargs):
        length = max(len(text), 1)
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }


class _PairModel:
    def __init__(self):
        self.compute_calls = 0
        self.overrides = []

    def compute_gating(self, input_ids, attention_mask):
        self.compute_calls += 1
        return torch.arange(len(ATTRIBUTES), dtype=torch.float32).unsqueeze(0)

    def __call__(self, input_ids, attention_mask=None, gating_output_override=None):
        self.overrides.append(gating_output_override)
        return SimpleNamespace(score=input_ids.float().sum())


def test_pair_scoring_computes_one_gate_and_reuses_it_for_both_candidates():
    model = _PairModel()
    prompt = [{"role": "user", "content": "prompt"}]
    chosen = prompt + [{"role": "assistant", "content": "chosen"}]
    rejected = prompt + [{"role": "assistant", "content": "rejected"}]
    _, _, gate = _score_pair_shared_gate(
        model, _Tokenizer(), prompt, chosen, rejected, "cpu", 128
    )
    assert model.compute_calls == 1
    assert len(model.overrides) == 2
    assert model.overrides[0] is gate
    assert model.overrides[1] is gate


def test_domain_specific_subset_is_reversible_and_keeps_every_domain():
    indices, names, excluded = resolve_active_attributes("domain_specific")
    assert len(indices) == len(names) == 17
    assert len(excluded) == 6
    for domain in DOMAIN_NAMES:
        assert set(indices) & set(DOMAIN_ATTRIBUTE_INDICES[domain])
    assert attribute_selection_suffix("full") == ""
    assert attribute_selection_suffix("domain_specific") == "_asds"


def test_masked_gate_assigns_exactly_zero_mass_to_excluded_attributes():
    indices, _, excluded = resolve_active_attributes("domain_specific")
    gate = GatingNetwork(
        4, len(ATTRIBUTES), hidden_dim=8, n_hidden=1, logit_scale=2.0,
        active_attribute_indices=indices,
    )
    output = gate(torch.randn(3, 4))
    excluded_indices = [ATTRIBUTES.index(attribute) for attribute in excluded]
    torch.testing.assert_close(output[:, excluded_indices], torch.zeros(3, len(excluded_indices)))
    torch.testing.assert_close(output.sum(-1), torch.full((3,), 2.0))



def test_model_registry_resolves_entries_and_preserves_explicit_cli_values():
    config = {
        "model_registry": {
            "gemma": {
                "model_path": "registry/gemma",
                "model_family": "gemma2",
                "output_model_name": "released-gemma",
            }
        }
    }
    args = SimpleNamespace(
        model_key="gemma", model_path="section/default",
        model_family="auto", output_model_name="section-name",
    )
    apply_model_registry(args, config, argv=[])
    assert args.model_path == "registry/gemma"
    assert args.model_family == "gemma2"
    assert args.output_model_name == "released-gemma"

    explicit = SimpleNamespace(
        model_key="gemma", model_path="cli/model",
        model_family="auto", output_model_name=None,
    )
    apply_model_registry(explicit, config, argv=["--model_path", "cli/model"])
    assert explicit.model_path == "cli/model"
    assert explicit.model_family == "gemma2"


def test_negative_boolean_cli_flag_beats_yaml_value():
    args = SimpleNamespace(balance_domains=False)
    apply_section_overrides(
        args, {"balance_domains": True}, argv=["--no-balance_domains"],
    )
    assert args.balance_domains is False


def test_unknown_model_registry_key_fails_closed():
    args = SimpleNamespace(model_key="missing")
    try:
        apply_model_registry(args, {"model_registry": {}}, argv=[])
    except ValueError as error:
        assert "Unknown model_key" in str(error)
    else:
        raise AssertionError("Unknown model registry key was accepted.")


def test_stable_identifiers_are_order_independent_and_fit_signed_int64():
    first = _stable_int64_id({"prompt": "p", "response": "r"})
    second = _stable_int64_id({"response": "r", "prompt": "p"})
    assert first == second
    assert 0 <= first < 2 ** 63


def _checkpoint_args(**overrides):
    defaults = dict(
        multi_objective_dataset_name="scoring", temperature=2.0, n_steps=30000,
        seed=0, learning_rate=0.0005, weight_decay=0.0, n_hidden=1,
        hidden_size=64, dropout=0.1, batch_size=2048, logit_scale=2.0,
        domain_loss_weight=0.25, entropy_weight=0.02,
        load_balance_weight=0.05, debiasing_dims=[18, 20],
        corr_threshold=0.04, curriculum=False, balance_difficulties=False,
        balance_domains=True, learnable_logit_scale=False,
        entropy_floor_fraction=0.35, attribute_subset="full",
        exclude_attributes=[], checkpoint_tag=None, train_on_all=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_checkpoint_name_encodes_debiasing_and_refit_without_collisions():
    base = _checkpoint_args()
    debiased = shared_gate_checkpoint_filename(base, "base", "preferences", "reference")
    disabled = shared_gate_checkpoint_filename(base, "base", "preferences", "null")
    refit = shared_gate_checkpoint_filename(
        _checkpoint_args(train_on_all=True), "base", "preferences", "reference"
    )
    assert "_db18-20_ct0p04" in debiased
    assert "_dbnone" in disabled
    assert debiased != disabled
    assert refit.endswith("_refit.pt")
    assert debiasing_checkpoint_suffix([20, 18, 20], 0.04) == "_db18-20_ct0p04"


def test_shared_routing_config_rejects_legacy_checkpoints():
    valid = {"format_version": 2, "shared_prompt_gating": True}
    assert validate_shared_routing_config(valid) is valid
    for invalid in ({}, {"format_version": 1, "shared_prompt_gating": True}, None):
        try:
            validate_shared_routing_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid routing config was accepted: {invalid}")


def test_shared_gate_forward_requires_an_explicit_prompt_gate():
    model = RewardModelWithGating.__new__(RewardModelWithGating)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_return_dict=True, pad_token_id=0, hidden_size=2,
        model_type="llama", shared_prompt_gating=True,
    )
    model.model = _Backbone()
    model.regression_layer = nn.Linear(2, len(ATTRIBUTES), bias=False)
    model.reward_transform_matrix = nn.Parameter(
        torch.eye(len(ATTRIBUTES)), requires_grad=False,
    )
    model.gating = nn.Identity()
    input_ids = torch.tensor([[4, 5]])
    attention_mask = torch.ones_like(input_ids)

    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    except ValueError as error:
        assert "gating_output_override" in str(error)
    else:
        raise AssertionError("Candidate-conditioned routing was accepted by a shared-gate model.")

    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        gating_output_override=torch.ones(1, len(ATTRIBUTES)),
    )
    assert output.score.shape == (1,)


def test_inference_path_resolution_gives_explicit_cli_highest_priority():
    config = {
        "inference": {
            "model_path": "config/exact",
            "model_parent_dir": "config-parent",
            "model_name": "config-name",
        }
    }
    assert _resolve_inference_model_path(config, "cli/exact", None, None) == "cli/exact"
    assert _resolve_inference_model_path(config, None, "cli-parent", "cli-name") == "cli-parent/cli-name"
    assert _resolve_inference_model_path(config, None, None, None) == "config/exact"
