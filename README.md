# Multi-Domain Model

This directory contains my custom version of ArmoRM to train a multi-objective reward model using custom data and a data preparation pipeline adapted to my workflow.

## Project Goal

The goal is to train a reward model in three stages:

1. **Stage 1 (Multi-objective regression):** Extract embeddings from conversations and adjust weights per attribute.
2. **Stage 2 (Gating network):** Learn to combine the objectives into a final preference score.
3. **Stage 3 (Packaging):** Merge Stage 1 regression weights and Stage 2 gating weights into a final packaged reward model for inference.

---

## Data Source

The multi-domain data (Multi-Domain-Data-Scoring.jsonl & Multi-Domain-Data-Preference-Pairs.jsonl) come from:

- https://github.com/mestecha/multidomain_data_scoring

### Datasets used

- **Multi-objective data:** [`Multi-Domain-Data-Scoring`](https://github.com/mestecha/multidomain_data_scoring/tree/main)
- **Preference data:** [`Multi-Domain-Data-Preference-Pairs`](https://github.com/mestecha/multidomain_data_scoring/tree/main)
- **Reference data:** [`RLHFlow/UltraFeedback-preference-standard`](https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard)
- **Reward bench:** [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench)

---

## Base Models

The following base reward models have been used in this project:

- **Llama3:** [`sfairXC/FsfairX-LLaMA3-RM-v0.1`](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
- **Gemma2:** [`sfairXC/FsfairX-Gemma2-RM-v0.1`](https://huggingface.co/sfairXC/FsfairX-Gemma2-RM-v0.1)
- **Qwen3:** [`nvidia/Qwen3-Nemotron-8B-BRRM`](https://huggingface.co/nvidia/Qwen3-Nemotron-8B-BRRM)

---

## Working Attributes

This version uses **23 custom attributes** defined in `attributes.py` (single source of truth, imported by all scripts):

### Coherence (`co_`)

- `co_discourse_structure`
- `co_logical_consistency`
- `co_mutual_grounding`
- `co_overall_coherence_score`
- `co_temporal_causal_coherence`
- `co_topic_coherence`

### Commonsense (`cs_`)

- `cs_causality`
- `cs_coherence`
- `cs_consistency`
- `cs_desire`
- `cs_empathy`
- `cs_reaction`

### Empathy (`em_`)

- `em_emotional_awareness`
- `em_emotional_validation`
- `em_helpful_response`
- `em_overall_empathy_score`
- `em_perspective_taking`
- `em_supportive_engagement`

### Multicultural (`mu_`)

- `mu_coherence`
- `mu_cultural_specificity`
- `mu_cultural_value`
- `mu_empathy`
- `mu_naturalness`

> Note: These 23 attributes are the regression targets for Stage 1.

---

## Quickstart Execution Flow

```bash
pip install -r requirements.txt
```

> Recommended: install `flash-attn` to speed up attention.

Base script: `mdorm.sh`

```bash
./mdorm.sh
```

`mdorm.sh` is intentionally fixed to Llama3 defaults for a stable baseline run.

### Stage 1 prepare
```bash
python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Scoring \
  --output_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 1 train
```bash
python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train
```

### Stage 2 prepare (preference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Preference-Pairs \
  --output_dataset_name Multi-Domain-Data-Preference-Pairs \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --output_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reward-bench eval data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path allenai/reward-bench \
  --output_dataset_name reward-bench \
  --dataset_split filtered \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 train
```bash
python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name null \
  --debiasing_dims -1 \
  --temperature 10.0 \
  --n_steps 2000 \
  --seed 0 \
  --eval_every 200 \
  --patience 5 \
  --dataset_split train \
  --eval reward-bench \
  --device 0
```

Additional hyperparameters (defaults shown):
- `--learning_rate 0.001` — AdamW learning rate
- `--weight_decay 0.0` — L2 regularization
- `--n_hidden 3` — Hidden layers in gating MLP
- `--hidden_size 1024` — Hidden layer dimension
- `--dropout 0.2` — Dropout probability
- `--logit_scale 1.0` — Post-softmax scaling factor
- `--eval_every 200` — Validation frequency (steps)
- `--patience 5` — Early stopping patience (based on val_loss)

> **Reference dataset and `debiasing_dims`:** The reference dataset is only used when `debiasing_dims` contains indices >= 0. If `debiasing_dims` is `-1` (disabled), the reference dataset will **not** be loaded or used, even if provided.
>
> `debiasing_dims` accepts one or more attribute dimension indices whose influence you want to decorrelate from the rest. For each target dimension and each other dimension *d*, it finds the smallest penalty *p* such that `adjusted_d = d - p * target_dim` has a Spearman correlation with the target dimension below `corr_threshold`. The result is a `reward_transform_matrix` that subtracts the leaking influence of the chosen dimensions before the gating network combines scores.
>
> Examples:
> - In ArmoRM's original setup, `debiasing_dims 4` pointed to `helpsteer-verbosity` to prevent longer responses from inflating all reward scores.
> - In a multi-domain setup, you could set `debiasing_dims` to one or more dominant dimensions (e.g. `--debiasing_dims 21 18` for cultural and empathy attributes) if you observe them correlating too strongly with others in the reference dataset.

### Stage 3 Packaging Model
```bash
python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name null \
  --temperature 10.0 \
  --n_steps 2000 \
  --seed 0 \
  --output_model_name multi-domain-rm-llama-3-8b-it
```
> `--reference_dataset_name`, `--temperature`, `--n_steps` and `--seed` must match the values used during Stage 2 training so the correct checkpoint file is found. Pass `null` for reference_dataset_name if Stage 2 was trained without a reference dataset.

### Evaluate the packaged model
```bash
python3 evaluate.py \
  --model_name multi-domain-rm-llama-3-8b-it
```
Results are auto-saved to `model/multi-domain-rm-llama-3-8b-it/results/eval.json`.

### Run quick prediction comparison
```bash
python3 predict.py \
  --model_name multi-domain-rm-llama-3-8b-it
```

### Analyze attribute correlations

Inspect inter-attribute and attribute-vs-length correlations in the scoring data. Helps decide whether `--debiasing_dims` is needed and which dimensions to target.

```bash
python3 analyze_correlations.py \
  --dataset_path data/Multi-Domain-Data-Scoring.jsonl \
  --threshold 0.3
```

Output sections:
- **Attribute statistics** — Unique values, range, mean, std per attribute. Flags low-variance attributes (std < 0.10).
- **Attribute vs response length** — Spearman correlation between each attribute and response length. Flags length-biased attributes.
- **Inter-attribute correlations** — Pairwise Spearman between all within-domain attribute pairs.
- **Within-domain correlation matrices** — Full NxN heatmap per domain with high-correlation markers.
- **PCA dimensionality analysis** — Effective independent dimensions per domain (eigenvalue decomposition).
- **Dimension dominance summary** — Which attributes appear in the most high-correlation pairs.
- **Debiasing recommendations** — Actionable suggestions: low-variance dims, redundant pairs, length-biased dims. Outputs the attribute indices to use with `--debiasing_dims` in `stage-2_train.py`.

### Evaluate baseline (no regression)

Evaluate a base reward model using its native reward score (no stage-1 regression weights). Use `--model_name` to save results as `eval_baseline.json` inside the corresponding packaged model's results directory.

```bash
# Scalar RM — scoring + preference (LLaMA3, Gemma2)
python3 evaluate_baseline.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --no_regression \
  --model_name multi-domain-rm-llama-3-8b-it

# Generative judge — preference only (BRRM)
python3 evaluate_baseline.py \
  --model_path nvidia/Qwen3-Nemotron-8B-BRRM \
  --generative_judge --skip_scoring \
  --model_name multi-domain-rm-qwen-3-8b-it
```
Results are saved to `model/<model_name>/results/eval_baseline.json`.

### Compare models

Load pre-computed results from all models and produce side-by-side comparison tables, CSVs, and plots.

```bash
python3 compare_models.py \
  --model_parent_dir model \
  --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it multi-domain-rm-qwen-3-8b-it
```

Discovers all models in `model/` that have `results/eval.json` or `results/eval_baseline.json`. Output includes:
- **Comparison tables** — Preference accuracy, scoring regression, global score distribution.
- **CSVs** — Saved to `model/compare_models/`.
- **Plots** — Comparative plots in `model/compare_models/`, per-model plots in `model/<model_name>/results/plots/`.

---

## Alternative Flow: `config.yaml`

Instead of hardcoded CLI parameters, you can use `config.yaml` to configure the pipeline. Each stage has its own flat section with `model_path`, `model_family`, and all relevant parameters. Change them directly before each training run.

> CLI arguments still override `config.yaml` values when explicitly provided.

### Config-driven commands

All scripts accept `--config_path config.yaml` (default) to read their corresponding section. For example:

```bash
python3 stage-1_prepare.py --config_path config.yaml
```

---

## Model Directory Tree

```text
model/
├── embeddings/
│   └── <model_name>/
│       ├── <multi_objective_dataset_name>-<split>/
│       │   └── <multi_objective_dataset_name>-<split>.safetensors
│       │
│       ├── reward-bench-filtered/
│       │   └── reward-bench-filtered.safetensors
│       │
│       ├── <preference_dataset_name>-<split>/
│       │   └── <preference_dataset_name>-<split>.safetensors
│       │
│       └── <reference_dataset_name>-<split>/
│           └── <reference_dataset_name>-<split>.safetensors
│
├── gating_network/
│   └── gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_t10.0_n2000_seed0.pt
│
├── regression_weights/
│   └── <model_name>_<multi_objective_dataset_name>.pt
│
├── multi-domain-rm-<model_name>/
│   ├── config.json
│   ├── model-00001-of-0000X.safetensors
│   ├── ...
│   └── results/
│       ├── eval.json
│       ├── eval_baseline.json
│       └── plots/
│
└── compare_models/
    ├── *.csv
    └── *.png
```

---

## Artifact Structure

- `model/embeddings/<model_name>/<dataset_name>/*.safetensors`
- `model/gating_network/gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_t10.0_n2000_seed0.pt`
- `model/regression_weights/<model_name>_<dataset_name>.pt`
- `model/<packaged_model_name>/`

---

## Credits

This work is based on the original [RLHFlow repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) (ArmoRM), but this `multidomain_model` folder documents and executes a custom adaptation focused on:

- custom multi-domain attributes,
- data from `multidomain_data_scoring`,
- and a more robust pipeline for local training.