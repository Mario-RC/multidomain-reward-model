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
  --debiasing_dim -1 \
  --dataset_split train \
  --eval reward-bench \
  --device 0
```

> **Reference dataset and `debiasing_dim`:** The reference dataset is only used when `debiasing_dim >= 0`. If `debiasing_dim` is `-1` (disabled), the reference dataset will **not** be loaded or used, even if provided.
>
> `debiasing_dim` points to any attribute dimension whose influence you want to decorrelate from the rest. For each other dimension *d*, it finds the smallest penalty *p* such that `adjusted_d = d - p * target_dim` has a Spearman correlation with the target dimension below `corr_threshold`. The result is a `reward_transform_matrix` that subtracts the leaking influence of the chosen dimension before the gating network combines scores.
>
> Examples:
> - In ArmoRM's original setup, `debiasing_dim=4` pointed to `helpsteer-verbosity` to prevent longer responses from inflating all reward scores.
> - In a multi-domain setup, you could set `debiasing_dim` to a dominant dimension (e.g. a coherence or cultural attribute) if you observe it correlating too strongly with others in the reference dataset.

### Stage 3 Packaging Model
```bash
python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name null \
  --output_model_name multi-domain-rm-llama-3-8b-it
```
> `--reference_dataset_name` must match the value used during Stage 2 training so the correct checkpoint file is found. Pass `null` if Stage 2 was trained without a reference dataset.

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

Inspect inter-attribute and attribute-vs-length correlations in the scoring data. Helps decide whether `--debiasing_dim` is needed and which dimension to target.

```bash
python3 analyze_correlations.py \
  --dataset_path data/Multi-Domain-Data-Scoring.jsonl \
  --threshold 0.5
```

Output sections:
- **Attribute vs response length** — Spearman correlation between each attribute and total assistant response length (characters). Flags attributes where longer responses systematically score higher/lower.
- **Inter-attribute correlations** — Pairwise Spearman between all attributes that share non-null rows (within-domain only, since cross-domain scores are null).
- **Dimension dominance summary** — Which attributes appear in the most high-correlation pairs (candidates for `--debiasing_dim`).
- **Length bias warning** — Attributes whose length correlation exceeds the threshold.

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
python3 benchmark.py \
  --model_parent_dir model
```

Discovers all models in `model/` that have `results/eval.json` or `results/eval_baseline.json`. Output includes:
- **Comparison tables** — Preference accuracy, scoring regression, global score distribution.
- **CSVs** — Saved to `model/benchmark/`.
- **Plots** — Comparative plots in `model/benchmark/`, per-model plots in `model/<model_name>/results/plots/`.

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
│   └── gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_T10.0_N2000_seed0.pt
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
└── benchmark/
    ├── *.csv
    └── *.png
```

---

## Artifact Structure

- `model/embeddings/<model_name>/<dataset_name>/*.safetensors`
- `model/gating_network/gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_T10.0_N2000_seed0.pt`
- `model/regression_weights/<model_name>_<dataset_name>.pt`
- `model/<packaged_model_name>/`

---

## Credits

This work is based on the original [RLHFlow repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) (ArmoRM), but this `multidomain_model` folder documents and executes a custom adaptation focused on:

- custom multi-domain attributes,
- data from `multidomain_data_scoring`,
- and a more robust pipeline for local training.