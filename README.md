# Multi-Domain Reward Model

This directory contains a multi-objective reward model that evaluates responses across four complementary domains: **Coherence**, **Commonsense**, **Empathy** and **Multicultural**. The model learns 23 fine-grained attributes spanning these domains and combines them through a prompt-conditioned gating network to produce a single preference score, enabling reward evaluation that captures domain-specific nuances.

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
- **Mistral:** [`weqweasdas/RM-Mistral-7B`](https://huggingface.co/weqweasdas/RM-Mistral-7B)

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

Additional per-model scripts are available for full train/package/evaluate runs:

- `mdorm_llama3.sh`
- `mdorm_gemma2.sh`
- `mdorm_qwen3.sh`
- `mdorm_mistral.sh`

### Stage 1 prepare
```bash
python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \      # Base reward model to extract embeddings from
  --model_family llama3 \                            # Architecture family (llama3, gemma2, qwen3, mistral)
  --dataset_path data/dataset/Multi-Domain-Data-Scoring \    # Path to multi-objective scoring dataset
  --output_dataset_name Multi-Domain-Data-Scoring \  # Name for saved embeddings
  --dataset_split train \                            # Dataset split to process
  --n_shards 1 \                                     # Total shards (for parallel processing)
  --shard_idx 1 \                                    # Current shard index
  --device 0                                         # GPU device index
```

### Stage 1 train
```bash
python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \               # Base reward model (used to name outputs)
  --model_family llama3 \                                     # Architecture family
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \  # Name of pre-computed embeddings
  --dataset_split train                                       # Split to train on
```

> **80pct vs 100pct weights:** Stage 1 splits the training data 80/20. It sweeps Ridge regression alphas on the 80% split, picks the best alpha by validation MSE on the 20% split, and then **retrains on 100% of the data** with that best alpha. Two weight files are saved:
> - `_100pct.pt` — final weights retrained on all data with the best alpha (used by default in Stage 2 and Stage 3).
> - `_80pct.pt` — weights from the validation-best model (80% training split only, useful as a sanity check).
>
> Subsequent stages auto-resolve to `_100pct.pt` unless `--stage_1_weights_path` is explicitly passed.

### Stage 2 prepare (preference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \               # Base reward model
  --model_family llama3 \                                     # Architecture family
  --dataset_path data/dataset/Multi-Domain-Data-Preference-Pairs \    # Input preference pairs dataset
  --output_dataset_name Multi-Domain-Data-Preference-Pairs \  # Name for saved embeddings
  --dataset_split train \                                     # Dataset split to process
  --n_shards 1 \                                              # Total shards
  --shard_idx 1 \                                             # Current shard index
  --device 0                                                  # GPU device index
```

### Stage 2 prepare (reference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \               # Base reward model
  --model_family llama3 \                                     # Architecture family
  --dataset_path RLHFlow/UltraFeedback-preference-standard \  # Reference dataset (HuggingFace)
  --output_dataset_name UltraFeedback-preference-standard \   # Name for saved embeddings
  --dataset_split train \                                     # Dataset split to process
  --n_shards 1 \                                              # Total shards
  --shard_idx 1 \                                             # Current shard index
  --device 0                                                  # GPU device index
```

### Stage 2 prepare (reward-bench eval data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \  # Base reward model
  --model_family llama3 \                        # Architecture family
  --dataset_path allenai/reward-bench \          # RewardBench evaluation dataset
  --output_dataset_name reward-bench \           # Name for saved embeddings
  --dataset_split filtered \                     # Use filtered split
  --n_shards 1 \                                 # Total shards
  --shard_idx 1 \                                # Current shard index
  --device 0                                     # GPU device index
```

### Stage 2 train
```bash
python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \                   # Base reward model (used for naming)
  --model_family llama3 \                                         # Architecture family
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \      # Pre-computed embeddings for scoring data
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \  # Pre-computed embeddings for preference pairs
  --reference_dataset_name null \                                 # Reference dataset for debiasing (null = disabled)
  --debiasing_dims 18 20 22 \                                     # Dims to decorrelate (mu_coherence, mu_cultural_value, mu_naturalness)
  --temperature 2.0 \                                             # Softmax temperature for gating weights
  --n_steps 30000 \                                               # Training steps
  --seed 0 \                                                      # Random seed
  --n_hidden 1 \                                                  # Hidden layers in gating MLP
  --hidden_size 64 \                                              # Hidden layer dimension
  --learning_rate 0.0005 \                                        # AdamW learning rate
  --weight_decay 0.0 \                                            # L2 regularization
  --dropout 0.1 \                                                 # Dropout probability
  --batch_size 2048 \                                             # Training batch size
  --corr_threshold 0.04 \                                         # Max allowed correlation after debiasing
  --logit_scale 2.0 \                                             # Post-softmax scaling factor
  --eval_every 200 \                                              # Validation frequency (steps)
  --patience 15 \                                                 # Early stopping patience (based on val_loss)
  --curriculum \                                                  # Enable phased curriculum learning (easy → easy+medium → all)
  --curriculum_phase1_frac 0.20 \                                 # Fraction of n_steps for easy-only phase
  --curriculum_phase2_frac 0.50 \                                 # Fraction of n_steps to end easy+medium phase
  --dataset_split train \                                         # Split to train on
  --eval reward-bench \                                           # Eval dataset name
  --device 0                                                      # GPU device index
```

> **`--stage_1_weights_path` (optional):** Override which Stage 1 regression weights to load. If omitted, auto-resolves to `model/regression_weights/{model_name}_{multi_objective_dataset_name}_100pct.pt`. If a bare filename is given (no `/`), the `_100pct` suffix is appended automatically unless the name already ends with `_100pct.pt` or `_80pct.pt`.
>
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
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \                   # Base reward model to package
  --model_family llama3 \                                         # Architecture family
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \      # Used to locate stage-1 regression weights
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \  # Used to locate stage-2 checkpoint
  --reference_dataset_name null \                                 # Must match value used in stage-2 training
  --temperature 2.0 \                                             # Must match stage-2 value
  --n_steps 30000 \                                               # Must match stage-2 value
  --seed 0 \                                                      # Must match stage-2 value
  --n_hidden 1 \                                                  # Must match stage-2 value
  --hidden_size 64 \                                              # Must match stage-2 value
  --learning_rate 0.0005 \                                        # Must match stage-2 value
  --weight_decay 0.0 \                                            # Must match stage-2 value
  --dropout 0.1 \                                                 # Must match stage-2 value
  --batch_size 2048 \                                             # Must match stage-2 value
  --corr_threshold 0.04 \                                         # Must match stage-2 value
  --logit_scale 2.0 \                                             # Must match stage-2 value
  --curriculum \                                                  # Must match stage-2 value (adds _cv suffix to checkpoint name)
  --output_model_name multi-domain-rm-llama-3-8b-it               # Name for the packaged HuggingFace model
```
> All hyperparameters (`--temperature`, `--n_steps`, `--seed`, `--learning_rate`, `--weight_decay`, `--n_hidden`, `--hidden_size`, `--dropout`, `--batch_size`, `--corr_threshold`, `--logit_scale`) must match the values used during Stage 2 training so the correct checkpoint file is found. Pass `null` for reference_dataset_name if Stage 2 was trained without a reference dataset. If Stage 2 was trained with `--curriculum`, add `--curriculum` here too so the `_cv` suffix is included in the checkpoint filename.
>
> **`--stage_1_weights_path` (optional):** Same auto-resolution logic as Stage 2 — defaults to `_100pct.pt` if omitted.

### Evaluate the packaged model
```bash
python3 evaluate.py \
  --model_name multi-domain-rm-llama-3-8b-it \  # Name of the packaged model to evaluate
  --eval data/test                              # Optional: cultural test data directory
```
Results are auto-saved to `model/<model_name>/results/eval.json` for each model. Per-model plots are generated in `model/<model_name>/results/plots/`.

> **Dual scoring evaluation (80pct / 100pct):** If the `_80pct.pt` weights file exists alongside the `_100pct.pt` used during packaging, `evaluate.py` evaluates scoring with **both** weight sets. Results are saved as `scoring_80pct` and `scoring_100pct` in the output JSON. The 80pct result reflects performance of the validation-best model; the 100pct result reflects the final model retrained on all data. Preference and cultural evaluations always use the 100pct weights (packaged in the model).

### Run quick prediction comparison
```bash
python3 predict.py \
  --model_name multi-domain-rm-llama-3-8b-it  # Name of the packaged model to run predictions with
```

### Analyze attribute correlations

Inspect inter-attribute and attribute-vs-length correlations in the scoring data. Helps decide whether `--debiasing_dims` is needed and which dimensions to target.

```bash
python3 analyze_correlations.py \
  --dataset_path data/dataset/Multi-Domain-Data-Scoring.jsonl \  # Path to scoring data JSONL
  --threshold 0.3                                        # Correlation threshold to flag high-correlation pairs
```

Output sections:
- **Attribute statistics** — Unique values, range, mean, std per attribute. Flags low-variance attributes (std < 0.10).
- **Attribute vs response length** — Spearman correlation between each attribute and response length. Flags length-biased attributes.
- **Inter-attribute correlations** — Pairwise Spearman between all within-domain attribute pairs.
- **Within-domain correlation matrices** — Full NxN heatmap per domain with high-correlation markers.
- **PCA dimensionality analysis** — Effective independent dimensions per domain (eigenvalue decomposition).
- **Dimension dominance summary** — Which attributes appear in the most high-correlation pairs.
- **Debiasing recommendations** — Actionable suggestions: low-variance dims, redundant pairs, length-biased dims. Outputs the attribute indices to use with `--debiasing_dims` in `stage-2_train.py`.

### Evaluate baseline

Evaluate a base reward model (as-is from HuggingFace) using its native reward score. Use `--model_name` to save results as `eval_baseline.json` inside the corresponding packaged model's results directory.

```bash
# Scalar RM — scoring + preference + cultural (LLaMA3, Gemma2, Mistral)
python3 evaluate_baseline.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \  # Base reward model path
  --eval data/test \                              # Optional: cultural test data directory
  --model_name multi-domain-rm-llama-3-8b-it      # Save results under this model's directory

# Generative judge — preference only (BRRM)
python3 evaluate_baseline.py \
  --model_path nvidia/Qwen3-Nemotron-8B-BRRM \  # Base reward model path
  --generative_judge \                          # Use generative judge mode
  --skip_scoring \                              # Skip scoring, preference only
  --model_name multi-domain-rm-qwen-3-8b-it     # Save results under this model's directory
```
Results are saved to `model/<model_name>/results/eval_baseline.json`. Per-model plots are generated in `model/<model_name>/results/plots/`.

### Compare models

Load pre-computed results from all models and produce side-by-side comparison tables, CSVs, and plots.

```bash
python3 compare_models.py \
  --model_parent_dir model \                                                                         # Parent directory containing model subdirectories
  --no_baselines \                                                                                   # Skip loading eval_baseline.json (optional)
  --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it multi-domain-rm-qwen-3-8b-it multi-domain-rm-mistral-7b-it  # Models to compare
```

Discovers all models in `model/` that have `results/eval.json` or `results/eval_baseline.json`. Output includes:
- **Comparison tables** — Preference accuracy, scoring regression, global score distribution.
- **CSVs** — Saved to `model/compare_models/`.
- **Plots** — Comparative plots in `model/compare_models/`.

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
│   └── gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_t2.0_n30000_seed0_le0.0005_we0.0_n_1_hi64_dr0.1_ba2048_co0.04_lo2.0.pt
│
├── regression_weights/
│   ├── <model_name>_<multi_objective_dataset_name>_100pct.pt
│   └── <model_name>_<multi_objective_dataset_name>_80pct.pt
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
- `model/gating_network/gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>_ref_<reference_dataset_name>_t2.0_n30000_seed0_le0.0005_we0.0_n_1_hi64_dr0.1_ba2048_co0.04_lo2.0.pt`
- `model/regression_weights/<model_name>_<dataset_name>_100pct.pt`
- `model/regression_weights/<model_name>_<dataset_name>_80pct.pt`
- `model/<packaged_model_name>/`

---

## Credits

This work is based on the original [RLHFlow repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) (ArmoRM), but this `multidomain_model` folder documents and executes a custom adaptation focused on:

- custom multi-domain attributes,
- data from `multidomain_data_scoring`,
- and a more robust pipeline for local training.
