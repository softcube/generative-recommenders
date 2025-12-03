# Generative Recommenders

Repository hosting code for ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations`` ([ICML'24 paper](https://proceedings.mlr.press/v235/zhai24a.html)) and related code, where we demonstrate that the ubiquitously used classical deep learning recommendation paradigm (DLRMs) can be reformulated as a generative modeling problem (Generative Recommenders or GRs) to overcome known compute scaling bottlenecks, propose efficient algorithms such as HSTU and M-FALCON to accelerate training and inference for large-scale sequential models by 10x-1000x, and demonstrate scaling law for the first-time in deployed, billion-user scale recommendation systems.

## Getting started

We recommend using `requirements.txt`. This has been tested with Ubuntu 22.04, CUDA 12.4, and Python 3.10.

```bash
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyTorch based on official instructions. Then,

```bash
pip3 install gin-config pandas fbgemm_gpu torchrec tensorboard
```

## Experiments

### Public Experiments

To reproduce the public experiments in our paper (traditional sequential recommender setting, Section 4.1.1) on MovieLens and Amazon Reviews in the paper, please follow these steps:

#### Download and preprocess data.

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

A GPU with 24GB or more HBM should work for most datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

Other configurations are included in configs/ml-1m, configs/ml-20m, and configs/amzn-books to make reproducing these experiments easier.

#### Verify results.

By default we write experimental logs to exps/. We can launch tensorboard with something like the following:

```bash
tensorboard --logdir ~/generative-recommenders/exps/ml-1m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/ml-20m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/amzn-books-l50/ --port 24001 --bind_all
```

With the provided configuration (.gin) files, you should be able to reproduce the following results (verified as of 04/15/2024):

**MovieLens-1M (ML-1M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------| --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2853           | 0.1603          | 0.5474          | 0.2185          | 0.7528          | 0.2498          |
| BERT4Rec      | 0.2843 (-0.4%)   | 0.1537 (-4.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2811 (-1.5%)   | 0.1648 (+2.8%)  |                 |                 |                 |                 |
| HSTU          | 0.3097 (+8.6%)   | 0.1720 (+7.3%)  | 0.5754 (+5.1%)  | 0.2307 (+5.6%)  | 0.7716 (+2.5%)  | 0.2606 (+4.3%)  |
| HSTU-large    | **0.3294 (+15.5%)**  | **0.1893 (+18.1%)** | **0.5935 (+8.4%)**  | **0.2481 (+13.5%)** | **0.7839 (+4.1%)**  | **0.2771 (+10.9%)** |

**MovieLens-20M (ML-20M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2889           | 0.1621          | 0.5503          | 0.2199          | 0.7661          | 0.2527          |
| BERT4Rec      | 0.2816 (-2.5%)   | 0.1703 (+5.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2813 (-2.6%)   | 0.1730 (+6.7%)  |                 |                 |                 |                 |
| HSTU          | 0.3273 (+13.3%)  | 0.1895 (+16.9%) | 0.5889 (+7.0%)  | 0.2473 (+12.5%) | 0.7952 (+3.8%)  | 0.2787 (+10.3%) |
| HSTU-large    | **0.3556 (+23.1%)**  | **0.2098 (+29.4%)** | **0.6143 (+11.6%)** | **0.2671 (+21.5%)** | **0.8074 (+5.4%)**  | **0.2965 (+17.4%)** |

**Amazon Reviews (Books)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------|---------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.0306           | 0.0164          | 0.0754          | 0.0260          | 0.1431          | 0.0362          |
| HSTU          | 0.0416 (+36.4%)  | 0.0227 (+39.3%) | 0.0957 (+27.1%) | 0.0344 (+32.3%) | 0.1735 (+21.3%) | 0.0461 (+27.7%) |
| HSTU-large    | **0.0478 (+56.7%)**  | **0.0262 (+60.7%)** | **0.1082 (+43.7%)** | **0.0393 (+51.2%)** | **0.1908 (+33.4%)** | **0.0517 (+43.2%)** |

for all three tables above, the ``SASRec`` rows are based on [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) but with the original binary cross entropy loss
replaced with sampled softmax losses proposed in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039). These rows are reproducible with ``configs/*/sasrec-*-final.gin``.
The ``BERT4Rec`` and ``GRU4Rec`` rows are based on results reported by [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) -
note that the comparison slightly favors these two, due to them using full negatives whereas the other rows used 128/512 sampled negatives. The ``HSTU`` and ``HSTU-large`` rows are based on [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152); in particular, HSTU rows utilize identical configurations as SASRec. ``HSTU`` and ``HSTU-large`` results can be reproduced with ``configs/*/hstu-*-final.gin``.

### Custom configs and training modes

Beyond the paper configs, this repo includes several small, self-contained .gin files to make it easy to experiment with different architectures and loss functions on the same data.

#### Architectures: `train_fn.main_module`

The sequential encoder architecture is controlled by:

```gin
train_fn.main_module = "SASRec"  # or "HSTU"
```

- `train_fn.main_module = "SASRec"`  
  Uses the SASRec Transformer encoder:
  - Configured via `sasrec_encoder.*`:
    - `sasrec_encoder.num_blocks` – number of Transformer layers.
    - `sasrec_encoder.num_heads` – attention heads per layer.
    - `sasrec_encoder.ffn_hidden_dim` – FFN hidden size.
    - `sasrec_encoder.ffn_activation_fn` – `"relu"` or `"gelu"`.
    - `sasrec_encoder.ffn_dropout_rate` – dropout used in attention/FFN.
  - Good as a strong, standard baseline and to mimic small GPT-like models.

- `train_fn.main_module = "HSTU"`  
  Uses the HSTU encoder from the ICML’24 paper:
  - Configured via `hstu_encoder.*`:
    - `hstu_encoder.num_blocks` – number of HSTU layers.
    - `hstu_encoder.num_heads` – attention heads.
    - `hstu_encoder.dqk` / `hstu_encoder.dv` – attention and linear dimensions.
    - `hstu_encoder.linear_dropout_rate` / `hstu_encoder.attn_dropout_rate`.
    - `hstu_encoder.normalization`, `hstu_encoder.linear_config`, `hstu_encoder.linear_activation`, `hstu_encoder.enable_relative_attention_bias`, etc.
  - This is the more expressive, time-aware architecture used for the main HSTU results in the paper.

All other pieces (loss, similarity function, data loaders) work with either main module. You can compare architectures by changing only `train_fn.main_module` and the corresponding `sasrec_encoder.*` or `hstu_encoder.*` hyperparameters, keeping the rest of the config identical.

#### Losses: `train_fn.loss_module`

The training objective is controlled by:

```gin
train_fn.loss_module = "SampledSoftmaxLoss"  # or "FullSoftmaxLoss", "BCELoss"
```

Supported values:

- `train_fn.loss_module = "SampledSoftmaxLoss"`  
  Uses sampled softmax, as in the paper configs:
  - Controlled by:
    - `train_fn.num_negatives` – number of negatives sampled per positive.
    - `train_fn.temperature` – divides logits by this scalar; must be > 0.
    - `train_fn.sampling_strategy` – `"in-batch"` or `"local"`:
      - `"in-batch"`: negatives are other items from the same batch.
      - `"local"`: negatives are sampled from the whole catalog via the item embedding table.
  - This is the default for large catalogs (MovieLens-20M, Amazon Books) and is what the paper uses.

- `train_fn.loss_module = "FullSoftmaxLoss"`  
  Uses an exact softmax over **all items** in the embedding table:
  - Ignores `train_fn.num_negatives` and does not use explicit negative sampling.
  - Still supports `train_fn.temperature > 0` to scale logits.
  - Recommended for smaller catalogs (e.g., ML-1M) and sanity-check experiments where you want a true cross-entropy objective over all items.
  - In the training logs, `sampling_strategy` will show up as `"no-explicit-negatives"` in the experiment name.

- `train_fn.loss_module = "BCELoss"`  
  Binary cross-entropy over (positive, negative) pairs:
  - Uses the same `NegativesSampler` as sampled softmax (`sampling_strategy = "in-batch"` or `"local"`), but with **one** sampled negative per positive.
  - Requires `train_fn.temperature = 1.0`.
  - Simpler than sampled softmax; useful as a pairwise ranking baseline.

For large-scale experiments, `SampledSoftmaxLoss` with `"local"` or `"in-batch"` negatives is the most scalable choice. For smaller datasets (e.g., ML-1M, Merlin) and model-comparison experiments, `FullSoftmaxLoss` lets you run a stricter, exact softmax objective at the cost of more compute. All three losses plug into the same training loop and metrics; you can switch between them by changing only `train_fn.loss_module` (and the relevant hyperparameters) in your .gin config.

#### Similarity: `train_fn.interaction_module_type`

The similarity function between user/query embeddings and item embeddings is controlled by:

```gin
train_fn.interaction_module_type = "DotProduct"  # or "MoL"
```

This selects which `SimilarityModule` is used inside the sequential encoder:

- `train_fn.interaction_module_type = "DotProduct"`  
  Uses a simple dot-product similarity:
  - Implemented by `DotProductSimilarity` (`research/rails/similarities/dot_product_similarity_fn.py`).
  - Score = `q · v_x` for query `q` and item embedding `v_x`.
  - Top‑K retrieval via `MIPSBruteForceTopK` (`train_fn.top_k_method = "MIPSBruteForceTopK"`), which computes dot products against all items and returns the top‑K.
  - This matches standard Transformer/LM setups and is the default in most configs (including GPT-like SASRec and HSTU on ML-1M).

- `train_fn.interaction_module_type = "MoL"`  
  Uses a Mixture-of-Logits (MoL) similarity:
  - Implemented by `MoLSimilarity` (`research/rails/similarities/mol/similarity_fn.py`).
  - Each query–item pair `(q, x)` has **multiple component logits** and a learned gating function:
    - `s(q, x) = Σ_p π_p(q, x) · ℓ_p(q, x)`, where `ℓ_p` are component logits and `π_p` are gating weights.
  - Constructed via `create_mol_interaction_module` with additional hyperparameters:
    - `dot_product_dimension`, `query_dot_product_groups`, `item_dot_product_groups`
    - MoL MLP sizes/dropouts via `create_mol_interaction_module.*` gin bindings.
    - Optional bfloat16 autocast via `get_similarity_function.bf16_training = True`.
  - Top‑K retrieval via `MoLBruteForceTopK`:

    ```gin
    train_fn.interaction_module_type = "MoL"
    train_fn.top_k_method = "MoLBruteForceTopK"
    ```

    which calls the MoL similarity to score all items and then returns the top‑K.

In practice:

- Use `"DotProduct"` when you want the simplest, fastest baseline (and to mimic GPT-like behavior).
- Use `"MoL"` when you want a more expressive similarity function (multiple components + gating), as in the RAILS papers. You can combine MoL with any of the losses (`SampledSoftmaxLoss`, `FullSoftmaxLoss`, `BCELoss`) and with either main module (`SASRec` or `HSTU`) by changing only `train_fn.interaction_module_type` and `train_fn.top_k_method` in your .gin config.

#### Retrieval: `train_fn.top_k_method`

The retrieval / evaluation backend is controlled by:

```gin
train_fn.top_k_method = "MIPSBruteForceTopK"  # or "MoLBruteForceTopK"
```

This affects **evaluation and inference**, not the training loss itself. It chooses how we score all candidate items when computing ranking metrics (HR, NDCG, MRR):

- `train_fn.top_k_method = "MIPSBruteForceTopK"`  
  - Implemented by `MIPSBruteForceTopK` (`research/rails/indexing/mips_top_k.py`).
  - Assumes a dot-product similarity (`DotProductSimilarity`).
  - Computes a dense matrix of dot products between query embeddings and item embeddings, then uses `torch.topk` to find the top‑K scores and item IDs.
  - Recommended when:
    - `train_fn.interaction_module_type = "DotProduct"`.
    - Catalog size is small/medium and brute-force scoring is acceptable.

- `train_fn.top_k_method = "MoLBruteForceTopK"`  
  - Implemented by `MoLBruteForceTopK` (`research/rails/indexing/mol_top_k.py`).
  - Uses the full `MoLSimilarity` scoring function (mixture-of-logits + gating).
  - Computes MoL scores for all items, then uses `torch.topk` to select top‑K.
  - Recommended when:
    - `train_fn.interaction_module_type = "MoL"`.
    - You want evaluation to be consistent with the MoL similarity used during training.

Under the hood, `get_top_k_module` in `research/indexing/utils.py` picks the appropriate `TopKModule` implementation based on `train_fn.top_k_method` and is used by `eval_metrics_v2_from_tensors`. You can switch between dot-product and MoL-style retrieval by adjusting only `train_fn.interaction_module_type` and `train_fn.top_k_method` in your .gin file; training losses and data loaders do not need to change.

#### 1. Quick ML-1M HSTU experiments

These configs run on MovieLens-1M with much smaller models and shorter training, useful for local sanity checks.

- **Mini HSTU, sampled softmax** (fast, uses sampled negatives)

  ```bash
  # Preprocess ML-1M once
  mkdir -p tmp/ && python3 preprocess_public_data.py

  # Train a small HSTU with sampled softmax on ML-1M
  CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/ml-1m/hstu-sampled-softmax-mini-train.gin \
    --master_port=12345
  ```

- **Mini HSTU, full softmax** (exact softmax over all items)

  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/ml-1m/hstu-full-softmax-mini-train.gin \
    --master_port=12345
  ```

  In this mode, `train_fn.loss_module = "FullSoftmaxLoss"` computes an exact softmax over the full item embedding table (no explicit negatives). This is useful to compare against sampled softmax on small catalogs like ML-1M.

#### 2. GPT-like SASRec baseline on ML-1M

`configs/ml-1m/sasrec-gpt-like-full-softmax-mini.gin` approximates a small GPT-style Transformer:

- 4 Transformer blocks, 2 heads, hidden size 1024.
- Dropout 0.01.
- FFN hidden dim 3072 with GELU.
- Full softmax loss over all items (`FullSoftmaxLoss`).

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/sasrec-gpt-like-full-softmax-mini.gin \
  --master_port=12345
```

This gives a dot-product SASRec model that is structurally close to a small GPT-2 block, trained on ML-1M with a full-softmax objective.

### Merlin session data (custom Parquet data)

We also provide a path to train on custom session data stored as Parquet files in `tmp/merlin`. This is intended for data exported from NVIDIA Merlin or similar tooling.

#### Expected Merlin Parquet format

For each split:

- `tmp/merlin/train.parquet`
- `tmp/merlin/valid.parquet`
- `tmp/merlin/test.parquet`

Each file should contain the following columns:

- `item_id_list_seq`: list of item IDs per session (e.g. `[922, 1141, 3617]`).
- `item_event_name_list_seq`: list of event-type IDs per session (e.g. `[3, 4, 4, 3, 3]`).
- `user_session`: a session or user identifier.
- `timestamp_first`: integer timestamp of the first event in the session.

#### Merlin preprocessing

We added `MerlinParquetDataProcessor` and a small entry script:

```bash
python3 preprocess_merlin_data.py
```

This will:

- Read `train/valid/test.parquet` from `tmp/merlin`.
- Remap item IDs to a contiguous range `[0..num_items-1]`.
- Remap event types from `item_event_name_list_seq` to `[0..num_event_types-1]`.
- Build per-session sequences:
  - `item_ids`   – remapped item IDs.
  - `ratings`    – remapped event-type IDs used as extra features.
  - `timestamps` – synthetic per-event timestamps (monotone per session).
  - `user_id`    – integer code for `user_session`.
- Write SASRec-format CSVs:
  - `tmp/merlin/sasrec_format_train.csv` (for training)
  - `tmp/merlin/sasrec_format_valid.csv` (for validation)
  - `tmp/merlin/sasrec_format_test.csv`  (for final test-only evaluation)

By default each row in the Parquet files is treated as **one full session sequence**
`[i1, ..., iT]`. If you prefer to explicitly materialize *all* prefix subsequences
before training (e.g. `[i1, i2]`, `[i1, i2, i3]`, ..., `[i1, ..., iT]`), you can
enable this at preprocessing time:

```bash
python3 preprocess_merlin_data.py --expand_sequences_to_prefixes
```

This sets `expand_sequences_to_prefixes=True` inside `MerlinParquetDataProcessor`,
so each original sequence `[i1, ..., iT]` generates multiple shorter rows. All
Merlin configs assume **no prefix expansion by default**
(`expand_sequences_to_prefixes=False`), so only use this flag if you intentionally
want that behavior.

Internally, these CSVs use the same `sequence_item_ids`, `sequence_ratings` and `sequence_timestamps` schema as the public MovieLens/Amazon pipelines, so they plug directly into `DatasetV2` and `RecoDataset`.

#### GPT-like SASRec on Merlin with event-type features

`configs/merlin/sasrec-gpt-like-full-softmax-mini.gin` configures:

- `train_fn.dataset_name = "merlin"` – use the Merlin data pipeline.
- Transformer encoder:
  - `train_fn.main_module = "SASRec"`
  - `train_fn.item_embedding_dim = 1024`
  - `sasrec_encoder.num_blocks = 4`
  - `sasrec_encoder.num_heads = 2`
  - `sasrec_encoder.ffn_hidden_dim = 3072`
  - `sasrec_encoder.ffn_activation_fn = "gelu"`
  - Dropout 0.01.
- Full-softmax loss:
  - `train_fn.loss_module = "FullSoftmaxLoss"`
  - `train_fn.temperature = 1.0`
  - `train_fn.interaction_module_type = "DotProduct"`
  - `train_fn.top_k_method = "MIPSBruteForceTopK"`
- Event types as extra input features:
  - `train_fn.use_rated_input_preproc = True`
  - `train_fn.rating_embedding_dim = 8`
  - `train_fn.num_ratings = 6`

With `use_rated_input_preproc=True`, the input preprocessor switches to `LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor`, which:

- Embeds items with dimension `item_embedding_dim`.
- Embeds event-type IDs (from `sequence_ratings`) with dimension `rating_embedding_dim`.
- Concatenates `[item_emb || rating_emb]` per timestep.
- Adds positional embeddings.

The training **target remains the same**: predict/rank items. Event types only affect the input representation.

Run Merlin training as:

```bash
cd /path/to/generative-recommenders
source .venv/bin/activate

# 1) Preprocess Merlin Parquet splits into SASRec CSVs
python3 preprocess_merlin_data.py

# 2) Train SASRec on Merlin with event-type features + full softmax
python3 main.py \
  --gin_config_file=configs/merlin/sasrec-gpt-like-full-softmax-mini.gin \
  --master_port=12345
```

For `dataset_name == "merlin"` the pipeline behaves as:

- Train loader: `sasrec_format_train.csv` (train.parquet only).
- Validation loader: `sasrec_format_valid.csv` (valid.parquet only).
- At the very end of training, a **single final test evaluation** is run using `sasrec_format_test.csv`, and metrics are logged to the console:
  - `NDCG@10`, `NDCG@50`, `HR@10`, `HR@50`, `MRR`, and the average test loss.

#### Experiment grid: Merlin configs

Under `configs/merlin` we provide a small grid of configs to explore how different architectural and loss choices behave on the same Merlin data:

- **1. SASRec + FullSoftmax, with event-type inputs**  
  - Config: `configs/merlin/sasrec-gpt-like-full-softmax-mini.gin`  
  - Model:
    - `train_fn.main_module = "SASRec"`
    - 4 layers, 2 heads, `item_embedding_dim = 1024`, FFN dim 3072, dropout 0.01.
  - Loss:
    - `train_fn.loss_module = "FullSoftmaxLoss"`
    - Exact softmax over all items, `train_fn.temperature = 1.0`.
  - Inputs:
    - `train_fn.use_rated_input_preproc = True`
    - `train_fn.rating_embedding_dim = 8`, `train_fn.num_ratings = 6`
    - Event types from `item_event_name_list_seq` are embedded and concatenated with item embeddings.
  - Usage:

    ```bash
    python3 main.py \
      --gin_config_file=configs/merlin/sasrec-gpt-like-full-softmax-mini.gin \
      --master_port=12345
    ```

- **2. SASRec + FullSoftmax, without event-type inputs**  
  - Config: `configs/merlin/sasrec-gpt-like-full-softmax-no-ratings-mini.gin`  
  - Model: same SASRec as (1).
  - Loss: same FullSoftmax as (1).
  - Inputs:
    - `train_fn.use_rated_input_preproc = False`
    - The encoder sees only item IDs + positions (no event-type embeddings).
  - Use this to measure the impact of event-type features by comparing to (1).

  ```bash
  python3 main.py \
    --gin_config_file=configs/merlin/sasrec-gpt-like-full-softmax-no-ratings-mini.gin \
    --master_port=12345
  ```

- **3. SASRec + SampledSoftmax, with event-type inputs**  
  - Config: `configs/merlin/sasrec-gpt-like-sampled-softmax-mini.gin`  
  - Model: same SASRec and rated inputs as (1).
  - Loss:
    - `train_fn.loss_module = "SampledSoftmaxLoss"`
    - `train_fn.num_negatives = 16`
    - `train_fn.temperature = 1.0`
    - `train_fn.sampling_strategy = "local"`
  - Use this to compare sampled softmax vs full softmax while keeping architecture and inputs fixed.

  ```bash
  python3 main.py \
    --gin_config_file=configs/merlin/sasrec-gpt-like-sampled-softmax-mini.gin \
    --master_port=12345
  ```

- **4. HSTU + FullSoftmax, with event-type inputs**  
  - Config: `configs/merlin/hstu-gpt-like-full-softmax-mini.gin`  
  - Model:
    - `train_fn.main_module = "HSTU"`
    - `hstu_encoder.num_blocks = 4`, `hstu_encoder.num_heads = 2`
    - `hstu_encoder.dqk = 1024` (attention_dim)
    - `hstu_encoder.dv = 3072` (linear_dim)
    - `hstu_encoder.linear_dropout_rate = 0.01`
    - `hstu_encoder.attn_dropout_rate = 0.01`
    - `train_fn.item_embedding_dim = 1024`
  - Loss:
    - `train_fn.loss_module = "FullSoftmaxLoss"`
    - `train_fn.temperature = 1.0`
  - Inputs:
    - `train_fn.use_rated_input_preproc = True`
    - `train_fn.rating_embedding_dim = 8`, `train_fn.num_ratings = 6`
  - Use this to compare SASRec vs HSTU at similar scale, both with full softmax and event-type features.

  ```bash
  python3 main.py \
    --gin_config_file=configs/merlin/hstu-gpt-like-full-softmax-mini.gin \
    --master_port=12345
  ```

- **5. HSTU + FullSoftmax, without event-type inputs**  
  - Config: `configs/merlin/hstu-gpt-like-full-softmax-no-ratings-mini.gin`  
  - Model: same HSTU as (4), but:
    - `train_fn.use_rated_input_preproc = False`
  - Inputs: items + positions only (no event-type embeddings).
  - Use this to measure the impact of event-type features for HSTU by comparing to (4).

  ```bash
  python3 main.py \
    --gin_config_file=configs/merlin/hstu-gpt-like-full-softmax-no-ratings-mini.gin \
    --master_port=12345
  ```

- **6. HSTU + SampledSoftmax, with event-type inputs**  
  - Config: `configs/merlin/hstu-gpt-like-sampled-softmax-mini.gin`  
  - Model: same HSTU and rated inputs as (4).
  - Loss:
    - `train_fn.loss_module = "SampledSoftmaxLoss"`
    - `train_fn.num_negatives = 16`
    - `train_fn.temperature = 1.0`
    - `train_fn.sampling_strategy = "local"`
  - Use this to compare sampled softmax vs full softmax for HSTU while keeping architecture and inputs fixed.

  ```bash
  python3 main.py \
    --gin_config_file=configs/merlin/hstu-gpt-like-sampled-softmax-mini.gin \
    --master_port=12345
  ```

All Merlin configs share the same data behavior:

- Train on `tmp/merlin/train.parquet` (via `sasrec_format_train.csv`).
- Validate on `tmp/merlin/valid.parquet` (via `sasrec_format_valid.csv`).
- After training, run a single final test evaluation on `tmp/merlin/test.parquet` (via `sasrec_format_test.csv`) and print `NDCG@5/10/50`, `HR@5/10/50`, `MRR`, and the average test loss.

#### Running the full Merlin grid and picking the best config

To run all Merlin configs sequentially and compare their TEST metrics in a single table, you can use `run_merlin_grid.py`. This script:

- Trains and evaluates these configs one after another:
  - `configs/merlin/sasrec-gpt-like-full-softmax-mini.gin`
  - `configs/merlin/sasrec-gpt-like-full-softmax-no-ratings-mini.gin`
  - `configs/merlin/sasrec-gpt-like-sampled-softmax-mini.gin`
  - `configs/merlin/hstu-gpt-like-full-softmax-mini.gin`
  - `configs/merlin/hstu-gpt-like-full-softmax-no-ratings-mini.gin`
  - `configs/merlin/hstu-gpt-like-sampled-softmax-mini.gin`
- Parses the final TEST eval line from each run.
- Prints a summary table with `NDCG@10`, `NDCG@50`, `HR@10`, `HR@50`, `MRR`, and `eval_loss` for each config, and marks the best one by `NDCG@10`.

Usage:

```bash
cd /path/to/generative-recommenders
source .venv/bin/activate

# Ensure Merlin CSVs exist
python3 preprocess_merlin_data.py

# Run all Merlin configs and summarize TEST metrics
python3 run_merlin_grid.py
```

This is a quick way to benchmark SASRec vs HSTU, FullSoftmax vs SampledSoftmax, and with vs without event-type inputs on your Merlin data, without manually running each config.

### Synthetic Dataset / MovieLens-3B

We support generating synthetic dataset with fractal expansion introduced in https://arxiv.org/abs/1901.08910. This allows us to expand the current 20 million real-world ratings in ML-20M to 3 billion.

To download the pre-generated synthetic dataset:

```bash
pip3 install gdown
mkdir -p tmp/ && cd tmp/
gdown https://drive.google.com/uc?id=1-jZ6k0el7e7PyFnwqMLfqUTRh_Qdumt-
unzip ml-3b.zip && rm ml-3b.zip
```

To generate the synthetic dataset on your own:

```bash
python3 run_fractal_expansion.py --input-csv-file tmp/ml-20m/ratings.csv --write-dataset True --output-prefix tmp/ml-3b/
```

### Efficiency experiments

``ops/triton`` contains triton kernels needed for efficiency experiments. ``ops/cpp`` contains efficient CUDA kernels. In particular, ``ops/cpp/hstu_attention`` contains the attention implementation based on [FlashAttention V3](https://github.com/Dao-AILab/flash-attention) with state-of-the-art efficiency on H100 GPUs.

## DLRM-v3

We have created a DLRM model using HSTU and have developed benchmarks for both training and inference to faciliate production RecSys use cases.

#### Run model training with 4 GPUs

```bash
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py --dataset debug --mode train
```

#### Run model inference with 4 GPUs

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python -m pip install .

LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug
```

## License
This codebase is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.

## Contributors
The overall project is made possible thanks to the joint work from many technical contributors (listed in alphabetical order):

Adnan Akhundov, Bugra Akyildiz, Shabab Ayub, Alex Bao, Renqin Cai, Jennifer Cao, Xuan Cao, Guoqiang Jerry Chen, Lei Chen, Li Chen, Sean Chen, Xianjie Chen, Huihui Cheng, Weiwei Chu, Ted Cui, Shiyan Deng, Nimit Desai, Fei Ding, Shilin Ding, Francois Fagan, Lu Fang, Leon Gao, Zhaojie Gong, Fangda Gu, Liang Guo, Liz Guo, Jeevan Gyawali, Yuchen Hao, Daisy Shi He, Michael Jiayuan He, Yu He, Samuel Hsia, Jie Hua, Yanzun Huang, Hongyi Jia, Rui Jian, Jian Jin, Rafay Khurram, Rahul Kindi, Changkyu Kim, Yejin Lee, Fu Li, Han Li, Hong Li, Shen Li, Rui Li, Wei Li, Zhijing Li, Lucy Liao, Xueting Liao, Emma Lin, Hao Lin, Chloe Liu, Jingzhou Liu, Xing Liu, Xingyu Liu, Kai Londenberg, Yinghai Lu, Liang Luo, Linjian Ma, Matt Ma, Yun Mao, Bert Maher, Ajit Mathews, Matthew Murphy, Satish Nadathur, Min Ni, Jongsoo Park, Colin Peppler, Jing Qian, Lijing Qin, Jing Shan, Alex Singh, Timothy Shi,  Yu Shi, Dennis van der Staay, Xiao Sun, Colin Taylor, Shin-Yeh Tsai, Rohan Varma, Omkar Vichare, Alyssa Wang, Pengchao Wang, Shengzhi Wang, Wenting Wang, Xiaolong Wang, Yueming Wang, Zhiyong Wang, Wei Wei, Bin Wen, Carole-Jean Wu, Yanhong Wu, Eric Xu, Bi Xue, Hong Yan, Zheng Yan, Chao Yang, Junjie Yang, Wen-Yun Yang, Ze Yang, Zimeng Yang, Yuanjun Yao, Chunxing Yin, Daniel Yin, Yiling You, Jiaqi Zhai, Keke Zhai, Yanli Zhao, Zhuoran Zhao, Hui Zhang, Jingjing Zhang, Lu Zhang, Lujia Zhang, Na Zhang, Rui Zhang, Xiong Zhang, Ying Zhang, Zhiyun Zhang, Charles Zheng, Erheng Zhong, Zhao Zhu, Xin Zhuang.

For the initial paper describing the Generative Recommender problem formulation and the algorithms used, including HSTU and M-FALCON, please refer to ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``([ICML'24 paper](https://dl.acm.org/doi/10.5555/3692070.3694484), [slides](https://icml.cc/media/icml-2024/Slides/32684.pdf)).
