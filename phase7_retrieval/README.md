# Phase 7: Prior Case Retrieval (PCR)

**Task.** For each query case, return the **7 most relevant prior cases** from its candidate pool. Metric: **micro-F1@7**.

**Best result:** 0.4637 (learning-to-rank v2, LightGBM LambdaRank, bagged, refit on train+dev; Codabench ID 1042516), rank 22, 16 valid submissions.

## Data

827 / 118 / 237 train / dev / test queries with candidate pools of 4,320 / 1,023 / 1,727 cases (9–15 candidates per query in the per-query pools used for ranking). A query has 6.4 relevant cases on average (median 4); cases are long (median about 4,500 words). **Each query also appears in its own pool and must never be returned**, which the code enforces everywhere. Because the pools are small, ranking *within* the pool matters more than first-stage recall. Query and candidates share one encoder and one chunking policy; cases longer than 12,000 words keep the first and last 6,000 words.

## Approach 1: neural bi-encoder on the Phase-1 backbone

- **Representation.** A case is encoded as up to 4–8 chunks (1,023 tokens each) spread over the case; the Arch-1 backbone encodes each chunk and the token states are mean-pooled.
- **Scoring.** `SCORE = "mean"`: cosine of the chunk-averaged vectors. `SCORE = "maxsim_sym"`: symmetric chunk-level "max-sim" (late interaction in the spirit of ColBERT).
- **Training.** Multi-positive **InfoNCE** loss (`TAU = 0.05`, 3 positives, 4 BM25-mined hard negatives drawn from the top-40 non-relevant BM25 candidates, 1 random negative per training query) or a triplet loss with margin. Training uses 2–4 chunks per case, inference 4–8 chunks.
- **Adaptation methods:** full fine-tuning, LoRMA+, LoRA, adapters, partial fine-tuning, and a 9-layer distilled student.

## Approach 2 (final system): learning-to-rank

A gradient-boosted ranker over the top-300 candidates of a fused score. For each (query, candidate) pair it combines features from several channels:

- BM25 (and BM25 with numbers), three TF-IDF variants (including a unigram+bigram TF-IDF), and six similarity channels from the **un-tuned Phase-1 model** at two chunk sizes. For each channel: the raw score, row and column z-scores and the rank;
- document lengths, and the similarity of the candidate to the query's other top candidates.

In total 54 features per pair.

| Version | Model | Notebook |
|---|---|---|
| LTR v1 | `HistGradientBoostingClassifier` (depth 5, lr 0.05, 50–300 rounds picked on dev) on the fused features; refit on train+dev | `p7_ltr_v1_histgb.ipynb` |
| **LTR v2** | **LightGBM `LGBMRanker` with the LambdaRank objective** (31 leaves, lr 0.05, 200 rounds), bagged over 3 seeds, refit on train+dev | `p7_ltr_v2_lightgbm_BEST.ipynb` |

The ranker's dev score is slightly optimistic because the 118 dev queries were also used to choose the ranker (dev 0.465, test 0.461).

## Notebooks

| Notebook | Settings |
|---|---|
| `p7_adapter_train2_infer4.ipynb` | adapters, 2 train / 4 inference chunks, InfoNCE, mean-cosine |
| `p7_adapter_train2_rand_seed52.ipynb` | adapters, random chunk selection, seed 52 |
| `p7_adapter_train4_infer8_maxsim_seed62.ipynb` | adapters, 4 / 8 chunks, max-sim scoring |
| `p7_full_kd_train2_infer4.ipynb` | full fine-tuning and KD student, 2 / 4 chunks |
| `p7_full_triplet_seed53.ipynb` | full fine-tuning, triplet loss |
| `p7_lorma_train2_infer4.ipynb` | LoRMA+, 2 / 4 chunks |
| `p7_lorma_train3_infer4_seed51.ipynb` | LoRMA+, 3 / 4 chunks (more hard negatives), seed 51 |
| `p7_lorma_train4_infer8_maxsim_seed63.ipynb` | LoRMA+, 4 / 8 chunks, max-sim scoring |
| `p7_partial_train2_infer4_seed54.ipynb` | partial fine-tuning, 2 / 4 chunks |
| `p7_partial_train4_infer8_maxsim_seed61.ipynb` | partial fine-tuning, 4 / 8 chunks, max-sim scoring |
| `p7_ltr_v1_histgb.ipynb` | LTR v1 (see above) |
| `p7_ltr_v2_lightgbm_BEST.ipynb` | LTR v2 (final system) |

All notebooks share the same bi-encoder code (cells `[1]`–`[7]`). The LTR notebooks add the lexical channels and the ranker after those cells, and they read similarity matrices saved by earlier runs, so run the bi-encoder notebooks first.

## Submissions

| Codabench ID | Method | Notebook | micro-F1@7 |
|---|---|---|---|
| 1037105 | Adapters, mean-cosine | `p7_adapter_train2_infer4` | 0.2735 |
| 1037109 | Full fine-tuning | `p7_full_kd_train2_infer4` | 0.2847 |
| 1037113 | KD student | `p7_full_kd_train2_infer4` | 0.2723 |
| 1037114 | LoRMA+ | `p7_lorma_train2_infer4` | 0.2788 |
| 1037180 | Score fusion (neural + BM25) | fusion cell of the bi-encoder / LTR notebooks | 0.2995 |
| 1039225 | LoRMA+, 3 train chunks, 6 hard negatives | `p7_lorma_train3_infer4_seed51` | 0.2782 |
| 1039226 | Partial fine-tuning | `p7_partial_train2_infer4_seed54` | 0.2853 |
| 1039229 | Adapters, random chunks | `p7_adapter_train2_rand_seed52` | 0.2758 |
| 1039230 | Full fine-tuning, triplet loss | `p7_full_triplet_seed53` | 0.2770 |
| 1040219 | Adapters, max-sim | `p7_adapter_train4_infer8_maxsim_seed62` | 0.2906 |
| 1040220 | LoRMA+, max-sim | `p7_lorma_train4_infer8_maxsim_seed63` | 0.2906 |
| 1040357 | Partial, max-sim | `p7_partial_train4_infer8_maxsim_seed61` | 0.2906 |
| 1041881 | LTR v1 | `p7_ltr_v1_histgb` | 0.4371 |
| 1041882 | LTR v1, refit on train+dev | `p7_ltr_v1_histgb` | 0.4418 |
| 1042515 | LTR v2 (bagged) | `p7_ltr_v2_lightgbm_BEST` | 0.4613 |
| **1042516** | **LTR v2, refit on train+dev** | `p7_ltr_v2_lightgbm_BEST` | **0.4637** |

## Results and analysis

| System | Dev | Test |
|---|---|---|
| BM25 alone | 0.185 | – |
| Bi-encoder, full FT, mean pooling | 0.296 | 0.2847 |
| Phase-1 zero-shot, max-sim | 0.320 | 0.2906 |
| LTR, lexical features only | 0.453 | – |
| LTR, Phase-1 features only | 0.421 | – |
| **LTR, all features, bagged, train+dev** | 0.465 | **0.4637** |

- **The task is mostly lexical.** Shared statutes, citations and facts are captured better by TF-IDF/BM25 than by the mean-pooled states of a 66M-parameter model. The Phase-1 features add a small complementary signal (lexical only 0.453 → all features 0.459).
- **Fine-tuning hurt max-sim.** Contrastive fine-tuning on 817 queries raised the mean-pooling bi-encoder from 0.280 to 0.296 dev, but for max-sim scoring every method's best epoch was epoch 0. The three "fine-tuned" max-sim submissions (1040219, 1040220, 1040357) are therefore the un-tuned Phase-1 model and score identically (0.2906).
- **The metric rewards exactly 7 answers**, even when a query has 1 or 52 relevant cases.

### Planned improvements
Use the LTR ranker's top candidates as hard negatives to retrain the neural retriever, and pretrain a bidirectional encoder for retrieval.

## Reproducing

1. Set `MODEL_CKPT_PATH` and the tokenizer path to your Kaggle dataset; `pip install lightgbm` if it is not installed.
2. Run a bi-encoder notebook with the settings of your choice (`TECHNIQUES`, `SCORE`, `LOSS`, `TRAIN_CHUNKS`, `INFER_CHUNKS`, `SEED`); it saves similarity matrices and a Codabench zip.
3. Run `p7_ltr_v2_lightgbm_BEST.ipynb` last: it builds the lexical channels, fits the ranker on train (dev for model selection), refits on train+dev, and writes the final submission zips.
