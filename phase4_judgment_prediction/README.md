# Phase 4: Court Judgment Prediction (CJPE)

**Task.** Given the text of a court case, predict the binary outcome (0/1). Metric: **macro-F1**.

**Best result:** 0.7495 (full fine-tuning, last 4 chunks, 3 epochs; Codabench ID 1005232), rank 48, 14 valid submissions.

## Data

37,387 train / 3,505 dev / 1,517 test cases. 5,082 training and 994 dev documents are exact duplicates; we removed them (13.6% of train, 28% of dev) so they would not inflate the dev score. The training set is 59% class 0 while dev and test are balanced, so the loss uses mild class weights (∝ 1/√frequency). Median length is 2,307 tokens. Indian judgments state the decision near the end, which motivated reading the **tail** of the document first.

Documents longer than 16,000 words keep the first 4,000 words and the end (`HEAD_WORDS = 4000`), and are then cut into 1,023-token chunks (+`<s>`).

## Method

```
judgment ─► BPE ─► chunks of 1,023 tokens
                       │  (which chunks? DOC_MODE)
                       ▼
          Arch-1 backbone (Phase-1 weights)
                       ▼
     chunk vector = mean of its token states
                       ▼
   attention pooling over chunks (zero-initialised → starts as a plain mean)
                       ▼
                 linear layer → 2 logits
```

Class-weighted cross-entropy; the epoch with the best dev macro-F1 is kept.

### Which part of the judgment to read (`DOC_MODE`)
| Mode | Chunks used |
|---|---|
| `tail` | last chunk only |
| `head` | first chunk only |
| `headtail` | first + last chunk |
| `tail4` (with `MAX_CHUNKS = k`) | the last *k* chunks (k = 2, 3, 4) |
| `spread` | `MAX_CHUNKS` chunks evenly spaced over the document |

Other options: `POOL_TOK = "meanlast"` concatenates the chunk's last-token state to the mean ("meanlast"); adaptation techniques (`full`, `lorma`, `lora`, `adapter`, `partial`, `normtune`, `kd`); and a probability-averaging ensemble. Predictions of Phases 2 and 3 are **not** used as features.

### Hyper-parameters of the best run
Full fine-tuning, backbone LR 3e-5 and head LR 5e-4, 3 epochs, 16 documents per step, dropout 0.1, last 4 chunks (`MAX_CHUNKS = 4`), argmax decision, seed 44.

## Notebooks

All notebooks share the same code (cell `[2] Config` differs). `phase4_cjpe_arch1_zoo.py` is the script version of that code.

| Notebook | Settings |
|---|---|
| `p4_full_tail1_ep2.ipynb` | full, tail (1 chunk), 2 epochs |
| `p4_full_tail1_ep4.ipynb` | full, tail, 4 epochs, seed 43 |
| `p4_full_headtail_ep2.ipynb` | full, head + tail, 2 epochs |
| `p4_full_last2chunks.ipynb` | full, last 2 chunks |
| `p4_full_last3chunks_ep3.ipynb` | full, last 3 chunks, 3 epochs |
| `p4_full_last4chunks_ep3_BEST.ipynb` | **full, last 4 chunks, 3 epochs (best)** |
| `p4_full_last2chunks_meanlast.ipynb` | last 2 chunks + mean/last-token pooling |
| `p4_full_last4chunks_meanlast.ipynb` | last 4 chunks + mean/last-token pooling |
| `p4_lorma_tail.ipynb` | LoRMA+, tail |
| `p4_adapter_tail.ipynb` | bottleneck adapters, tail |
| `p4_kd_tail.ipynb` | 9-layer distilled student, tail |
| `p4_ensemble_all_members.ipynb` | ensemble of saved members, `ENSEMBLE_USE = None` (all) |
| `p4_ensemble_full_only.ipynb` | ensemble variant with `ENSEMBLE_USE = ["full"]` |

The ensemble notebooks read the `doc_probs.pkl` files written by the single-model runs (`ENSEMBLE_MEMBERS`), so run those first.

## Submissions

| Codabench ID | Setting | Notebook | Macro-F1 |
|---|---|---|---|
| 994635 | Full FT, tail (1 chunk) | `p4_full_tail1_ep2` | 0.7347 |
| 994637 | Full FT, last 2 chunks | `p4_full_last2chunks` | 0.7422 |
| 994640 | Ensemble (re-run 3) | `p4_ensemble_full_only` | 0.7240 |
| 994641 | Ensemble 2 | `p4_ensemble_all_members` | 0.7194 |
| 994643 | Ensemble (re-run) | `p4_ensemble_all_members` | 0.7155 |
| 994686 | KD student, tail | `p4_kd_tail` | 0.7116 |
| 994687 | LoRMA+, tail | `p4_lorma_tail` | 0.6812 |
| 994691 | Full FT, tail, 4 epochs | `p4_full_tail1_ep4` | 0.7240 |
| 1002067 | Adapters, tail | `p4_adapter_tail` | 0.7022 |
| 1002076 | Full FT, head + tail | `p4_full_headtail_ep2` | 0.7171 |
| **1005232** | **Full FT, last 4 chunks, 3 epochs** | `p4_full_last4chunks_ep3_BEST` | **0.7495** |
| 1005234 | Full FT, last 3 chunks, 3 epochs | `p4_full_last3chunks_ep3` | 0.7481 |
| 1007902 | Last 2 chunks + meanlast | `p4_full_last2chunks_meanlast` | 0.6833 |
| 1008587 | Last 4 chunks + meanlast | `p4_full_last4chunks_meanlast` | 0.6997 |

## Results and analysis

Which part of the judgment is read (full fine-tuning, test macro-F1):

| Chunks read | Last 1 | Last 2 | Last 3 | Last 4 | First + last |
|---|---|---|---|---|---|
| Macro-F1 | 0.735 | 0.742 | 0.748 | **0.750** | 0.717 |

- **Reading more of the end helps steadily**, because the operative order sits there. Adding the first chunk (head + tail) *hurts* (0.717).
- **Adding the last-token state to the mean (meanlast) hurt** (0.683 / 0.700).
- Parameter-efficient methods (LoRMA+, adapters, KD) trail full fine-tuning by 0.03–0.07, and a 5-run probability ensemble (0.724) did not beat the best single model.
- Local dev and Codabench agree closely (0.748 vs 0.750).

### Error analysis
The best tail model predicts class 0 too often (class-1 recall 0.64 vs 0.83 for class 0 on dev). When the operative order sits in the middle of a long judgment, or is phrased as a partial remand, the last chunk does not contain it.

## Reproducing

1. Set `MODEL_CKPT_PATH` and the tokenizer path to your Kaggle dataset.
2. Set `DOC_MODES`, `MAX_CHUNKS`, `TECHNIQUES`, `SEED` in cell `[2]`; set `FINAL_ALL = True` to train on train+dev for a fixed number of epochs.
3. Runs take 50–210 minutes on one T4 (16 documents per optimiser step, checkpoints saved for resuming).
