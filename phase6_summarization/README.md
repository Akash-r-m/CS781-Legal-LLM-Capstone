# Phase 6: Abstractive Summarization

**Task.** Generate an abstractive summary of a court judgment. Metric: **BERTScore F1**.

**Best result:** 0.8453 (Arch 1, full fine-tuning, 3,832 judgment tokens in a 4,096-token context via RoPE interpolation; Codabench ID 1026236), rank 41, 11 valid submissions.

## Data

7,030 (judgment, summary) training pairs and 100 test judgments. 41% of judgments are longer than 3,832 tokens and 69% of reference summaries are longer than 509 tokens, so the 1,024-token context of the backbone forces a trade-off between how much judgment is read and how long the summary can be. Local evaluation: 300 training pairs held out and scored with the official BERTScore call. The header of a judgment (appeal numbers, counsel, "delivered by") is stripped.

## Method

Arch 1 stays a **decoder-only** model. Each training example is one sequence

```
<s> [judgment tokens] </s> [summary tokens] </s>
```

and the loss is computed on the **summary tokens only**. At test time the model continues after the first `</s>`.

### Token budgets (the experiment)
| Judgment tokens | Summary tokens | Context | Position handling |
|---|---|---|---|
| 512 | 509 | 1,024 | native |
| 760 | 261 | 1,024 | native |
| 1,784 | 261 | 2,048 | RoPE interpolation |
| 2,808 | 261 | 3,072 | RoPE interpolation |
| 3,832 | 261 | 4,096 | RoPE interpolation |

**RoPE position interpolation** (Chen et al., 2023) extends the context of the backbone without re-pretraining: positions are scaled by `1024 / L` (L = new context) so that they stay inside the range seen during Phase 1, and the model is fine-tuned at the longer length. Summaries are cut to 261 training tokens in the long-context runs.

### Head vs head+tail input
`DOC_MODE = "headtail"` keeps the first two thirds and last third of the judgment budget instead of only the beginning. It gave no real gain (0.8403 vs 0.8402).

### Decoding
Greedy decoding with 4-gram blocking, repetition penalty 1.1 and a minimum generation length of 150 tokens (`MIN_GEN = 150`). No post-training or alignment was used.

### Adaptation methods compared
Full fine-tuning (LR 5e-5, 3–4 epochs, dropout 0.1), LoRMA+, adapters, and a 9-layer distilled student (`kd`). Training uses fp16 autocast.

## Notebooks

| Notebook | Judgment tokens | Techniques |
|---|---|---|
| `p6_full_kd_doc512.ipynb` | 512 | full, kd |
| `p6_lorma_doc512.ipynb` | 512 | LoRMA+ |
| `p6_adapter_doc512.ipynb` | 512 | adapters |
| `p6_full_kd_doc760_mingen150.ipynb` | 760 | full, kd (min generation length 150) |
| `p6_full_doc760_headtail.ipynb` | 760 | full, head + tail mode |
| `p6_full_kd_doc1784_ctx2048_rope.ipynb` | 1,784 | full, kd, RoPE interpolation to 2,048 |
| `p6_full_doc2808_ctx3072_rope.ipynb` | 2,808 | full, RoPE interpolation to 3,072 |
| `p6_full_doc3832_ctx4096_rope_BEST.ipynb` | 3,832 | **full, RoPE interpolation to 4,096 (best)** |

## Submissions

| Codabench ID | Setting | Notebook | BERTScore F1 |
|---|---|---|---|
| 1021429 | LoRMA+, 512 | `p6_lorma_doc512` | 0.8297 |
| 1021736 | Adapters, 512 | `p6_adapter_doc512` | 0.8312 |
| 1022945 | Full FT, 512 | `p6_full_kd_doc512` | 0.8323 |
| 1022946 | KD student, 512 | `p6_full_kd_doc512` | 0.8340 |
| 1024337 | Full FT, 760 | `p6_full_kd_doc760_mingen150` | 0.8402 |
| 1024338 | KD student, 760 | `p6_full_kd_doc760_mingen150` | 0.8411 |
| 1024342 | Full FT, 760, head+tail | `p6_full_doc760_headtail` | 0.8403 |
| 1025495 | KD student, 1,784 (PI to 2,048) | `p6_full_kd_doc1784_ctx2048_rope` | 0.8438 |
| 1025497 | Full FT, 1,784 (PI to 2,048) | `p6_full_kd_doc1784_ctx2048_rope` | 0.8447 |
| 1026304 | Full FT, 2,808 (PI to 3,072) | `p6_full_doc2808_ctx3072_rope` | 0.8432 |
| **1026236** | **Full FT, 3,832 (PI to 4,096)** | `p6_full_doc3832_ctx4096_rope_BEST` | **0.8453** |

## Results and analysis

Input length (full fine-tuning):

| Judgment tokens | 512 | 760 | 1,784 | 2,808 | 3,832 |
|---|---|---|---|---|---|
| Local BERTScore | 0.834 | 0.841 | 0.847 | – | 0.844 |
| Codabench | 0.8323 | 0.8402 | 0.8447 | 0.8432 | 0.8453 |

- **Reading more of the judgment helps up to about 1,800 tokens**; beyond that the gain is within noise (0.843–0.845). Local and Codabench scores agree (0.845 vs 0.845).
- **Parameter-efficient methods trail full fine-tuning at 512 tokens** (0.830–0.831 vs 0.832), and the distilled 9-layer student is competitive (0.8438 vs 0.8447 at 1,784 tokens).
- **BERTScore is coarse:** all Phase-6 models span only 0.830–0.845, and a fluent summary with a wrong section number or wrong party still scores well.

### Error analysis: length
Summaries are cut to 261 training tokens while 94% of the references are longer, so the generated summaries (about 245 words) are systematically shorter than the references. Planned improvement: train on full-length summaries, using RoPE interpolation on the output side as well.

## Reproducing

1. Set `MODEL_CKPT_PATH` and the tokenizer path to your Kaggle dataset.
2. Set `TECHNIQUES`, the context length (`BLOCK`/`CTX`), the judgment and summary token budgets and `MIN_GEN` in cell `[2]`.
3. The notebook trains, generates summaries for the test set, scores the held-out pairs with BERTScore, and writes `prediction.json` and a Codabench zip. Runs take 25–85 minutes on one T4.
