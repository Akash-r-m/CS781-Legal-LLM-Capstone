# Phase 5: Legal Statute Identification

**Task.** Given a legal document, predict the **set of statutes** it cites, out of 100 statute ids (multi-label). Metric: **macro-F1** over the 100 labels.

**Best result:** 0.4105 (9-layer distilled student, trained on train+dev for 4 epochs, global threshold 0.55; Codabench ID 1011886), rank 31, 14 valid submissions.

## Data

42,750 train / 10,181 dev / 13,019 test documents. 100 statute ids, 3.7 per document on average; the labels are very skewed (median 679 positives per id, the most frequent id has 11,730). The median document is only 630 tokens, and a statute can be cited anywhere in it, so chunks are sampled **across the whole document** rather than from the start or end.

## Method

```
document ─► 4 chunks of 1,023 tokens spread evenly over the document
                      ▼
        Arch-1 backbone (or 9-layer student)
                      ▼
     per chunk: 100 sigmoid logits   (POOL_TOK = "mean" or "lwan")
                      ▼
  document logit = max over chunks   (POOL = "maxlogit")
                      ▼
   one global threshold (tuned on dev) + at least one statute forced
```

- **Binary relevance head.** One sigmoid per statute id; the bias is initialised to the label prior; loss is BCE with positive weights √(neg/pos), capped at 20.
- **Max over chunks.** A statute applies if *any* part of the document cites it (`POOL = "maxlogit"`; `attn`, `mean` and `lse` poolings are also implemented in the notebooks).
- **Label-wise attention (LWAN).** `POOL_TOK = "lwan"`: every statute id has its own attention over the token states of a chunk, instead of one mean-pooled vector for all labels.
- **Threshold.** One global threshold, tuned on dev over a grid (0.05–0.90). The best runs use 0.55–0.60. At least one statute is always predicted.
- **Knowledge distillation.** The student is built from the 12-layer backbone by keeping 9 layers (`STUDENT_KEEP = [0, 1, 3, 4, 6, 7, 9, 10, 11]`, 54.3M parameters) and trained on the teacher's soft probabilities **plus** the gold labels. The distilled student *beats its own teacher* (see results).
- **Adaptation methods compared:** full fine-tuning, LoRMA+, adapters, KD (LoRMA+ and adapters use the settings from the common protocol: r = 8 and m = 32).

Final settings: LR 1e-4 backbone / 1e-3 head, 16 documents per step, dropout 0.1, 4 epochs, `FINAL_ALL = True` (train+dev, no dev score) with the threshold fixed to 0.55 from the earlier dev runs.

## Notebooks

| Notebook | Contents |
|---|---|
| `p5_full_then_kd_maxlogit.ipynb` | Full fine-tuning teacher, then the KD student, mean token pooling, max over chunks |
| `p5_full_lwan_ep4.ipynb` | Full fine-tuning with label-wise attention, 4 epochs |
| `p5_lorma.ipynb` | LoRMA+ |
| `p5_adapter.ipynb` | Bottleneck adapters |
| `p5_kd_ep5.ipynb` | KD student, 5 epochs |
| `p5_kd_lwan_ep5.ipynb` | KD student + label-wise attention, 5 epochs |
| `p5_kd_trainplusdev_ep4_BEST.ipynb` | **KD student trained on train+dev, 4 epochs, threshold 0.55 (best)** |
| `p5_kd_trainplusdev_lwan_teacher_thr030.ipynb` | KD on train+dev, LWAN teacher, untuned threshold 0.30 |
| `p5_kd_trainplusdev_full_teacher_thr030.ipynb` | KD on train+dev, full-model teacher, untuned threshold 0.30 |

The KD notebooks load their teacher from a checkpoint in a Kaggle dataset (`TEACHER_CKPT`), written by the earlier teacher run, so run the teacher first.

## Submissions

| Codabench ID | Method | Notebook | Macro-F1 |
|---|---|---|---|
| 1011690, 1011693 | Full fine-tuning (submitted twice, same file) | `p5_full_then_kd_maxlogit` | 0.3531 |
| 1011692 | Adapters | `p5_adapter` | 0.3445 |
| 1011694 | KD student | `p5_full_then_kd_maxlogit` | 0.3858 |
| 1011695 | KD student, 5 epochs | `p5_kd_ep5` | 0.3935 |
| **1011886** | **KD student, train+dev, 4 epochs, thr 0.55** | `p5_kd_trainplusdev_ep4_BEST` | **0.4105** |
| 1011888 | Full FT + LWAN | `p5_full_lwan_ep4` | 0.3743 |
| 1011890 | KD student + LWAN, 5 epochs | `p5_kd_lwan_ep5` | 0.4044 |
| 1013776 | KD, train+dev, full teacher, thr 0.30 | `p5_kd_trainplusdev_full_teacher_thr030` | 0.3515 |
| 1014531 | KD, train+dev, LWAN teacher, thr 0.30 | `p5_kd_trainplusdev_lwan_teacher_thr030` | 0.3589 |
| 1016729 | LoRMA+ | `p5_lorma` | 0.3379 |
| 1011891 | Ensemble (failed submission) | *notebook to be added* | – |
| 1016705 | Ensemble 1 | *notebook to be added* | 0.3764 |
| 1016745 | Ensemble 2 (failed submission) | *notebook to be added* | – |
| 1016753 | Ensemble 3 | *notebook to be added* | 0.4077 |

## Results and analysis

| Variant | Dev | Codabench |
|---|---|---|
| Full fine-tuning, mean pooling | 0.357 | 0.3531 |
| LoRMA+ / adapters | 0.34–0.35 | 0.34 |
| **KD student (9 layers)** | 0.386 | 0.3858 |
| KD + LWAN | 0.406 | 0.4044 |
| KD, train+dev, 4 epochs | – | **0.4105** |

- **Distillation beats its own teacher** (dev 0.386 vs 0.357, test 0.386 vs 0.353): soft teacher targets regularise 100 noisy binary labels better than hard labels alone.
- **LWAN improves both** full fine-tuning (0.374) and KD (0.404).
- Local dev and Codabench scores agree (0.386 vs 0.386).

### Error analysis: the threshold
Two train+dev runs used an untuned threshold of 0.30. They predicted about 9 statutes per document against 3.7 in the gold data and dropped to 0.35–0.36 (IDs 1013776 and 1014531). With one global threshold, frequent statutes are under-predicted and rare ones over-predicted. Planned improvement: per-label thresholds tuned by cross-validation.

## Reproducing

1. Set `MODEL_CKPT_PATH` and the tokenizer path to your Kaggle dataset.
2. Set `TECHNIQUES` (`full`, `lorma`, `adapter`, `kd`, with `full` before `kd`), `POOL`, `POOL_TOK`, `FINAL_ALL` and `THR_FINAL` in cell `[2]`.
3. The notebook writes per-document probabilities, `prediction.json` and a Codabench zip to `/kaggle/working/`. Runs take 160–310 minutes on one T4.
