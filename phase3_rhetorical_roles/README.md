# Phase 3: Rhetorical Role Prediction

**Task.** Assign each sentence of a judgment one of 13 rhetorical roles (Fact, Argument-Petitioner, Argument-Respondent, Ratio of the Decision, Precedent Relied Upon, …). Two domains, **CL** (Competition Law) and **IT** (Income Tax), are trained jointly. Metric: **macro-F1** over the 13 labels.

**Best result:** 0.5566 (ensemble with a role-transition Viterbi decoder, members trained on train+dev; Codabench ID 987880), rank 24, 11 valid submissions.

## Data

80 training / 10 dev / 10 test documents (CL + IT); documents come as pre-split sentence lists (16,845 training sentences, median 120 sentences per document). The labels are very imbalanced (`Fact` 4,098 sentences vs `PrecedentOverruled` 3), and two roles (`Dissent`, `PrecedentOverruled`) have **no sentences in the dev set**, so they score 0 on dev whatever we do; the loss is class-weighted for this reason (weights ∝ 1/√frequency). The provided sentence lists are used unchanged; each sentence is cut to 128 tokens and closed by `</s>`.

## Method

### One stream per document
Sentences are packed into a single token stream

```
<s> s1 </s> s2 </s> s3 </s> ...
```

and the backbone's hidden state at each sentence's closing `</s>` is classified by a linear head. Because Arch 1 is causal, **every sentence is conditioned on all earlier sentences**.

- **Windows.** Long documents are cut into windows of 640 target tokens (labelled) preceded by 320 tokens of earlier, unlabelled context (`CTX_TOK = 320`). During training a second, 400-token layout is used as augmentation.
- **Layout features.** Zero-initialised embeddings of the sentence's relative position, its index and the domain (CL/IT) are added to the sentence representation, so training starts from a neutral model.

### Ensemble and role-transition prior
- Class probabilities of several runs (full fine-tuning, adapters, LoRMA+, KD, and a sentence-only run) are averaged with weights chosen on dev.
- Each document is then decoded with a **Viterbi pass** whose transition scores are the log pointwise mutual information (PMI) of consecutive roles counted on the training data, scaled by λ = 0.25 (chosen on dev).
- Final members are refit on train+dev with the epoch count that was best on dev.

### Adaptation methods
Full fine-tuning, LoRMA+, LoRA, bottleneck adapters, partial fine-tuning, norm-only tuning and KD (9-layer student) share one skeleton, selected with `TECHNIQUES`.

## Notebooks

| Notebook | Contents |
|---|---|
| `phase3_base_techniques.ipynb` | The first full pipeline: whole-document context model with a `TECHNIQUES` switch (`full`, `adapter`, `lorma`, …), writes probabilities and one zip per technique. |
| `phase3_modes_kd_finalall_ensemble.ipynb` | Same code with a `MODE` switch: `kd` (distilled student), `final_all` (full/adapter/LoRMA+ trained on train+dev), `ensemble_only`. |
| `phase3_modes_ctx_ablation.ipynb` | Adds `MODE = "ctx_ablation"`: the **sentence-only baseline**. Every sentence is classified alone (`CTX_TOK = 0`, no layout features), for the "does context help?" analysis. |
| `phase3_ensemble_pmi_viterbi_final.ipynb` | Loads the saved probability files, searches ensemble weights on dev (prints the top 8, because dev is noisy), picks λ for the PMI-Viterbi decoder and writes the final ensemble submission. `FINAL_ALL = True` produces members trained on train+dev. |

**Run order:** base / modes notebooks first (they save `*.pkl` probability files), then the ensemble notebook.

## Submissions

| Codabench ID | Method | Notebook | Macro-F1 |
|---|---|---|---|
| 985044 | Adapters (seed 42) | `phase3_base_techniques` | 0.4788 |
| 985045 | Full fine-tuning, with context | `phase3_base_techniques` | 0.5132 |
| 985046 | LoRMA+ | `phase3_base_techniques` | 0.4792 |
| 985722 | Ensemble + PMI-Viterbi | `phase3_ensemble_pmi_viterbi_final` | 0.5412 |
| 986760 | KD student | `phase3_modes_kd_finalall_ensemble` (`kd`) | 0.4573 |
| 987158 | KD with best members, ensemble | `phase3_modes_ctx_ablation` | 0.4967 |
| 987310 | LoRMA+, train+dev | `phase3_modes_kd_finalall_ensemble` (`final_all`) | 0.4847 |
| 987311 | Full FT, train+dev | `phase3_modes_kd_finalall_ensemble` (`final_all`) | 0.5134 |
| 987313 | Adapters, train+dev | `phase3_modes_kd_finalall_ensemble` (`final_all`) | 0.5268 |
| 987829 | Full FT, **sentence only** (no context) | `phase3_modes_ctx_ablation` (`ctx_ablation`) | 0.3781 |
| **987880** | **Ensemble + PMI-Viterbi, members trained on train+dev** | `phase3_ensemble_pmi_viterbi_final` | **0.5566** |

## Results and analysis

- **Document context is the biggest factor.** The same fully fine-tuned model scores 0.378 on test classifying each sentence alone and 0.513 with context (+0.135).
- **Ensemble + role-transition prior add another 0.03–0.04** (0.5132 single model → 0.5412 ensemble with Viterbi → 0.5566 with train+dev members).
- Local dev (0.43–0.44) and Codabench (0.48–0.53) differ because the 10 dev documents contain no `Dissent` or `PrecedentOverruled` sentences, so 2 of 13 labels always score 0 on dev.

### Error analysis
The largest confusion is `ArgumentPetitioner → ArgumentRespondent` (135 of 400 dev sentences for full fine-tuning), then `Fact` and `PrecedentReliedUpon → RatioOfTheDecision` (60 and 64). The two argument roles use the same language and differ only in who is speaking, which is often stated a paragraph earlier. CL is harder than IT (dev 0.34 vs 0.46). A causal model sees only earlier sentences; a bidirectional encoder would help here.

## Reproducing

1. Set `MODEL_CKPT_PATH` (Phase-1 checkpoint) and the tokenizer path to your Kaggle dataset.
2. Choose `TECHNIQUES` / `MODE`, run all cells; probabilities (`*.pkl`) and zips are written to `/kaggle/working/`.
3. Put the `*.pkl` files of the single-model runs in a Kaggle dataset and edit the paths in the ensemble notebook (`ALL_PKLS`, cell `[X1]`).
