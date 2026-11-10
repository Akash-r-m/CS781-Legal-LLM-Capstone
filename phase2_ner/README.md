# Phase 2: Legal Named-Entity Recognition (L-NER)

**Task.** Given a judgment, predict `(start, end, label)` character spans over 12 legal entity labels. Metric: **strict macro-F1** (the span must match exactly).

**Best result:** 0.4667 (Arch 1 + bottleneck adapters, word-level tags + Viterbi; Codabench ID 974273), rank 91, 14 valid submissions.

## Data

70 annotated judgments (8,393 entities) for training, 35 test documents. Documents are about 6,000 tokens and about 120 entities each. Six of the 12 labels are person/organisation names that differ only by their *role* (for example appellant's counsel vs respondent's counsel); these are the hard ones (184–407 mentions each). No gold spans overlap, so flat BIO tags lose nothing. Local evaluation used 10 held-out training documents.

## Method

### Exact character offsets
The Phase-1 tokenizer splits on whitespace and loses offsets. We re-tokenize with the regex `\S+` over the **original text** and map each UTF-8 byte back to its character, so every BPE token has an exact character range (an assertion checks that the ids equal the Phase-1 encoding). Gold offsets are end-exclusive (verified on all 8,393 spans). Predicted spans are trimmed of leading brackets/quotes and trailing punctuation. An oracle on the training data shows 94.0% of gold spans are reachable from token boundaries and 90.5% from word boundaries.

### Tagger
A linear head predicts 25 BIO tags (12 labels × B/I, plus O) on top of the Arch 1 backbone. Documents are cut into 1,023-token windows with stride 512; since Arch 1 is causal, each token takes its prediction from the window in which it has the **most left context**.

### The two changes that mattered
1. **Word-level tags.** Tagging every sub-word with argmax produced fragments (for example a case-number span "A", or "15" instead of a full date). Now every whitespace word gets one tag, read from the word's **last sub-token**, the first position at which a causal model has seen the whole word.
2. **BIO-constrained Viterbi decoding** over words (an `I-X` tag is only allowed after `B-X` or `I-X`).

### Adaptation methods compared
Full fine-tuning; LoRMA+ and LoRA (rank 8, on the query/value projections); bottleneck adapters after attention and FFN (m=32); partial fine-tuning (top 4 blocks); norm-only tuning (BitFit-style, tuning only RMSNorm gains); and knowledge distillation into a 6-layer student of the fully fine-tuned teacher. Backbone and head use separate learning rates (adapters: 4e-4 / 1e-3, up to 15 epochs, dropout 0.1). LoRMA+ is implemented as `h = ((alpha/r) B A + I) W0 x` with `B = 0` at initialisation, r = 8, alpha = 8.

### A bidirectional tagger
Arch 5 (prefix-LM) from Phase 1 was tried as a bidirectional tagger so that a name can see the role word that follows it. It failed (0.1477): its prompt positions never received any loss during Phase-1 pretraining.

## Notebooks

| Notebook | Contents |
|---|---|
| `phase2_exp1_subword_full_ft.ipynb` | Experiment 1: sub-word BIO tags, full fine-tuning, argmax decoding. Includes the oracle check and the punctuation-trim rule search. |
| `phase2_word_level_viterbi_full_lorma.ipynb` | Word-level tags + Viterbi, with a `TECHNIQUE` switch (`full` or `lorma`), error-analysis table (correct / boundary / label confusion / missed / spurious). |
| `phase2_a5_bidirectional_tagger.ipynb` | Experiment 2: Arch 5 as a bidirectional tagger, windows merged by centrality, argmax vs Viterbi. `HOLDOUT_DOCS = 0` trains on all 70 documents. |
| `phase2_all_techniques_run_all.ipynb` | Run-all notebook: trains every technique one after another (`full`, `lorma`, `lora`, `adapter`, `partial`, `normtune`, `kd`) and writes one submission zip per technique. Keep `full` before `kd` (KD reuses its teacher). |

## Submissions

| Codabench ID | Method | Notebook | Strict macro-F1 |
|---|---|---|---|
| 970227 | Sub-word tags, argmax, full FT (exp1) | `phase2_exp1_subword_full_ft` | 0.3093 |
| 971820 | Arch 5 bidirectional tagger (exp2) | `phase2_a5_bidirectional_tagger` | 0.1477 |
| 973771 | LoRMA+, words + Viterbi | `phase2_word_level_viterbi_full_lorma` | 0.4505 |
| 973772 | Full FT, words + Viterbi | `phase2_word_level_viterbi_full_lorma` | 0.4660 |
| 974268 | LoRMA+ (seed 42) | `phase2_all_techniques_run_all` | 0.4505 |
| 974269 | LoRA (seed 42) | `phase2_all_techniques_run_all` | 0.4433 |
| 974271 | KD, 6-layer student | `phase2_all_techniques_run_all` | 0.2819 |
| **974273** | **Adapters (1.4% of parameters)** | `phase2_all_techniques_run_all` | **0.4667** |
| 974276 | Partial fine-tuning | `phase2_all_techniques_run_all` | 0.4617 |
| 974277 | Norm-only tuning | `phase2_all_techniques_run_all` | 0.4132 |
| 974524 | Full FT (seed 42) | `phase2_all_techniques_run_all` | 0.4663 |
| 976576 | Full FT, "rep1pos1" variant | *notebook to be added* | 0.4635 |
| 976578 | Adapters, "rep1pos1" variant (second run) | *notebook to be added* | 0.4665 |
| 978241 | Ensemble | *notebook to be added* | 0.4635 |

## Results and analysis

| Variant | Held-out F1 | Codabench |
|---|---|---|
| Sub-word tags, argmax | 0.270 | 0.3093 |
| Word-level tags alone | 0.321 | – |
| Word-level tags + Viterbi (full FT) | 0.397 | 0.4663 |
| Adapters | 0.429 | 0.4667 |

- Changing only the **output representation** lifted the test score from 0.309 to 0.466. Viterbi on the same model added +0.076 held-out F1 and helped all seven adaptation methods (+0.04 to +0.10).
- Adapters, partial and full fine-tuning then tie at 0.46–0.47. Norm-only tuning and KD fall behind (the 6-layer student is too small for this task).
- **Local vs Codabench:** the 10 held-out documents are a noisy proxy; Codabench scores were 0.03–0.07 higher and kept the ranking only roughly.

### Error analysis
`DATE`, `COURT` and `STAT` reach 0.61–0.86 F1, but the six role labels stay at 0.00–0.39. Errors are mostly confusions (37 of 53 held-out `A.COUNSEL` and 28 of 58 `RESP` mentions get another label; `WIT` is never found). The role usually follows the name ("…, counsel for the respondent"), but a causal backbone reading the last sub-token only sees left context. This is the main limitation and the main motivation for a bidirectional encoder.

## Reproducing

1. Provide the Phase-1 checkpoint and tokenizer as a Kaggle dataset and set `MODEL_CKPT_PATH` / the tokenizer path in cell `[2]`.
2. Pick the technique (`TECHNIQUE` or `TECHNIQUES`) and run all cells. Predictions, log-probabilities and the Codabench zip are written to `/kaggle/working/`.
