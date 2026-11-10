# CS781 Capstone: In-House Legal Intelligence System

**Akash (230091) · IIT Kanpur · Codabench user `a_230091`**

A self-hosted legal NLP pipeline built only from the course data. A law firm cannot send privileged documents to a third-party LLM API, so we pre-trained a small legal language model **from scratch** (own 32k byte-level BPE tokenizer, no pretrained weights) and then adapted that one backbone to six downstream tasks from the [IL-TUR benchmark](https://arxiv.org/abs/2407.05399).

All data comes from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Everything was trained on free Kaggle GPUs (one T4 per run).

The full write-up (design choices, ablations, error analysis, limitations) is in the project report. This repository contains the code for **every valid Codabench submission** and a README per phase describing the architecture, method and implementation.

---

## Results

Best valid score per phase (ranks recorded on 9 November 2026).

| Phase | Task | Primary metric | Best score | Rank | Valid submissions | Best-run folder |
|---|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | **0.8300** | 82 | 6 | [`phase1_pretraining`](phase1_pretraining) |
| 2 | Legal NER | strict macro-F1 | **0.4667** | 91 | 14 | [`phase2_ner`](phase2_ner) |
| 3 | Rhetorical role prediction | macro-F1 | **0.5566** | 24 | 11 | [`phase3_rhetorical_roles`](phase3_rhetorical_roles) |
| 4 | Court judgment prediction | macro-F1 | **0.7495** | 48 | 14 | [`phase4_judgment_prediction`](phase4_judgment_prediction) |
| 5 | Legal statute identification | macro-F1 | **0.4105** | 31 | 14 | [`phase5_statute_id`](phase5_statute_id) |
| 6 | Abstractive summarization | BERTScore F1 | **0.8453** | 41 | 11 | [`phase6_summarization`](phase6_summarization) |
| 7 | Prior case retrieval | micro-F1@7 | **0.4637** | 22 | 16 | [`phase7_retrieval`](phase7_retrieval) |

Each phase README lists every submission with its Codabench ID, score and the notebook that produced it.

## Key findings

- **Phase 1:** a 66M-parameter LLaMA-style decoder (RoPE, RMSNorm, SwiGLU) was the best of five architectures and became the backbone for Phases 2–7. An attention-free gated CNN came within 0.004 BERTScore.
- **What moved the scores was input/output representation**, not model size: whole-word tags with BIO-constrained Viterbi decoding (NER), whole-document context plus a role-transition prior (rhetorical roles), reading the *end* of the judgment (judgment prediction), max-over-chunks with distillation (statutes), and longer input via RoPE interpolation (summarization).
- **Parameter-efficient methods** (adapters, LoRMA+) usually came within 0.01–0.04 of full fine-tuning.
- **Retrieval:** a lexical learning-to-rank model (BM25/TF-IDF + the un-tuned Phase-1 model as a feature source, LightGBM LambdaRank) clearly beat every fine-tuned neural retriever.

## Architecture overview

```
              ┌───────────────────────────────────────────────┐
 raw text ──► │ own byte-level BPE tokenizer (32k, from scratch)│
              └───────────────────────┬───────────────────────┘
                                      ▼
              ┌───────────────────────────────────────────────┐
              │ Phase 1: Arch 1 decoder, 66.2M params          │
              │ 12 layers, d=576, 12 heads, RoPE, RMSNorm,     │
              │ SwiGLU, context 1024                           │
              └───────────────────────┬───────────────────────┘
        ┌────────┬────────┬───────────┼───────────┬────────┬────────┐
        ▼        ▼        ▼           ▼           ▼        ▼        ▼
     P2 NER   P3 roles  P4 CJPE     P5 statutes  P6 summ.  P7 PCR   (task heads)
   BIO/word  per-</s>   chunk mean  100 sigmoids  continue  chunk embeddings
   +Viterbi  +PMI-Vit.  +attn pool  max-over-     <s>doc</s> + BM25/TF-IDF
                                    chunks        summary    → LightGBM ranker
```

Phases 2–7 load the Phase-1 weights, drop the LM head, add a small task head, and compare the same family of adaptation methods: full fine-tuning, LoRMA+, LoRA, bottleneck adapters, partial fine-tuning (top 4 blocks), norm-only tuning, and knowledge distillation into a 9-layer (6-layer in Phase 2) student made by deleting layers.

## Repository layout

```
.
├── README.md                      this file
├── requirements.txt
├── phase1_pretraining/            5 architectures, notebooks + README
├── phase2_ner/
├── phase3_rhetorical_roles/
├── phase4_judgment_prediction/    (+ phase4_cjpe_arch1_zoo.py, script version of the notebook code)
├── phase5_statute_id/
├── phase6_summarization/
└── phase7_retrieval/
```

Every phase folder has a `README.md` and a `notebooks/` folder. The notebooks are Kaggle notebooks written as sequences of `# %% [n]` cells.

## How to reproduce

1. **Environment.** Kaggle notebook, GPU T4, internet on. Libraries are the Kaggle defaults plus `ijson` (installed inside the Phase 1 notebooks) and `lightgbm` (Phase 7).
2. **Secrets.** Add your Hugging Face token as a Kaggle secret. Almost all notebooks read it with `UserSecretsClient().get_secret("HF")`; the Phase 1 decoder notebook uses the name `HF_TOKEN` instead (check the `get_secret(...)` call at the top of each notebook). No tokens are stored in this repository.
3. **Run Phase 1 first** (`phase1_pretraining/notebooks/phase1_decoder.ipynb`). It trains the tokenizer and the backbone and writes `legal_bpe.json` and `model.pt`.
4. **Upload the Phase-1 outputs as a Kaggle dataset** and point `MODEL_CKPT_PATH` (and, where present, the tokenizer path) at it. The notebooks in Phases 2–7 currently contain the paths of the author's own Kaggle datasets, for example `/kaggle/input/datasets/akashmalgav2/phase2-model1/model_best.pt`. Change these to your own dataset.
5. **Run any later phase notebook.** Settings are at the top of each notebook (cell `[2] Config`): `TECHNIQUES`, seeds, `FINAL_ALL` (train on train+dev), chunking options, and so on. A notebook writes `prediction.json` and a ready-to-upload `submission_*.zip`.
6. **Ensembling notebooks** (Phases 3, 4, 5) read the probability files (`*.pkl`) saved by the earlier single-model runs, so run those first.

Model checkpoints and data are **not** committed (see `.gitignore`).

## Known issues and honest notes

- **Phase 1, Architecture 2 has a future-token leak.** Its local-attention mask is built as `rel = j - i` with `0 <= rel < 256`, so each position can attend to the *next* 255 tokens, including the token it must predict. Validation perplexity (1.09) was therefore meaningless and the submission scored worst (0.7581). The correct mask is `0 <= i - j < 256`. The notebook is kept unchanged because it is the submission that was scored. Details in [`phase1_pretraining/README.md`](phase1_pretraining/README.md).
- **Phase 1 validation loss is not a reliable comparison across architectures**: the validation loader is unshuffled and scores only 50 batches, so it covers mostly Acts and High Court text. Architectures were compared on Codabench instead.
- **Phase 5:** two train+dev runs used an untuned threshold of 0.30 and scored 0.35–0.36 (about 9 statutes per document predicted against 3.7 in the gold data).
- **Phase 7:** the three max-sim "fine-tuned" submissions have identical scores (0.2906) because for max-sim scoring every method's best epoch was epoch 0, which is the un-tuned Phase-1 model.
- Local scores and Codabench scores agree closely in Phases 4, 5 and 6; they diverge in Phase 2 (10 held-out documents) and Phase 3 (the dev set has no `Dissent` or `PrecedentOverruled` sentences).

## Tokenizer

The tokenizer implements byte-level BPE from scratch (the Assignment-1 algorithm): whitespace pre-split, UTF-8 bytes as the base alphabet plus an end-of-word marker, deterministic lexicographic tie-breaking among equally frequent pairs, reserved tokens `<pad>`, `<unk>`, `<s>`, `</s>` in that order, 32,000 tokens in total. The code is in the first cells of `phase1_pretraining/notebooks/phase1_decoder.ipynb`. In Phases 2–7 the same merges are reused, with a regex re-tokenization over the original text to recover exact character offsets (needed for strict NER matching).

## Report

The project report (PDF) describes all experiments, ablations and the error analysis. See also the LLM usage declaration at the end of the report.
