# Phase 1: Legal Language-Model Pretraining

**Task.** Train a 50M–300M-parameter language model from scratch (no pretrained weights). The test input is a 512-word prompt; the model must generate a 128-word continuation, scored by **BERTScore F1**.

**Best result:** 0.8300 (Architecture 1, retrained with best-checkpoint selection; Codabench ID 961143), rank 82, 6 valid submissions.

## Data and preprocessing

| Source | Notes |
|---|---|
| Acts, High Courts, Supreme Court, Tribunals | JSON files, up to 10.7 GB each (High Courts) and 6.5 GB (Tribunals) |

- **Streaming parse.** The large files are read record by record with `ijson`; a plain `json.load` runs out of memory.
- **Equal sampling per source.** We take **15M training tokens from each of the four sources (60M total)**. Sampling by size would almost remove statutory language (Acts are about 4% of bytes, High Courts 56%); Acts are therefore up-sampled about 6.6×.
- **Validation split.** 2% of the documents of each source (`VAL_FRACTION = 0.02`).
- **Only about 35% of documents** are long enough for a 512-token prompt plus a continuation. This matters for the encoder–decoder and prefix-LM models (see below).

### Tokenizer

A byte-level BPE written from scratch (the Assignment-1 algorithm): whitespace pre-split, UTF-8 bytes as the base alphabet, an end-of-word marker (`\u0100`), lexicographic tie-breaking among equally frequent pairs, reserved tokens `<pad>`, `<unk>`, `<s>`, `</s>`, vocabulary 32,000. Byte fallback means symbols such as `§` are never unknown; newlines are lost because of the whitespace pre-split. Merges are learned with incremental pair-count updates so that training stays tractable on tens of millions of tokens. The tokenizer is saved as `legal_bpe.json` and reused in every later phase.

## The five architectures

All five share the tokenizer, the data, AdamW with warm-up and cosine decay, fp16 autocast, an effective batch of 32 sequences, and one Kaggle T4. Each notebook asserts a parameter count (tied embeddings counted once) between 50M and 300M. Each architecture is a known family plus one modification of our own.

| # | Family | Params | Notebook | Idea |
|---|---|---|---|---|
| 1 | Decoder-only, LLaMA-style | 66.2M | `phase1_decoder.ipynb` | 12 layers, d=576, 12 heads, RoPE, RMSNorm, SwiGLU (FFN ×8/3), causal attention, context 1,024 |
| 2 | Local attention + persistent memory | 66.8M | `phase1_local_memory.ipynb` | 256-token window and 32 memory slots, fused by a learned per-head sigmoid gate; LayerNorm, GELU, learned relative-position bias |
| 3 | Encoder–decoder | 74.6M | `phase1_encoder_decoder.ipynb` | 6+6 layers, 512-token prompt → 160 tokens, **gated cross-attention** (learned sigmoid gate per decoder token) |
| 4 | Gated dilated CNN | 106.6M | `phase1_gated_cnn.ipynb` | 16 blocks, d=640, no attention; dilated causal convolutions plus a causal running-mean "global context" fused with the GLU output by a learned gate |
| 5 | Prefix-LM hybrid | 166.8M | `phase1_prefix_lm.ipynb` | Bidirectional over the 512-token prompt, causal over the continuation; two gated FFN experts |

### Training settings (from the notebooks)

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 | Arch 5 |
|---|---|---|---|---|---|
| Steps | 6,000 | 1,000 | 6,000 | 6,000 | 4,000 (was 15,000) |
| Peak LR | 3e-4 → 3e-5 | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| Batch × accumulation | 8 × 4 | 4 × (accum) | 16 × (accum) | 8 × 4 | 8 × (accum) |
| Sampling | T=0.8, top-p 0.95 | T=0.8, top-p 0.95 | T=0.7, top-p 0.95, trigram block | T=0.7, top-p 0.95, trigram block | T=0.7, top-p 0.95, trigram block |
| Train time on T4 | 196 min | 176 min | 54 min | 259 min | 123 min |

Prompt handling for Arch 1, 2 and 4: the first `BLOCK_SIZE - MAX_GEN_TOKENS = 640` tokens of the prompt are kept; the generation is cut to 128 words.

### Generation

Nucleus sampling at the temperature given above. The continuation is decoded back to text and cut to 128 whitespace-separated words, then written to `prediction.json` and zipped for Codabench.

## Submissions

| Codabench ID | Architecture | Notebook | BERTScore F1 |
|---|---|---|---|
| 954458 | Arch 1, first run (final-step checkpoint) | `phase1_decoder.ipynb` | 0.8297 |
| 961143 | **Arch 1, retrained with best-checkpoint selection (step 5,700, val loss 1.998)** | *notebook to be added* | **0.8300** |
| 956491 | Arch 3, encoder–decoder | `phase1_encoder_decoder.ipynb` | 0.8160 |
| 956656 | Arch 2, local attention + memory | `phase1_local_memory.ipynb` | 0.7581 |
| 958201 | Arch 5, prefix-LM | `phase1_prefix_lm.ipynb` | 0.8191 |
| 958516 | Arch 4, gated CNN | `phase1_gated_cnn.ipynb` | 0.8262 |

The retrained Arch 1 checkpoint (ID 961143) is the backbone of **every later phase**.

## Findings

- **Arch 1 wins** because it learns from all 60M tokens, while Arch 3 and Arch 5 train only on the roughly 35% of documents long enough for prompt/continuation pairs (about 28× fewer target tokens), so they overfit. Arch 5's validation loss bottomed out around step 3,000–3,600 and then rose while the training loss collapsed; the notebook therefore stops at 4,000 steps and keeps the best checkpoint.
- **The attention-free CNN (Arch 4) came within 0.004 BERTScore** of the transformer.
- **BERTScore is a coarse metric here:** all five models span only 0.758–0.830.

## Known issue: future-token leak in Architecture 2

The local mask in `phase1_local_memory.ipynb` is built as

```python
rel = idx[None, :] - idx[:, None]          # j - i
local_mask = (rel >= 0) & (rel < window)   # attends to positions i .. i+255
```

so position *i* can see tokens *i … i+255*, including the one it must predict. Validation perplexity therefore dropped to 1.09 while the generated text (no future to copy from) scored worst. The correct mask is `0 <= i - j < window`. The bug was found after the deadline; the notebook is left unchanged because it is the submission that was scored.

A second, smaller issue: with about 1.13 tokens per word, a 512-word prompt is about 580 tokens, so Arch 3 and Arch 5 (which keep the first 512 tokens) never see the last 60 or so words before the continuation.

## Reproducing

1. Kaggle notebook with GPU and internet; add your Hugging Face token as a secret (the decoder notebook uses the secret name `HF_TOKEN`).
2. Run the `# %%` cells in order. Outputs go to `/kaggle/working/cs781_phase1_arch<k>/`: `legal_bpe.json`, `model.pt`, `loss.png`, `prediction.json` and the submission zip.
3. Save `legal_bpe.json` and `model.pt` as a Kaggle dataset; later phases load them from there.

## Code notes

- Comments at the top of each notebook state the source idea, our modification and the rationale.
- The `~110M params` comment in the Arch 1 config is out of date; the asserted parameter count is 66.2M.
