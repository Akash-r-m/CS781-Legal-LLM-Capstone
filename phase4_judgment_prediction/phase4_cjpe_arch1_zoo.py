# =============================================================================
# CS-781 Phase 4 -- Court Judgment Prediction (CJPE, binary 0/1) with Arch #1 (decoder-only, Phase-1 model)
#
# Long-document design
#   * A judgment is tokenised with the Phase-1 BPE, cut into chunks of CHUNK_TOK (=1023) tokens (+ <s>).
#   * DOC_MODE chooses WHICH chunks are used (this is the "which section matters" experiment):
#         tail     : last chunk only            head   : first chunk only
#         headtail : first + last chunk         tail4  : last MAX_CHUNKS chunks
#         spread   : MAX_CHUNKS chunks evenly spaced over the document (always incl. first and last)
#   * Every chunk goes through the backbone; a chunk vector = mean of its token states (causal model).
#   * Chunk vectors -> document vector by attention pooling (zero-init => starts as plain mean pooling)
#     -> linear classifier -> 2 logits.
#   * Class-weighted CE (sqrt inverse frequency), macro-F1 model selection on dev.
#
# Techniques (TECHNIQUES): full | lorma | lora | adapter | partial | normtune | kd
#   (Arch #1 has no bias terms in the backbone, so BitFit is not applicable; normtune = tune only RMSNorm gains.)
#
# Kaggle: paste each "# %%" block as its own cell, in order. Check the paths in cell [2].
# FIRST RUN: keep TECHNIQUES=["lorma"], read the "est. epoch time" line and the data stats, then scale up.
# NOT yet run on Kaggle by the assistant -- run cell [4] first and check the printed statistics.
# =============================================================================



# %% [1] Imports & HF login
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, math, random, time, zipfile, pickle, gc, traceback, functools
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, hf_hub_download

HF_TOKEN = UserSecretsClient().get_secret("HF")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [2] Config
TECHNIQUES = ["lorma"]        # e.g. ["lorma", "adapter", "full", "kd"]  ('full' must come before 'kd')
DOC_MODES = ["tail"]          # e.g. ["tail", "headtail", "spread"] -> every technique is run once per mode
FINAL_ALL = False             # True = train on train+dev for the fixed epochs (no dev score)
SEED = 42
SKIP_DONE = True
TEACHER_CKPT = ""             # KD only: path to the saved FULL run's cjpe_best.pt ("" = look in working dir, else retrain it)

ARTIFACT_PATH = "/kaggle/input/datasets/akashmalgav2/arch1final"                       # tokenizer (legal_bpe.json)
MODEL_CKPT_PATH = "/kaggle/input/datasets/akashmalgav2/phase2-model1/model_best.pt"       # Arch #1 Phase-1 checkpoint

HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
FILES = {"train": "Phase_4_CJPE/Train/train.json", "dev": "Phase_4_CJPE/Train/dev.json",
         "test": "Phase_4_CJPE/Test/test.json"}

CHUNK_TOK = 1023              # tokens per chunk (+ <s> = 1024 = Arch #1 context)
MAX_CHUNKS = 4                # used by tail4 / spread
MAX_DOC_WORDS = 16000         # very long documents: keep first HEAD_WORDS words + the tail (middle dropped)
HEAD_WORDS = 4000
MAX_TRAIN_DOCS = None         # e.g. 6000 to cap training time (random subset, fixed seed); None = all
EVAL_CAP = 800                # dev docs used for the per-epoch model-selection score (final report uses ALL dev docs)
POOL = "attn"                 # "attn" (learned, starts as mean) | "mean"
CLASS_WEIGHT = "sqrt"         # "none" | "sqrt"

CHUNKS_PER_BATCH = 8          # chunks processed per forward pass (memory budget; 8 x 1024 tokens fits a T4)
TARGET_DOC_BATCH = 16         # effective documents per optimiser step (via gradient accumulation)
WEIGHT_DECAY, WARMUP_FRAC, DROPOUT_FT = 0.01, 0.06, 0.1

LORMA_R, LORMA_ALPHA = 8, 8
ADAPTER_DIM = 32
PARTIAL_LAYERS = 4
STUDENT_KEEP = [0, 1, 3, 4, 6, 7, 9, 10, 11]   # KD student keeps 9 of 12 Phase-1 layers
KD_T, KD_ALPHA = 2.0, 0.5

HP = {   # starting points -- epochs are small because documents are long; adjust after the epoch-time estimate
    "full":     dict(epochs=3, lr_body=3e-5, lr_head=5e-4),
    "lorma":    dict(epochs=5, lr_body=4e-4, lr_head=1e-3),
    "lora":     dict(epochs=5, lr_body=4e-4, lr_head=1e-3),
    "adapter":  dict(epochs=5, lr_body=4e-4, lr_head=1e-3),
    "partial":  dict(epochs=4, lr_body=1e-4, lr_head=5e-4),
    "normtune": dict(epochs=6, lr_body=1e-3, lr_head=1e-3),
    "kd":       dict(epochs=3, lr_body=1e-4, lr_head=5e-4),
}
for _t in TECHNIQUES:
    assert _t in HP, _t
if "kd" in TECHNIQUES and "full" in TECHNIQUES:
    assert TECHNIQUES.index("full") < TECHNIQUES.index("kd"), "put 'full' before 'kd'"

ROOT_OUT = "/kaggle/working"
DOC_MODE = DOC_MODES[0]
def out_dir_for(tech):
    return f"{ROOT_OUT}/cs781_phase4_arch1_{tech}_{DOC_MODE}_s{SEED}" + ("_all" if FINAL_ALL else "")
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
print("Will run:", TECHNIQUES, "| doc modes:", DOC_MODES, "| seed:", SEED, "| final-all:", FINAL_ALL)


# %% [3] Tokenizer (Phase-1 BPE), cached; documents are stored as uint16 arrays to save RAM
with open(f"{ARTIFACT_PATH}/legal_bpe.json", "r", encoding="utf-8") as f:
    _tok = json.load(f)
vocab = _tok["itos"]
merges = [tuple(p) for p in _tok["merges"]]
token_to_id = {tok: i for i, tok in enumerate(vocab)}
merges_rank = {p: i for i, p in enumerate(merges)}
PAD_ID, UNK_ID = token_to_id["<pad>"], token_to_id["<unk>"]
BOS_ID, EOS_ID = token_to_id["<s>"], token_to_id["</s>"]
VOCAB_SIZE = len(vocab)
assert VOCAB_SIZE < 65536
EOW = "\u0100"

def word_to_symbols(word):
    return tuple(chr(b) for b in word.encode("utf-8", "ignore")) + (EOW,)

def bpe_encode_word(word):
    symbols = list(word_to_symbols(word))
    while len(symbols) > 1:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        ranked = [(merges_rank[p], p) for p in pairs if p in merges_rank]
        if not ranked:
            break
        _, best = min(ranked)
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                new_symbols.append(best[0] + best[1]); i += 2
            else:
                new_symbols.append(symbols[i]); i += 1
        symbols = new_symbols
    return symbols

@functools.lru_cache(maxsize=None)
def encode_word_cached(word):
    return tuple(token_to_id.get(sym, UNK_ID) for sym in bpe_encode_word(word))

def encode_doc(text):
    words = str(text).split()
    if len(words) > MAX_DOC_WORDS:
        words = words[:HEAD_WORDS] + words[-(MAX_DOC_WORDS - HEAD_WORDS):]
    ids = []
    for w in words:
        ids.extend(encode_word_cached(w))
    return np.asarray(ids, dtype=np.uint16)


# %% [4] Load Phase-4 data and print statistics
def hf_local(path):
    return hf_hub_download(repo_id=HF_DATASET_REPO, filename=path, repo_type="dataset", token=HF_TOKEN)

def read_records(path):
    raw = json.load(open(hf_local(path), encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        raw = raw["data"]
    recs = []
    if isinstance(raw, list):
        for i, r in enumerate(raw):
            assert isinstance(r, dict), f"{path}: record {i} is {type(r)}"
            recs.append(r)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                recs.append({"id": k, "text": v})
            else:
                d = dict(v); d.setdefault("id", k); recs.append(d)
    else:
        raise ValueError(f"{path}: unexpected top-level type {type(raw)}")
    return recs

def load_split(name, cap=None):
    recs = read_records(FILES[name])
    print(f"{name}: {len(recs)} records | keys of first record: {list(recs[0].keys())}")
    if cap is not None and len(recs) > cap:
        rng = random.Random(SEED); recs = rng.sample(recs, cap)
        print(f"  -> random subset of {cap}")
    docs, t0 = [], time.time()
    for n, r in enumerate(recs):
        lab = r.get("label", None)
        docs.append({"id": str(r["id"]), "toks": encode_doc(r.get("text") or ""),
                     "label": None if lab is None else int(lab)})
        if (n + 1) % 2000 == 0:
            print(f"  tokenised {n+1}/{len(recs)} docs ({time.time()-t0:.0f}s)")
    if name != "test":
        # the combined single+multi-instance data can repeat ids: drop exact repeats (same id, label AND text),
        # keep repeats whose content differs (they are different training examples)
        seen, uniq = set(), []
        for d in docs:
            key = (d["id"], d["label"], hash(d["toks"].tobytes()))
            if key not in seen:
                seen.add(key); uniq.append(d)
        n_same_id = len(uniq) - len({d["id"] for d in uniq})
        print(f"  {name}: removed {len(docs)-len(uniq)} exact duplicates | {n_same_id} further docs share an id with a "
              f"different text/label (kept)")
        docs = uniq
    else:
        assert len({d["id"] for d in docs}) == len(docs), "duplicate ids in TEST (Codabench needs one prediction per id)"
    for i, d in enumerate(docs):
        d["uid"] = f"{d['id']}#{i}"            # internal unique key; the original id is used only in the submission
    print(f"  {name} ready in {time.time()-t0:.0f}s")
    return docs

train_docs = load_split("train", MAX_TRAIN_DOCS)
dev_docs = load_split("dev")
test_docs = load_split("test")
assert all(d["label"] in (0, 1) for d in train_docs + dev_docs), "labels must be 0/1"
assert all(d["label"] is None for d in test_docs) or True

def stats(name, docs):
    n = np.array([len(d["toks"]) for d in docs])
    lab = Counter(d["label"] for d in docs)
    print(f"{name}: {len(docs)} docs | labels {dict(lab)} | tokens median {int(np.median(n))} "
          f"p90 {int(np.percentile(n, 90))} max {int(n.max())} | chunks(1023) median {int(np.median(np.ceil(n/1023)))}")
for nm, ds in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
    stats(nm, ds)

rng_e = random.Random(SEED)
eval_subset = dev_docs if len(dev_docs) <= EVAL_CAP else rng_e.sample(dev_docs, EVAL_CAP)
fit_docs = train_docs + dev_docs if FINAL_ALL else train_docs
eval_docs_epoch = [] if FINAL_ALL else eval_subset
print(f"fit docs {len(fit_docs)} | per-epoch eval docs {len(eval_docs_epoch)} | final report docs {0 if FINAL_ALL else len(dev_docs)}")

cnt = np.array([sum(1 for d in fit_docs if d["label"] == k) for k in (0, 1)], dtype=np.float64)
if CLASS_WEIGHT == "none":
    cw = np.ones(2)
else:
    cw = (cnt.sum() / (2 * np.maximum(cnt, 1))) ** 0.5
    cw = cw / cw.mean()
class_w = torch.tensor(cw, dtype=torch.float32).to(device)
print("class counts", cnt.tolist(), "| class weights", np.round(cw, 3).tolist())


# %% [5] Chunking, chunk selection, datasets
def chunk_starts(n, C):
    """Start offsets of 1023-token chunks; the last chunk is tail-aligned (may overlap the previous one)."""
    if n <= C:
        return [0]
    st = list(range(0, n - C + 1, C))
    if st[-1] != n - C:
        st.append(n - C)
    return st

def select_starts(starts):
    m = len(starts)
    if DOC_MODE == "tail":
        idx = [m - 1]
    elif DOC_MODE == "head":
        idx = [0]
    elif DOC_MODE == "headtail":
        idx = sorted({0, m - 1})
    elif DOC_MODE == "tail4":
        idx = list(range(max(0, m - MAX_CHUNKS), m))
    elif DOC_MODE == "spread":
        idx = sorted({int(round(x)) for x in np.linspace(0, m - 1, min(m, MAX_CHUNKS))})
    else:
        raise ValueError(DOC_MODE)
    return [starts[i] for i in idx]

def chunks_per_doc():
    return {"tail": 1, "head": 1, "headtail": 2}.get(DOC_MODE, MAX_CHUNKS)

class DocDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs
    def __len__(self):
        return len(self.docs)
    def __getitem__(self, i):
        d = self.docs[i]
        toks = d["toks"]
        starts = select_starts(chunk_starts(len(toks), CHUNK_TOK))
        chunks = [np.concatenate(([BOS_ID], toks[s:s + CHUNK_TOK])).astype(np.int64) for s in starts]
        return {"chunks": chunks, "label": -100 if d["label"] is None else d["label"]}

def collate(batch):
    chunks = [c for x in batch for c in x["chunks"]]
    Nc, T = len(chunks), max(len(c) for c in chunks)
    ids = torch.full((Nc, T), PAD_ID, dtype=torch.long)
    lens = torch.zeros(Nc, dtype=torch.long)
    for i, c in enumerate(chunks):
        ids[i, :len(c)] = torch.from_numpy(c); lens[i] = len(c)
    B, Cm = len(batch), max(len(x["chunks"]) for x in batch)
    cidx = torch.zeros((B, Cm), dtype=torch.long)
    cmask = torch.zeros((B, Cm), dtype=torch.bool)
    k = 0
    for b, x in enumerate(batch):
        for j in range(len(x["chunks"])):
            cidx[b, j] = k; cmask[b, j] = True; k += 1
    lab = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return ids, lens, cidx, cmask, lab

def loader_settings():
    dpb = max(1, CHUNKS_PER_BATCH // chunks_per_doc())
    accum = max(1, TARGET_DOC_BATCH // dpb)
    return dpb, accum

def make_loader(docs, shuffle):
    dpb, _ = loader_settings()
    return DataLoader(DocDataset(docs), batch_size=dpb, shuffle=shuffle, collate_fn=collate, num_workers=0)


# %% [6] Model: Arch #1 backbone + PEFT modules + document head
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class LoRMAPlus(nn.Module):
    """h' = h + (alpha/r) B(A h) = ((alpha/r) B A + I) W0 x with h = W0 x (multiplicative; B = 0 at init)."""
    def __init__(self, d, r, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(d, r))
        self.scale = alpha / r
    def forward(self, h):
        return h + self.scale * F.linear(F.linear(h, self.A), self.B)

class LoRAAdapter(nn.Module):
    """Additive update (alpha/r) B(A x), added to W0 x (LoRA; B = 0 at init)."""
    def __init__(self, d, r, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(d, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.scale * F.linear(F.linear(x, self.A), self.B)

class BottleneckAdapter(nn.Module):
    """Houlsby-style adapter: h + up(gelu(down(h))); `up` starts at zero (identity at init)."""
    def __init__(self, d, m):
        super().__init__()
        self.down = nn.Linear(d, m)
        self.up = nn.Linear(m, d)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head, self.head_dim, self.n_embd = n_head, n_embd // n_head, n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.adapt_mode = None
        self.q_ad = None
        self.v_ad = None
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        if self.adapt_mode == "lorma":
            q, v = self.q_ad(q), self.v_ad(v)
        elif self.adapt_mode == "lora":
            q, v = q + self.q_ad(x), v + self.v_ad(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, n_embd, mult, dropout):
        super().__init__()
        hidden = ((int(mult * n_embd) + 63) // 64) * 64
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, ffn_mult, dropout):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, ffn_mult, dropout)
        self.ad1 = None
        self.ad2 = None
    def forward(self, x, cos, sin):
        a = self.attn(self.ln1(x), cos, sin)
        if self.ad1 is not None:
            a = self.ad1(a)
        x = x + a
        f = self.ffn(self.ln2(x))
        if self.ad2 is not None:
            f = self.ad2(f)
        return x + f

class LegalLLaMA(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, ffn_mult, dropout):
        super().__init__()
        self.block_size, self.head_dim = block_size, n_embd // n_head
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, ffn_mult, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        cos, sin = build_rope_cache(block_size, self.head_dim, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
    def forward_hidden(self, idx):
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos.to(idx.device), self.rope_sin.to(idx.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)

class DocModel(nn.Module):
    """Backbone over every selected chunk -> mean-pool tokens -> pool chunks (attention or mean) -> classifier."""
    def __init__(self, backbone, d, n_cls, dropout, pool):
        super().__init__()
        self.backbone, self.pool = backbone, pool
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, n_cls)
        nn.init.normal_(self.classifier.weight, std=0.02); nn.init.zeros_(self.classifier.bias)
        self.attn_score = nn.Linear(d, 1)
        nn.init.zeros_(self.attn_score.weight); nn.init.zeros_(self.attn_score.bias)   # uniform weights at init
    def head_parameters(self):
        return list(self.classifier.parameters()) + list(self.attn_score.parameters())
    def forward(self, ids, lens, cidx, cmask):
        h = self.backbone.forward_hidden(ids)                                   # (Nc, T, d)
        T = h.size(1)
        tm = (torch.arange(T, device=ids.device)[None, :] < lens[:, None]).unsqueeze(-1).float()
        cv = (h.float() * tm).sum(1) / lens.unsqueeze(-1).float()               # (Nc, d) mean over real tokens
        D = cv[cidx]                                                            # (B, Cmax, d)
        if self.pool == "attn":
            s = self.attn_score(D).squeeze(-1).masked_fill(~cmask, -1e4)
            w = torch.softmax(s, dim=1)
        else:
            w = cmask.float() / cmask.float().sum(1, keepdim=True)
        doc = (w.unsqueeze(-1) * D).sum(1)
        return self.classifier(self.drop(doc))                                  # (B, 2)

ckpt = torch.load(MODEL_CKPT_PATH, map_location="cpu")
cfg = ckpt["config"]
assert cfg["vocab_size"] == VOCAB_SIZE, (cfg["vocab_size"], VOCAB_SIZE)
assert CHUNK_TOK + 1 <= cfg["block_size"], f"CHUNK_TOK+1 exceeds the model context {cfg['block_size']}"
print(f"Phase-1 model: {cfg['n_layer']} layers, d={cfg['n_embd']}, context {cfg['block_size']}")

def make_student_state(sd, keep):
    new = {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            old = int(parts[1])
            if old in keep:
                parts[1] = str(keep.index(old))
                new[".".join(parts)] = v
        else:
            new[k] = v
    return new

def build_model(technique, keep_layers=None):
    sd, n_layer = ckpt["model_state_dict"], cfg["n_layer"]
    if keep_layers is not None:
        sd, n_layer = make_student_state(sd, keep_layers), len(keep_layers)
    bb = LegalLLaMA(cfg["vocab_size"], n_layer, cfg["n_head"], cfg["n_embd"],
                    cfg["block_size"], cfg["ffn_mult"], DROPOUT_FT)
    bb.load_state_dict(sd)
    d = cfg["n_embd"]
    if technique in ("lorma", "lora", "adapter", "partial", "normtune"):
        for p in bb.parameters():
            p.requires_grad = False
        if technique in ("lorma", "lora"):
            for blk in bb.blocks:
                blk.attn.adapt_mode = technique
                make = LoRMAPlus if technique == "lorma" else LoRAAdapter
                blk.attn.q_ad, blk.attn.v_ad = make(d, LORMA_R, LORMA_ALPHA), make(d, LORMA_R, LORMA_ALPHA)
        elif technique == "adapter":
            for blk in bb.blocks:
                blk.ad1, blk.ad2 = BottleneckAdapter(d, ADAPTER_DIM), BottleneckAdapter(d, ADAPTER_DIM)
        elif technique == "partial":
            for blk in bb.blocks[-PARTIAL_LAYERS:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in bb.ln_f.parameters():
                p.requires_grad = True
        elif technique == "normtune":
            for m in bb.modules():
                if isinstance(m, RMSNorm):
                    m.weight.requires_grad = True
    net = DocModel(bb, d, 2, DROPOUT_FT, POOL).to(device)
    n_tot = sum(p.numel() for p in net.parameters())
    n_trn = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[{technique}{' student' if keep_layers else ''}] trainable {n_trn/1e6:.3f}M of {n_tot/1e6:.1f}M params ({100*n_trn/n_tot:.2f}%)")
    return net


# %% [7] Inference and metrics (macro-F1 over classes 0 and 1)
@torch.no_grad()
def predict_probs(net, docs):
    net.eval()
    out, k = {}, 0
    for ids, lens, cidx, cmask, _ in make_loader(docs, False):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = net(ids.to(device), lens.to(device), cidx.to(device), cmask.to(device))
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        for j in range(len(probs)):
            out[docs[k]["uid"]] = probs[j]; k += 1
    return out

def score(probs_by_id, docs, thr=0.5):
    y_true = [d["label"] for d in docs]
    y_pred = [int(probs_by_id[d["uid"]][1] >= thr) for d in docs]
    p, r_, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)), \
           float(p.mean()), float(r_.mean()), (p, r_, f, s), y_true, y_pred

def report(probs_by_id, docs):
    f1, mp, mr, (p, r_, f, s), y_true, y_pred = score(probs_by_id, docs)
    print(f"dev macro-F1 {f1:.4f} | macro-P {mp:.4f} | macro-R {mr:.4f}  ({len(docs)} docs)")
    for k in (0, 1):
        print(f"  class {k}: P {p[k]:.3f} R {r_[k]:.3f} F1 {f[k]:.3f} support {int(s[k])}")
    print("  confusion [rows=true 0/1, cols=pred 0/1]:", confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist())
    best = max(((score(probs_by_id, docs, t)[0], t) for t in np.arange(0.30, 0.71, 0.05)))
    print(f"  (info) best-threshold dev macro-F1 {best[0]:.4f} at thr {best[1]:.2f}  -- tuned on dev, optimistic")
    return f1


# %% [8] Training function (gradient accumulation; optional teacher for distillation)
def fit(model, loader, epochs, lr_body, lr_head, best_path, teacher=None):
    _, ACCUM = loader_settings()
    head_params = model.head_parameters()
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": lr_head, "base": lr_head}]
    if body_params:
        groups.insert(0, {"params": body_params, "lr": lr_body, "base": lr_body})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    steps_per_epoch = math.ceil(len(loader) / ACCUM)
    total_steps = epochs * steps_per_epoch
    warm = max(1, int(WARMUP_FRAC * total_steps))
    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * prog)))

    best_f1, step, t0, history, eta_done = -1.0, 0, time.time(), [], False
    if teacher is not None:
        teacher.eval()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, n_b = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, (ids, lens, cidx, cmask, lab) in enumerate(loader):
            ids, lens, cidx, cmask, lab = ids.to(device), lens.to(device), cidx.to(device), cmask.to(device), lab.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(ids, lens, cidx, cmask)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(ids, lens, cidx, cmask)
            logits = logits.float()
            loss = F.cross_entropy(logits, lab, weight=class_w)
            if teacher is not None:
                s_ = F.log_softmax(logits / KD_T, dim=-1)
                t_ = F.softmax(t_logits.float() / KD_T, dim=-1)
                kd = F.kl_div(s_, t_, reduction="batchmean") * (KD_T ** 2)
                loss = (1 - KD_ALPHA) * loss + KD_ALPHA * kd
            scaler.scale(loss / ACCUM).backward()
            run_loss += loss.item(); n_b += 1
            if (bi + 1) % ACCUM == 0 or bi + 1 == len(loader):
                for g in optimizer.param_groups:
                    g["lr"] = g["base"] * lr_factor(step)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if not eta_done and bi == 49:
                per = (time.time() - t0) / 50
                print(f"  est. epoch time {per*len(loader)/60:.1f} min ({len(loader)} batches, {per:.2f}s/batch) "
                      f"-> {epochs} epochs ~ {epochs*per*len(loader)/60:.0f} min (+ eval)")
                eta_done = True
        rec_h = {"epoch": epoch, "train_loss": run_loss / max(1, n_b)}
        if eval_docs_epoch:
            f1, mp, mr, *_ = score(predict_probs(model, eval_docs_epoch), eval_docs_epoch)
            rec_h["dev_f1"] = f1
            print(f"[epoch {epoch}] loss {rec_h['train_loss']:.4f} | dev(sub) macro-F1 {f1:.4f} (P {mp:.4f} R {mr:.4f}) | {(time.time()-t0)/60:.1f}m")
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"model": model.state_dict(), "epoch": epoch, "dev_f1": f1}, best_path)
                print("  -> new best checkpoint saved")
        else:
            print(f"[epoch {epoch}] loss {rec_h['train_loss']:.4f} (final run on train+dev; no dev score) | {(time.time()-t0)/60:.1f}m")
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev_f1": None}, best_path)
        history.append(rec_h)
    model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    return history


# %% [9] One technique end-to-end (train -> dev report -> test predictions -> save probabilities + efficiency numbers)
def set_seed(s):
    random.seed(s); torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def run_technique(tech):
    out_dir = out_dir_for(tech)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "cjpe_best.pt")
    pred_path = os.path.join(out_dir, "prediction.json")
    summ_path = os.path.join(out_dir, "summary.json")
    prob_path = os.path.join(out_dir, "doc_probs.pkl")
    if SKIP_DONE and all(os.path.exists(p) for p in [pred_path, summ_path, prob_path]):
        print(f"\n##### [{tech}/{DOC_MODE}] already finished -> skipping")
        return json.load(open(summ_path))

    print(f"\n{'='*78}\n##### Phase 4 | Arch1 | {tech} | doc mode {DOC_MODE} | seed {SEED} | {HP[tech]}\n{'='*78}")
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    loader = make_loader(fit_docs, True)
    teacher = None
    if tech == "kd":
        teacher = build_model("full")
        full_ckpt = TEACHER_CKPT or os.path.join(out_dir_for("full"), "cjpe_best.pt")
        if os.path.exists(full_ckpt):
            teacher.load_state_dict(torch.load(full_ckpt, map_location="cpu")["model"])
            print("Loaded teacher from", full_ckpt)
        else:
            print("No saved teacher found -> training the teacher (full fine-tuning) first")
            fit(teacher, loader, best_path=os.path.join(out_dir, "teacher_best.pt"), **HP["full"])
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        if eval_docs_epoch:
            print(f"TEACHER dev(sub) macro-F1: {score(predict_probs(teacher, eval_docs_epoch), eval_docs_epoch)[0]:.4f}")
        model = build_model("full", keep_layers=STUDENT_KEEP)
        t_p, s_p = sum(p.numel() for p in teacher.parameters()), sum(p.numel() for p in model.parameters())
        print(f"Teacher {t_p/1e6:.1f}M -> student {s_p/1e6:.1f}M params ({100*(1-s_p/t_p):.0f}% smaller)")
        history = fit(model, loader, best_path=best_path, teacher=teacher, **HP["kd"])
    else:
        model = build_model(tech)
        history = fit(model, loader, best_path=best_path, **HP[tech])
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    train_min = (time.time() - t_run) / 60
    print(f"[{tech}] training finished in {train_min:.1f} min")

    dev_f1, dev_probs = None, {}
    if not FINAL_ALL:
        dev_probs = predict_probs(model, dev_docs)
        dev_f1 = report(dev_probs, dev_docs)
    t_inf = time.time()
    test_probs = predict_probs(model, test_docs)
    inf_sec = time.time() - t_inf
    pred = [{"id": d["id"], "pred": int(test_probs[d["uid"]].argmax())} for d in test_docs]
    assert len(pred) == len(test_docs) and {p["id"] for p in pred} == {d["id"] for d in test_docs}
    assert all(p["pred"] in (0, 1) for p in pred)
    print(f"[{tech}] test docs {len(pred)} | predicted label counts:", dict(Counter(p["pred"] for p in pred)),
          f"| test inference {inf_sec:.0f}s")
    json.dump(pred, open(pred_path, "w"))
    pickle.dump({"labels": [0, 1], "dev": {k: v.astype(np.float32) for k, v in dev_probs.items()},
                 "test": {k: v.astype(np.float32) for k, v in test_probs.items()}}, open(prob_path, "wb"))
    n_tot = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"technique": tech, "doc_mode": DOC_MODE, "dev_f1": dev_f1, "train_minutes": round(train_min, 1),
               "test_inference_sec": round(inf_sec, 1), "params_total_M": round(n_tot / 1e6, 2),
               "params_trainable_M": round(n_trn / 1e6, 3),
               "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
    json.dump(summary, open(summ_path, "w"), indent=2)
    del model, teacher
    return summary


# %% [10] Run everything chosen in cell [2]
all_results = []
for _mode in DOC_MODES:
    DOC_MODE = _mode
    for tech in TECHNIQUES:
        try:
            res = run_technique(tech)
        except Exception:
            print(f"\n!!!!! [{tech}/{DOC_MODE}] FAILED -- continuing")
            traceback.print_exc()
            res = {"technique": tech, "doc_mode": DOC_MODE, "error": True}
        all_results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\n" + "=" * 78 + "\nSUMMARY (dev macro-F1 on the full dev set)\n" + "=" * 78)
for r in all_results:
    if r.get("error"):
        print(f"{r['technique']:9s} {r['doc_mode']:9s} FAILED")
    else:
        d = f"{r['dev_f1']:.4f}" if r["dev_f1"] is not None else "n/a"
        print(f"{r['technique']:9s} {r['doc_mode']:9s} dev macro-F1 {d} | train {r['train_minutes']} min | "
              f"params {r['params_total_M']}M (trainable {r['params_trainable_M']}M) | mem {r['peak_gpu_mem_GB']} GB")


# %% [11] Zips: one Codabench submission zip per run (prediction.json at the zip root) + a results zip
SUB_DIR, RES_DIR = f"{ROOT_OUT}/submissions", f"{ROOT_OUT}/results_zips"
os.makedirs(SUB_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
for _mode in DOC_MODES:
    DOC_MODE = _mode
    for t in TECHNIQUES:
        d = out_dir_for(t)
        if not os.path.exists(f"{d}/prediction.json"):
            print(f"[{t}/{_mode}] prediction.json not found -> skipped"); continue
        tag = f"phase4_arch1_{t}_{_mode}_s{SEED}" + ("_all" if FINAL_ALL else "")
        with zipfile.ZipFile(f"{SUB_DIR}/submission_{tag}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(f"{d}/prediction.json", arcname="prediction.json")
        with zipfile.ZipFile(f"{RES_DIR}/results_{tag}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in ["prediction.json", "doc_probs.pkl", "history.json", "summary.json"]:
                if os.path.exists(f"{d}/{fn}"):
                    zf.write(f"{d}/{fn}", arcname=fn)
        print(f"[{t}/{_mode}] submission_{tag}.zip")
print("Upload the zips in", SUB_DIR, "to Codabench (all of them are different experiments only if technique/representation differ).")


# %% [12] OPTIONAL ensemble of saved runs (CPU only). Fill ENSEMBLE_MEMBERS with doc_probs.pkl paths; {} = skip.
ENSEMBLE_MEMBERS = {   # name: (path to doc_probs.pkl, weight)
    # "lorma_tail":   ("/kaggle/working/cs781_phase4_arch1_lorma_tail_s42/doc_probs.pkl", 1),
    # "adapter_tail": ("/kaggle/working/cs781_phase4_arch1_adapter_tail_s42/doc_probs.pkl", 1),
}
if ENSEMBLE_MEMBERS:
    mem = []
    for name, (p, w) in ENSEMBLE_MEMBERS.items():
        d = pickle.load(open(p, "rb")); mem.append((d, w))
    tot = sum(w for _, w in mem)
    avg = lambda split, i: sum(w * d[split][i] for d, w in mem) / tot
    thr = 0.5
    if all(d["dev"] for d, _ in mem):
        dp = {x["uid"]: avg("dev", x["uid"]) for x in dev_docs}
        print("ensemble dev macro-F1 @0.5:", round(score(dp, dev_docs)[0], 4))
        best = max((score(dp, dev_docs, t)[0], t) for t in np.arange(0.35, 0.66, 0.05))
        if best[0] - score(dp, dev_docs)[0] > 0.01:
            thr = float(best[1]); print("tuned threshold", thr, "->", round(best[0], 4))
    pred = [{"id": x["id"], "pred": int(avg("test", x["uid"])[1] >= thr)} for x in test_docs]
    assert len(pred) == len(test_docs)
    ED = f"{ROOT_OUT}/phase4_ensemble"; os.makedirs(ED, exist_ok=True)
    json.dump(pred, open(f"{ED}/prediction.json", "w"))
    with zipfile.ZipFile(f"{ED}/submission_phase4_ensemble.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(f"{ED}/prediction.json", arcname="prediction.json")
    print("predicted counts", dict(Counter(p["pred"] for p in pred)), "| wrote", f"{ED}/submission_phase4_ensemble.zip")

