# finetune_gpt2_midi_single_file.py
# A-to-Z: Encode MIDI -> build vocab -> load pretrained GPT-2 -> fine-tune -> generate -> decode to MIDI
#
# Requirements:
#   pip install torch transformers
# Plus your local encode/decode:
#   from encode_decode import encode_midi, decode_tokens
#
# Notes:
# - This uses a custom word-level tokenizer by mapping tokens <-> ids directly.
# - We fine-tune GPT-2 (pretrained) by resizing token embeddings to your MIDI vocab.
# - Works on Apple Silicon (mps) or CPU/CUDA.
#
# Run:
#   python finetune_gpt2_midi_single_file.py

import os
import math
import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2LMHeadModel, GPT2Config

# ---- your code ----
from encode_decode import encode_midi, decode_tokens


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_dir: str = "data/giant_midiV1"
    out_midi: str = "data/output_gpt2_finetuned.mid"

    # data
    repeat_tokens: int = 3          # like your tokens = tokens * 10
    seq_len: int = 128              # GPT-2 likes longer context than 8
    batch_size: int = 16
    epochs: int = 2

    # training
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # generation
    gen_max_new: int = 600
    sample: bool = True
    temperature: float = 1.0
    top_k: int = 64

    seed: int = 42


CFG = CFG()


# ============================================================
# DEVICE
# ============================================================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print("Using device:", device)


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# ============================================================
# 1) LOAD MIDI -> TOKENS
# ============================================================
midi_files = [
    "a-jag-je-t-aime-juliette-oxc7fd0zn8o.mid",
    "aaron-michael-piano-course-v8wvkk-1b2c.mid",
    "abel-frederic-lola-polka-slnjf0uiqrw.mid",
    "abramowitsch-a-3-nocturnes-lkk-7l2ppw8.mid",
    "abreu-zequinha-tico-tico-no-fubá-kyeufpjuwfc.mid",
    "achron-joseph-2-pieces-op-56-snqknjnxsoe.mid",
    "abt-franz-7-lieder-aus-dem-buche-der-liebe-op-39-k33a-r6ikea.mid",
    "achron-joseph-symphonic-variations-and-sonata-op-39-rtrrqwo2hng.mid",
    "ackerman-gabriel-j-ballade-no-1-op-5-koy5xwvk5c4.mid",
    "adalid-marcial-del-24-romances-sans-paroles-fazuwhltgtm.mid",
    "adalid-marcial-del-sonata-for-piano-four-hands-5vmrdbna2o4.mid",
]

tokens: List[str] = []
for fn in midi_files:
    path = os.path.join(CFG.data_dir, fn)
    toks = encode_midi(path)
    tokens.extend(toks)

tokens = [t.upper() for t in tokens]
tokens = tokens * CFG.repeat_tokens
print("Total tokens:", len(tokens))


# ============================================================
# 2) BUILD VOCAB (WORD/EVENT LEVEL) + SPECIAL TOKENS
# ============================================================
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"

vocab = [PAD, BOS, EOS] + sorted(set(tokens))
stoi = {t: i for i, t in enumerate(vocab)}
itos = {i: t for t, i in stoi.items()}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

# Encode full stream
encoded = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)


# ============================================================
# 3) DATASET
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


dataset = SeqDataset(encoded, CFG.seq_len)
loader = DataLoader(
    dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,  # keep 0 for MPS
)

print("Batches per epoch:", len(loader))


# ============================================================
# 4) LOAD PRETRAINED GPT-2 AND RESIZE FOR MIDI VOCAB
# ============================================================
# IMPORTANT:
# - We keep GPT-2 weights, but change vocab size and context length.
# - ignore_mismatched_sizes=True allows loading even though embeddings shapes differ.
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=CFG.seq_len,
    n_ctx=CFG.seq_len,
)

model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    config=config,
    ignore_mismatched_sizes=True,
)

model.resize_token_embeddings(vocab_size)
model.to(device)


# ============================================================
# 5) OPTIMIZER
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.weight_decay,
)


# ============================================================
# 6) TRAIN (CAUSAL LM)
# ============================================================
def train():
    model.train()
    for epoch in range(CFG.epochs):
        total = 0.0
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)

            out = model(input_ids=X, labels=Y)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if CFG.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

            optimizer.step()
            total += float(loss.item())

        print(f"epoch {epoch:02d} | loss {total/len(loader):.4f}")


# ============================================================
# 7) GENERATION
# ============================================================
@torch.inference_mode()
def generate(start_tokens: List[str], max_new: int):
    model.eval()

    # seed sequence = BOS + provided start tokens
    ids = [stoi[BOS]] + [stoi[t] for t in start_tokens if t in stoi]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        idx_cond = idx[:, -CFG.seq_len:]
        logits = model(idx_cond).logits[:, -1, :]  # (B, vocab)

        if CFG.sample:
            logits = logits / max(1e-8, CFG.temperature)

            if CFG.top_k is not None and CFG.top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=min(CFG.top_k, logits.size(-1)))
                probs = torch.softmax(topk_vals, dim=-1)
                next_local = torch.multinomial(probs, num_samples=1)
                next_id = topk_idx.gather(-1, next_local)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat([idx, next_id], dim=1)

        if next_id.item() == stoi[EOS]:
            break

    return [itos[int(i)] for i in idx[0].tolist()]


# ============================================================
# 8) RUN END-TO-END
# ============================================================
if __name__ == "__main__":
    print("\n--- TRAINING ---")
    train()

    print("\n--- GENERATION ---")
    # pick a small seed from your real data (must exist in vocab)
    seed = tokens[:8]
    out_tokens = generate(seed, max_new=CFG.gen_max_new)

    # Strip special tokens before decoding
    out_tokens = [t for t in out_tokens if t not in (PAD, BOS, EOS)]

    print("Generated tokens:", len(out_tokens))
    print("Decoding to MIDI...")

    decode_tokens(out_tokens, CFG.out_midi)
    print("Saved:", CFG.out_midi)
