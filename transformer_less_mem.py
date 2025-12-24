import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from encode_decode import encode_midi, decode_tokens

# ============================================================
# DEVICE
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. LOAD MIDI → TOKENS
# ============================================================
tokens = encode_midi("data/sample1.mid")
tokens = [t.upper() for t in tokens]

# ============================================================
# 2. TOKENIZER (WORD / EVENT LEVEL)
# ============================================================
tokens = tokens * 10
vocab = sorted(set(tokens))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)

encoded = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)

# ============================================================
# 3. DATASET (STREAMING, NO HUGE TENSORS)
# ============================================================
SEQ_LEN = 8

class SeqDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

dataset = SeqDataset(encoded, SEQ_LEN)
loader = DataLoader(
    dataset,
    batch_size=64,     # ← try 64 or even 128
    shuffle=True,
    drop_last=True,
    num_workers=0      # keep 0 for MPS
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "mps"))
# ============================================================
# 4. GPT-2 STYLE MODEL (MEMORY OPTIMIZED)
# ============================================================
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # register causal mask ONCE
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2Mini(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_layers, n_heads):
        super().__init__()
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList(
            [GPT2Block(d_model, n_heads, seq_len) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

# ============================================================
# 5. TRAINING
# ============================================================
model = GPT2Mini(
    vocab_size=vocab_size,
    seq_len=SEQ_LEN,
    d_model=64,
    n_layers=4,
    n_heads=4,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="mps", dtype=torch.float16):
            logits = model(X)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                Y.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    if epoch % 5 == 0:
        print(f"epoch {epoch} | loss {total_loss / len(loader):.4f}")


# ============================================================
# 6. GENERATION (LOW-MEM)
# ============================================================
@torch.inference_mode()
def generate(start_tokens, max_new=100):
    model.eval()

    idx = torch.tensor(
        [[stoi[t] for t in start_tokens]],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new):
        idx_cond = idx[:, -SEQ_LEN:]
        logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1], dim=-1)
        idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)

    return [itos[i.item()] for i in idx[0]]

# ============================================================
# 7. RUN
# ============================================================
print("\nGenerating...")
out_tokens = generate(tokens[:2], max_new=200)

decode_tokens(out_tokens, "data/output_transformer.mid")
print("Saved: data/output_transformer.mid")
