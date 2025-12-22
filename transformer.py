import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DEVICE (MPS / CPU)
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. RAW TEXT
# ============================================================
text = """
the cat sat on the mat
the dog sat on the log
the cat saw the dog
the dog saw the cat
"""

# increase data size (safe)
text = text * 1000
words = text.lower().split()

# ============================================================
# 2. TOKENIZER (word-level)
# ============================================================
vocab = sorted(set(words))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)

encoded = torch.tensor([stoi[w] for w in words], dtype=torch.long)

# ============================================================
# 3. TRAINING DATA
# ============================================================
def make_batches(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + 1:i + seq_len + 1])
    return torch.stack(X), torch.stack(Y)

SEQ_LEN = 5
X, Y = make_batches(encoded, SEQ_LEN)

X = X.to(device)
Y = Y.to(device)

# ============================================================
# 4. GPT-2 STYLE MODEL
# ============================================================
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

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
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads)
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
            [GPT2Block(d_model, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-2 weight tying
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
    n_layers=4,   # GPT-2 depth
    n_heads=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(400):
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch} | loss {loss.item():.4f}")

# ============================================================
# 6. GENERATION
# ============================================================
@torch.no_grad()
def generate(start_words, max_new=10):
    model.eval()

    idx = torch.tensor(
        [[stoi[w] for w in start_words]],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new):
        idx_cond = idx[:, -SEQ_LEN:]
        logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)

    return " ".join(itos[i.item()] for i in idx[0])

print("\nGenerated:")
print(generate(["the", "cat"]))
