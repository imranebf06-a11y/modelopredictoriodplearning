# train_and_save.py
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Config ---
ACTIONS = ["jab","cross","hook","uppercut","low_kick","body_kick","takedown","clinch","defense","idle"]
N_ACTIONS = len(ACTIONS)
MODEL_PATH = "model.pth"
VOCAB_PATH = "vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQLEN = 16
EMBED_DIM = 64
BATCH_SIZE = 128
EPOCHS = 6
LR = 1e-3
# --------------

# Save vocab (so app.py puede cargarlo)
vocab = {a:i for i,a in enumerate(ACTIONS)}
with open(VOCAB_PATH,"w") as f:
    json.dump(vocab,f)

class SyntheticUFCDataset(Dataset):
    def __init__(self, n_samples=12000, seq_len=SEQLEN):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab = vocab
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        seq = []
        for _ in range(self.seq_len):
            if random.random() < 0.12:
                # small patterned behaviour
                p = random.choice([["jab","cross","hook"],["low_kick","body_kick"],["jab","jab","cross"]])
                choice = random.choice(p)
                seq.append(self.vocab.get(choice, self.vocab["idle"]))
            else:
                seq.append(random.randrange(N_ACTIONS))
        last = seq[-3:]
        if any(tok == self.vocab["low_kick"] for tok in last) and random.random() < 0.6:
            target = self.vocab["body_kick"]
        elif any(tok == self.vocab["jab"] for tok in last) and random.random() < 0.5:
            target = self.vocab["cross"]
        else:
            target = random.randrange(N_ACTIONS)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class SimpleTransformer(nn.Module):
    def __init__(self, n_tokens, embed_dim=EMBED_DIM, n_heads=4, ff_dim=128, n_layers=2, seq_len=SEQLEN, out_classes=N_ACTIONS):
        super().__init__()
        self.token_emb = nn.Embedding(n_tokens, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim//2, out_classes)
        )
    def forward(self, x):
        b,s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b,-1)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,2,0)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def train():
    ds = SyntheticUFCDataset(n_samples=12000)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleTransformer(n_tokens=N_ACTIONS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for seq, tgt in loader:
            seq = seq.to(DEVICE)
            tgt = tgt.to(DEVICE)
            logits = model(seq)
            loss = loss_fn(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * seq.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == tgt).sum().item()
            n += seq.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss/n:.4f} acc={correct/n:.4f}")
    torch.save({"model_state": model.state_dict(),
                "config": {"n_tokens": N_ACTIONS, "embed_dim": EMBED_DIM, "seq_len": SEQLEN}}, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    train()
