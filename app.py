# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import json
import os

# Cargar vocab (debe existir tras entrenamiento)
if os.path.exists("vocab.json"):
    with open("vocab.json","r") as f:
        vocab = json.load(f)
else:
    # fallback si no existe
    vocab = {"jab":0,"cross":1,"hook":2,"uppercut":3,"low_kick":4,"body_kick":5,"takedown":6,"clinch":7,"defense":8,"idle":9}

inv_vocab = {int(v):k for k,v in vocab.items()}
ACTIONS = [inv_vocab[i] for i in range(len(inv_vocab))]

# Modelo (misma arquitectura que en train)
import torch.nn as nn
SEQLEN = 16
EMBED_DIM = 64
N_ACTIONS = len(ACTIONS)

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

@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
        # modelo no entrenado: devolvemos modelo con pesos aleatorios
        m = SimpleTransformer(n_tokens=N_ACTIONS)
        return m
    ckpt = torch.load("model.pth", map_location="cpu")
    m = SimpleTransformer(n_tokens=N_ACTIONS)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m

st.title("UFC Next-action predictor (demo)")
st.markdown("Introduce una secuencia separada por comas (ej: jab,cross,hook,defense). Últimos 16 eventos usados como entrada.")

model = load_model()

user_seq = st.text_input("Secuencia (máx 16 eventos)", ",".join(ACTIONS[:8]))
top_k = st.slider("Top-k", min_value=1, max_value=min(5, len(ACTIONS)), value=3)

def seq_to_tensor(s):
    tokens = [t.strip() for t in s.split(",") if t.strip()!=""]
    tokens = tokens[-SEQLEN:]
    ids = []
    for t in tokens:
        ids.append(vocab.get(t, vocab.get("idle",0)))
    if len(ids) < SEQLEN:
        pad = [vocab.get("idle",0)]*(SEQLEN-len(ids))
        ids = pad + ids
    return torch.tensor([ids], dtype=torch.long)

if st.button("Predecir"):
    seq_t = seq_to_tensor(user_seq)
    logits = model(seq_t)
    probs = F.softmax(logits, dim=-1).detach().numpy()[0]
    top_idxs = probs.argsort()[::-1][:top_k]
    st.write("Entrada (últimos tokens):", user_seq.split(",")[-SEQLEN:])
    st.write("Predicciones:")
    for i in top_idxs:
        st.write(f"- **{inv_vocab[i]}** — prob: {probs[i]:.3f}")

st.markdown("---")
st.write("Vocab:", ACTIONS)
