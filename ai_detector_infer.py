"""
AI Detector — Inference
========================
Load trained_model.json and classify new texts.

Usage:
    python ai_detector_infer.py "Your text to classify here"
    python ai_detector_infer.py --file myfile.txt
"""

import json, sys, math, re
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import os; os.system("pip install torch --quiet")
    import torch, torch.nn as nn, torch.nn.functional as F


# ─── Re-import model classes from training script ─────────────────────────────
# (In production, factor these into a shared module; here we inline for portability)

class SimpleTokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def __init__(self, vocab_size=30_000):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}

    def _pre_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text, max_len=256):
        tokens = [self.BOS]
        for tok in self._pre_tokenize(text):
            tokens.append(self.token2id.get(tok, self.UNK))
        tokens.append(self.EOS)
        tokens = tokens[:max_len]
        tokens += [self.PAD] * (max_len - len(tokens))
        return tokens

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["vocab_size"])
        obj.token2id = {k: int(v) for k, v in d["token2id"].items()}
        obj.id2token = {v: k for k, v in obj.token2id.items()}
        return obj


class TrajectoryExtractor(nn.Module):
    def __init__(self, embed_dim, traj_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + 1, traj_dim * 2), nn.GELU(),
            nn.Linear(traj_dim * 2, traj_dim),
        )
        self.norm = nn.LayerNorm(traj_dim)

    def forward(self, embeds):
        delta = embeds[:, 1:, :] - embeds[:, :-1, :]
        dist  = delta.norm(dim=-1, keepdim=True)
        traj  = torch.cat([delta, dist], dim=-1)
        return self.norm(self.proj(traj))


class PatternCodebook(nn.Module):
    def __init__(self, n_codes, traj_dim, top_k=8):
        super().__init__()
        self.codebook = nn.Parameter(F.normalize(torch.randn(n_codes, traj_dim), dim=-1))
        self.top_k    = min(top_k, n_codes)

    def forward(self, traj):
        B, T, D  = traj.shape
        flat      = traj.reshape(-1, D)
        cb_norm   = F.normalize(self.codebook, dim=-1)
        flat_norm = F.normalize(flat, dim=-1)
        sims      = flat_norm @ cb_norm.T
        topk_sim, topk_idx = sims.topk(self.top_k, dim=-1)
        weights   = F.softmax(topk_sim, dim=-1)
        selected  = self.codebook[topk_idx]
        assigned  = (weights.unsqueeze(-1) * selected).sum(dim=1).reshape(B, T, D)
        return assigned, torch.tensor(0.0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


class AIDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V, D    = cfg["vocab_size"], cfg["embed_dim"]
        T_dim   = cfg["traj_dim"]
        self.embed          = nn.Embedding(V, D, padding_idx=0)
        self.traj_extractor = TrajectoryExtractor(D, T_dim)
        self.codebook       = PatternCodebook(cfg["pattern_codes"], T_dim, cfg["pattern_top_k"])
        self.pos_embed      = nn.Embedding(cfg["max_seq_len"], T_dim)
        self.encoder        = nn.ModuleList([
            TransformerBlock(T_dim, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])
        self.out_norm       = nn.LayerNorm(T_dim)
        self.classifier     = nn.Sequential(
            nn.Linear(T_dim, T_dim // 2), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(T_dim // 2, 2),
        )
        self.cfg = cfg

    def forward(self, token_ids):
        B, T   = token_ids.shape
        emb    = self.embed(token_ids)
        traj   = self.traj_extractor(emb)
        assigned, _ = self.codebook(traj)
        x      = traj + assigned
        seq_len= x.size(1)
        pos_ids= torch.arange(seq_len, device=x.device).unsqueeze(0)
        x      = x + self.pos_embed(pos_ids)
        pad_mask = (token_ids[:, :-1] == 0)
        for block in self.encoder:
            x = block(x, key_padding_mask=pad_mask)
        x        = self.out_norm(x)
        non_pad  = (~pad_mask).float().unsqueeze(-1)
        pooled   = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        logits   = self.classifier(pooled)
        return logits, torch.tensor(0.0)


# ─── Loader ──────────────────────────────────────────────────────────────────

def load_trained_model(path="trained_model (13).json"):
    print(f"Loading model from {path}…")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg       = payload["config"]
    tokenizer = SimpleTokenizer.from_dict(payload["tokenizer"])
    model     = AIDetector(cfg)

    # Reconstruct state dict
    sd = {}
    for k, v in payload["weights"].items():
        arr    = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
        sd[k]  = torch.tensor(arr)
    model.load_state_dict(sd)
    model.eval()

    # Use calibrated threshold saved during training (fixes false positives)
    threshold = payload["metrics"].get("calibrated_threshold",
                cfg.get("decision_threshold", 0.5))
    print(f"Model loaded | balanced_acc={payload['metrics'].get('final_balanced_accuracy', '?')} "
          f"| decision threshold={threshold:.2f}")
    return model, tokenizer, cfg, threshold


@torch.no_grad()
def predict(text: str, model: AIDetector, tokenizer: SimpleTokenizer,
            cfg: dict, threshold: float = 0.5) -> dict:
    ids    = tokenizer.encode(text, cfg["max_seq_len"])
    tensor = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]
    prob_ai = float(probs[1])
    pred    = 1 if prob_ai >= threshold else 0
    return {
        "prediction"  : "AI" if pred == 1 else "Human",
        "confidence"  : round((prob_ai if pred == 1 else float(probs[0])) * 100, 2),
        "prob_human"  : round(float(probs[0]) * 100, 2),
        "prob_ai"     : round(prob_ai * 100, 2),
        "threshold"   : threshold,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Text Detector — Inference")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--file", "-f", help="Path to a text file to classify")
    parser.add_argument("--model", "-m", default="trained_model (13).json")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Override decision threshold (default: use calibrated value from model)")
    args = parser.parse_args()

    model, tokenizer, cfg, cal_threshold = load_trained_model(args.model)
    threshold = args.threshold if args.threshold is not None else cal_threshold
    
    def get_sentences(text):
        """Simple regex-based sentence splitter."""
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = get_sentences(f.read())
    elif args.text:
        texts = get_sentences(args.text)
    else:
        print("\nEnter text to classify (empty line to quit):")
        texts = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            texts.extend(get_sentences(line))

    print(f"\n{'='*60}  (threshold={threshold:.2f})")
    all_ai_probs = []
    for text in texts:
        result = predict(text, model, tokenizer, cfg, threshold=threshold)
        all_ai_probs.append(result['prob_ai'])
        print(f"\nText    : {text[:80]}{'…' if len(text) > 80 else ''}")
        print(f"Result  : {result['prediction']}  ({result['confidence']}% confident)")
        print(f"Probs   : Human={result['prob_human']}%  AI={result['prob_ai']}%")

    if all_ai_probs:
        avg_ai = sum(all_ai_probs) / len(all_ai_probs)
        overall_pred = "AI" if (avg_ai / 100.0) >= threshold else "Human"
        print(f"\n{'-'*60}")
        print(f"OVERALL ANALYSIS ({len(all_ai_probs)} sentences)")
        print(f"Average AI Probability: {avg_ai:.2f}%")
        print(f"Final Conclusion: {overall_pred}")

    print("="*60)