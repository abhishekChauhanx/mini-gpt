"""Train MiniGPT from scratch. Usage: python train.py"""
import math
import time
import torch

from config import Config
from model import MiniGPT
from tokenizer import CharTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, cfg):
    """Linear warmup then cosine decay."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main(cfg=None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}")

    text = open(cfg.data_path, encoding="utf-8").read()
    tok = CharTokenizer(text=text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(cfg.train_split * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"chars: {len(text):,} | vocab: {tok.vocab_size}")

    model = MiniGPT(tok.vocab_size, cfg.n_embd, cfg.n_head, cfg.n_layer,
                    cfg.block_size, cfg.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    best_val = float("inf")
    t0 = time.time()

    for it in range(cfg.max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, cfg)

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"step {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "chars": tok.chars,
                    "config": cfg.__dict__,
                }, cfg.checkpoint_path)

        if it == cfg.max_iters:
            break

        xb, yb = get_batch(splits["train"], cfg, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    print(f"done. best val loss: {best_val:.4f} | saved to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
