"""Generate text from a trained checkpoint.
Usage: python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
"""
import argparse
import torch

from model import MiniGPT
from tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    c = ckpt["config"]
    tok = CharTokenizer(chars=ckpt["chars"])

    model = MiniGPT(tok.vocab_size, c["n_embd"], c["n_head"], c["n_layer"],
                    c["block_size"], c["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
