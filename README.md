# MiniGPT: a language model built from scratch

A small GPT-style transformer trained from random initialization (no pretrained weights).
Default setup: character-level model (~10.7M params) trained on Tiny Shakespeare.

## Files
| File | Purpose |
|---|---|
| `config.py` | All hyperparameters |
| `tokenizer.py` | Character-level encode/decode |
| `model.py` | Transformer architecture |
| `train.py` | Training loop (warmup + cosine LR, AdamW, checkpoints) |
| `generate.py` | Sample text from a checkpoint |
| `data/input.txt` | Training text (swap in your own) |

## Run on Google Colab (recommended)
1. Open colab.research.google.com -> Runtime -> Change runtime type -> **T4 GPU**
2. Upload the `mini_gpt` folder (or zip it and unzip in Colab)
3. In a cell:
   ```
   %cd mini_gpt
   !pip install -q torch matplotlib
   !python train.py
   !python generate.py --prompt "ROMEO:" --tokens 600
   ```
Training takes roughly 10-20 min on a T4. Val loss should fall from ~4.2 to ~1.5.

## Run locally
```
pip install -r requirements.txt
python train.py
python generate.py --prompt "ROMEO:"
```
On CPU, shrink the model first in `config.py`
(e.g. `n_layer=4, n_embd=128, block_size=128, batch_size=32, max_iters=2000`).

## Use your own text
Replace `data/input.txt` with any plain-text file (1 MB+ works best), then retrain.

## Troubleshooting
- Out of memory: lower `batch_size` or `block_size`
- Gibberish: normal early on; check val loss is dropping, train longer
- Val loss rising while train loss falls: overfitting; add data or raise `dropout`
