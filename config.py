"""All hyperparameters in one place. Edit these to scale the model up or down."""
from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_path: str = "data/input.txt"
    train_split: float = 0.9

    # model
    block_size: int = 256     # context length (characters the model sees at once)
    n_embd: int = 384         # embedding size
    n_head: int = 6           # attention heads
    n_layer: int = 6          # transformer blocks
    dropout: float = 0.2

    # training
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 100

    # io
    checkpoint_path: str = "checkpoints/minigpt.pt"
    seed: int = 1337
