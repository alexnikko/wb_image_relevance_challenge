from dataclasses import dataclass


@dataclass
class Config:
    image_size: int = 384
    model_name: str = "vit_base_patch16_384"
    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-2
    num_workers: int = 16
    folds: int = 5
    seed: int = 42