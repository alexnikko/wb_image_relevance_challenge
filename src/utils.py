import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def make_folds(train_csv: str, n_splits: int = 5):
    df = pd.read_csv(train_csv)
    assert "card_identifier_id" in df.columns, "train.csv must contain card_identifier_id"
    assert "label" in df.columns, "train.csv must contain label"
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["fold"] = -1
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df["card_identifier_id"])):
        df.loc[val_idx, "fold"] = fold
    out_path = train_csv.replace(".csv", f"_folds{n_splits}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved folds to {out_path}")
    return out_path




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)