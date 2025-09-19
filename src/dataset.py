import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(size: int, is_train: bool = True):
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _load_one_image(img_dir: str, _id: int):
    path = os.path.join(img_dir, f"{_id}.jpg")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _id, img


def _format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def preload_images_to_ram(df: pd.DataFrame, img_dir: str, num_workers: int = None, verbose: bool = True):
    """
    Возвращает словарь {id: np.ndarray(RGB)} для всех уникальных id из df.
    Загрузка распараллелена по потокам (IO-bound), прогресс в tqdm.
    В конце печатает объём памяти, занимаемый кэшем (сумма .nbytes всех изображений).
    """
    unique_ids = df["id"].unique().tolist()
    if num_workers is None:
        # разумный дефолт для IO-bound: 2 * CPU, но не более 32
        cpu = os.cpu_count() or 4
        num_workers = min(32, max(4, 2 * cpu))

    cache = {}
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_load_one_image, img_dir, int(_id)) for _id in unique_ids]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preloading images to RAM"):
            _id, img = fut.result()
            cache[_id] = img
            total_bytes += img.nbytes

    if verbose:
        print(f"[RAM cache] images: {len(cache)} | size: {_format_bytes(total_bytes)} | workers: {num_workers}")

    return cache


def load_ocr_cache(parquet_path: str) -> Dict[int, np.ndarray]:
    """
    Считывает parquet с колонками:
      id, ocr_density, white_ratio, edge_density, aspect
    Возвращает словарь: id -> np.float32[4]
    """
    df = pd.read_parquet(parquet_path)
    feats = {}
    for row in df.itertuples(index=False):
        feats[int(row.id)] = np.array(
            [row.ocr_density, row.white_ratio, row.edge_density, row.aspect],
            dtype=np.float32
        )
    return feats



class ImgOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, size: int, is_train: bool = True,
                 ram_cache=None):   # ram_cache добавлен ранее
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = size
        self.is_train = is_train
        self.tfm = build_transforms(size, is_train)
        self.ram_cache = ram_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        if self.ram_cache is not None:
            img = self.ram_cache[int(row.id)]
        else:
            img_path = f"{self.img_dir}/{row.id}.jpg"
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.tfm(image=img)["image"].astype(np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1)
        y = torch.tensor(float(row.label)) if "label" in self.df.columns else torch.tensor(0.0)
        return x, y, int(row.id)
