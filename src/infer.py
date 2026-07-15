import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .utils import ensure_dir
from .train import FusionModel, FusionDataset, pick_device, mean_pool
from .dataset import preload_images_to_ram

def build_card_text_cache(model: FusionModel, df: pd.DataFrame, tokenizer, device: str, max_len: int):
    cache = {}
    was_training = model.txt_enc.training
    model.txt_enc.eval()
    with torch.inference_mode():
        for cid, g in df.groupby("card_identifier_id"):
            text = f"{g.iloc[0].title} [SEP] {g.iloc[0].description}"
            toks = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model.txt_enc(**toks)
            if hasattr(out, "last_hidden_state"):
                emb = mean_pool(out.last_hidden_state, toks["attention_mask"])  # [1,d]
            else:
                emb = out.pooler_output
            cache[int(cid)] = emb.detach()
    model.txt_enc.train(was_training)
    return cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--out_file", type=str, default="submission_fusion.csv",
                    help="Имя файла сабмита внутри out_dir/sub")
    ap.add_argument("--image_size", type=int, default=Config.image_size)
    ap.add_argument("--folds", type=int, default=Config.folds)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--txt_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--img_model", type=str, default="vit_base_patch16_384")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--preload_ram", action="store_true")
    ap.add_argument("--cache_text", action="store_true",
                    help="Кешировать txt-эмбеддинги по card_identifier_id (рекомендую включить)")
    args = ap.parse_args()

    device = pick_device()
    use_amp = (device != "cpu")
    print("Inference device:", device)

    df = pd.read_csv(args.test_csv)
    # модель и голова
    model = FusionModel(img_model=args.img_model, txt_model=args.txt_model).to(device)
    model.init_head(device=device, image_size=args.image_size, max_len=args.max_len)
    tokenizer = model.txt_tok

    # датасет / лоадер
    ram_cache = preload_images_to_ram(df, args.img_dir) if args.preload_ram else None
    ds = FusionDataset(
        df,
        args.img_dir,
        args.image_size,
        tokenizer,
        args.max_len,
        is_train=False,
        ram_cache=ram_cache,
    )
    nw = 8 if not args.preload_ram else 0
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=nw,
                        pin_memory=True, persistent_workers=(nw > 0))

    fold_probs = []
    for f in range(args.folds):
        ckpt = os.path.join(args.out_dir, "folds", f"fusion_fold{f}.pt")
        assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)  # это EMA-веса из тренера
        model.eval()

        # Text embeddings must match the encoder weights of the current fold.
        card_txt_cache = (
            build_card_text_cache(model, df, tokenizer, device, args.max_len)
            if args.cache_text
            else None
        )

        preds = []
        with torch.inference_mode():
            pbar = tqdm(loader, desc=f"Infer fold {f}")
            for x_img, toks, _, ids, card_ids in pbar:
                x_img = x_img.to(device, non_blocking=True)

                if card_txt_cache is not None:
                    embs = [card_txt_cache[int(cid)].squeeze(0) for cid in card_ids.tolist()]
                    txt_cached = torch.stack(embs, dim=0).to(device, non_blocking=True)
                    toks_dev = None
                else:
                    toks_dev = {k: v.to(device, non_blocking=True) for k, v in toks.items()}
                    txt_cached = None

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    logit = model(x_img, toks=toks_dev, txt_emb_cached=txt_cached)
                    p = torch.sigmoid(logit).detach().cpu().numpy()
                preds.append(p)

        fold_probs.append(np.concatenate(preds))

    probs = np.mean(np.stack(fold_probs, axis=0), axis=0)

    ensure_dir(os.path.join(args.out_dir, "sub"))
    sub = pd.DataFrame({"id": df["id"], "y_pred": probs})
    out_path = os.path.join(args.out_dir, "sub", args.out_file)
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")

if __name__ == "__main__":
    main()
