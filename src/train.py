import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC
import timm, cv2, albumentations as A
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from torch.optim.swa_utils import AveragedModel  # EMA

from .utils import seed_everything, ensure_dir
from .config import Config
from .dataset import preload_images_to_ram
from .optimization import step_optimizer

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)


def build_tfm(size, train=True):
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class FusionDataset(Dataset):
    """
    Возвращает (x_img, toks, y, id, card_id)
    """
    def __init__(self, df, img_dir, size, tokenizer, max_len=256, is_train=True, ram_cache=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = build_tfm(size, is_train)
        self.is_train = is_train
        self.ram_cache = ram_cache
        self.tok = tokenizer
        self.max_len = max_len
        assert "card_identifier_id" in self.df.columns, "Need column card_identifier_id"

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # image
        if self.ram_cache is not None:
            img = self.ram_cache[int(row.id)]
        else:
            p = os.path.join(self.img_dir, f"{row.id}.jpg")
            img = cv2.imread(p)
            if img is None: raise FileNotFoundError(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(image=img)["image"].astype(np.float32)
        x_img = torch.from_numpy(img).permute(2,0,1)

        text = f"{row.title} [SEP] {row.description}"
        toks = self.tok(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        toks = {k: v.squeeze(0) for k, v in toks.items()}

        y = torch.tensor(float(row.label)) if "label" in self.df.columns else torch.tensor(0.0)
        return x_img, toks, y, int(row.id), int(row.card_identifier_id)


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class FusionModel(nn.Module):
    def __init__(self, img_model="vit_base_patch16_384", txt_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__()
        self.img_enc = timm.create_model(img_model, pretrained=True, num_classes=0)
        self.txt_tok = AutoTokenizer.from_pretrained(txt_model)
        self.txt_enc = AutoModel.from_pretrained(txt_model)
        self.head = None
        self.gate = None  # добавим после init_head

    @torch.no_grad()
    def init_head(self, device, image_size=384, max_len=256):
        self.eval()
        # image dummy
        dummy_img = torch.zeros(1,3,image_size,image_size, device=device)
        img_emb = self.img_enc(dummy_img)
        d_img = img_emb.shape[-1] if img_emb.ndim==2 else img_emb.view(1,-1).shape[-1]
        # text dummy
        toks = self.txt_tok("dummy", return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        txt_out = self.txt_enc(**toks)
        if hasattr(txt_out, "last_hidden_state"):
            txt_emb = mean_pool(txt_out.last_hidden_state, toks["attention_mask"])
        else:
            txt_emb = txt_out.pooler_output
        d_txt = txt_emb.shape[-1]

        # gating + head
        self.gate = nn.Sequential(nn.Linear(d_txt, d_img), nn.Sigmoid()).to(device)
        self.head = nn.Sequential(
            nn.Linear(d_img + d_txt, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        ).to(device)
        self.train()

    def forward(self, x_img, toks=None, txt_emb_cached=None):
        img_emb = self.img_enc(x_img)  # [B, d_img]
        if txt_emb_cached is None:
            txt_out = self.txt_enc(**toks)
            if hasattr(txt_out, "last_hidden_state"):
                txt_emb = mean_pool(txt_out.last_hidden_state, toks["attention_mask"])
            else:
                txt_emb = txt_out.pooler_output
        else:
            txt_emb = txt_emb_cached  # [B, d_txt]
        g = self.gate(txt_emb)             # [B, d_img]
        img_mod = img_emb * (1.0 + g)      # FiLM-модуляция
        x = torch.cat([img_mod, txt_emb], dim=1)
        return self.head(x).squeeze(1)


def pairwise_bpr_loss(logits: torch.Tensor, labels: torch.Tensor, card_ids: torch.Tensor) -> torch.Tensor:
    loss = logits.new_tensor(0.0); valid = 0
    for cid in torch.unique(card_ids):
        m = (card_ids == cid)
        l = logits[m]; y = labels[m]
        pos = l[y == 1]; neg = l[y == 0]
        if pos.numel() and neg.numel():
            diff = neg[:, None] - pos[None, :]
            loss = loss + F.softplus(diff).mean()
            valid += 1
    return loss / valid if valid > 0 else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--image_size", type=int, default=Config.image_size)
    ap.add_argument("--folds", type=int, default=Config.folds)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=3e-5)         # после разморозки
    ap.add_argument("--freeze_epochs", type=int, default=2)      # сначала учим только head
    ap.add_argument("--txt_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--img_model", type=str, default="vit_base_patch16_384")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--preload_ram", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_checkpoint", action="store_true")

    # Новые крутые флаги:
    ap.add_argument("--pair_lambda", type=float, default=0.01, help="Вес BPR pairwise-loss")
    ap.add_argument("--pair_warmup_epochs", type=int, default=None,
                    help="Сколько эпох греть вес pairwise (по умолчанию freeze_epochs+1)")
    ap.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay (0.99 – хороший старт)")
    ap.add_argument("--clip_norm", type=float, default=1.0, help="Max gradient norm")
    ap.add_argument("--cache_text", action="store_true", help="Кешировать текстовые эмбеддинги по карточке, когда текст заморожен")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    print("Device:", device)

    df = pd.read_csv(args.train_csv)
    assert "fold" in df.columns and "card_identifier_id" in df.columns

    for fold in range(args.folds):
        print(f"=== Fusion fold {fold} ===")
        trn_df = df[df.fold != fold].copy()
        val_df = df[df.fold == fold].copy()

        # RAM cache для ускорения IO
        ram_cache = preload_images_to_ram(pd.concat([trn_df, val_df]), args.img_dir) if args.preload_ram else None

        model = FusionModel(img_model=args.img_model, txt_model=args.txt_model).to(device)
        if args.grad_checkpoint and hasattr(model.img_enc, "set_grad_checkpointing"):
            model.img_enc.set_grad_checkpointing(True)
        model.init_head(device=device, image_size=args.image_size, max_len=args.max_len)

        tokenizer = model.txt_tok

        # фризы: сначала учим только head
        for p in model.img_enc.parameters(): p.requires_grad = False
        for p in model.txt_enc.parameters(): p.requires_grad = False
        for p in model.head.parameters():    p.requires_grad = True

        # датасеты/лоадеры
        train_ds = FusionDataset(trn_df, args.img_dir, args.image_size, tokenizer, args.max_len, True, ram_cache)
        val_ds   = FusionDataset(val_df,  args.img_dir, args.image_size, tokenizer, args.max_len, False, ram_cache)
        nw = 0 if args.preload_ram else 8
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=nw, pin_memory=True)

        loss_fn = nn.BCEWithLogitsLoss()
        metric  = BinaryAUROC()

        # оптимайзер (фаза HEAD)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr_head, weight_decay=1e-2)

        # EMA
        ema = AveragedModel(model, avg_fn=lambda avg, cur, _: args.ema_decay*avg + (1-args.ema_decay)*cur)

        use_amp = (device != "cpu")
        scaler  = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        best_auc = -1.0
        global_step = 0

        def build_card_text_cache(local_df):
            card_cache = {}
            model.txt_enc.eval()
            with torch.inference_mode():
                for cid, g in local_df.groupby("card_identifier_id"):
                    text = f"{g.iloc[0].title} [SEP] {g.iloc[0].description}"
                    toks = tokenizer(text, padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
                    toks = {k: v.to(device) for k, v in toks.items()}
                    out = model.txt_enc(**toks)
                    if hasattr(out, "last_hidden_state"):
                        emb = mean_pool(out.last_hidden_state, toks["attention_mask"])  # [1, d_txt]
                    else:
                        emb = out.pooler_output
                    card_cache[int(cid)] = emb.detach()   # [1, d_txt] на device
            model.txt_enc.train()
            return card_cache

        card_txt_cache_train = build_card_text_cache(trn_df) if args.cache_text else None
        card_txt_cache_val   = build_card_text_cache(val_df) if args.cache_text else None

        pair_warmup = args.pair_warmup_epochs if args.pair_warmup_epochs is not None else (args.freeze_epochs + 1)

        for epoch in range(args.epochs):
            model.train()
            trn_loss = 0.0
            pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch+1}/{args.epochs} [train]")

            if epoch == args.freeze_epochs:
                for p in model.img_enc.parameters(): p.requires_grad = True
                for p in model.txt_enc.parameters(): p.requires_grad = True
                opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr_ft, weight_decay=1e-2)
                card_txt_cache_train = None
                card_txt_cache_val = None

            opt.zero_grad(set_to_none=True)
            for x_img, toks, y, ids, card_ids in pbar:
                x_img = x_img.to(device, non_blocking=True)
                y     = y.to(device, non_blocking=True)
                card_ids = torch.as_tensor(card_ids, dtype=torch.long, device=device)

                txt_cached = None
                toks_dev = None
                if card_txt_cache_train is not None:
                    # соберём батч txt-эмбеддингов по card_id
                    embs = [card_txt_cache_train[int(cid)].squeeze(0) for cid in card_ids.tolist()]
                    txt_cached = torch.stack(embs, dim=0)  # [B, d_txt]
                else:
                    toks_dev = {k: v.to(device, non_blocking=True) for k, v in toks.items()}

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    logits = model(x_img, toks=toks_dev, txt_emb_cached=txt_cached)
                    bce = loss_fn(logits, y)
                    pair_w = args.pair_lambda * min(1.0, (epoch+1) / max(1, pair_warmup))
                    pw = pairwise_bpr_loss(logits, (y > 0.5).float(), card_ids)
                    loss = bce + pair_w * pw

                if device == "cuda":
                    scaler.scale(loss / args.grad_accum).backward()
                else:
                    (loss / args.grad_accum).backward()

                if (global_step + 1) % args.grad_accum == 0:
                    step_optimizer(
                        (p for p in model.parameters() if p.requires_grad),
                        opt,
                        max_norm=args.clip_norm,
                        scaler=scaler if device == "cuda" else None,
                    )
                    opt.zero_grad(set_to_none=True)
                    # EMA обновление
                    ema.update_parameters(model)

                global_step += 1
                trn_loss += loss.item() * x_img.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            trn_loss /= len(train_ds)

            # validation (по EMA-модели для стабильности)
            model.eval()
            ema.module.eval()
            v_logits=[]; v_labels=[]
            use_amp_val = (device != "cpu")
            with torch.inference_mode():
                pbar_v = tqdm(val_loader, desc=f"fold {fold} epoch {epoch+1}/{args.epochs} [val]")
                for x_img, toks, y, ids, card_ids in pbar_v:
                    x_img = x_img.to(device, non_blocking=True)
                    y     = y.to(device, non_blocking=True)

                    # кэш текста на валидации, если есть
                    if card_txt_cache_val is not None:
                        embs = [card_txt_cache_val[int(cid)].squeeze(0) for cid in card_ids.tolist()]
                        txt_cached = torch.stack(embs, dim=0).to(device, non_blocking=True)
                        toks_dev = None
                    else:
                        txt_cached = None
                        toks_dev = {k: v.to(device, non_blocking=True) for k, v in toks.items()}

                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp_val):
                        logit = ema.module(x_img, toks=toks_dev, txt_emb_cached=txt_cached)
                    v_logits.append(torch.sigmoid(logit).cpu())
                    v_labels.append(y.cpu())
            v_logits = torch.cat(v_logits); v_labels = torch.cat(v_labels).int()
            auc = metric(v_logits, v_labels).item()
            print(f"[fold {fold}] epoch {epoch+1}: train_loss={trn_loss:.4f} | val_auc(EMA)={auc:.4f}")

            # save best
            if auc > best_auc:
                best_auc = auc
                ensure_dir(os.path.join(args.out_dir, "folds"))
                ckpt = {"model": ema.module.state_dict(), "auc": best_auc}
                torch.save(ckpt, os.path.join(args.out_dir, "folds", f"fusion_fold{fold}.pt"))
                print("  -> saved best (EMA)")

if __name__ == "__main__":
    main()
