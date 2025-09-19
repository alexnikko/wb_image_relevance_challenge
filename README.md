# Описание решения

Кросс-модальная модель: изображение (timm) + текст (HF Transformers) с FiLM-модуляцией и EMA.  
Тренировка по K-fold, инференс по фолдам с усреднением.

## Содержание

- [Установка](#установка)
  - [Вариант A: как пакет](#вариант-a-как-пакет)
  - [Вариант B: по requirements.txt](#вариант-b-по-requirementstxt)
- [Данные](#данные)
- [Разбиение на фолды](#разбиение-на-фолды)
- [Тренировка](#тренировка)
- [Инференс](#инференс)
- [Требования к окружению](#требования-к-окружению)

## Установка

### Вариант A: как пакет

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip

pip install -e .
````

Появятся консольные команды:

* `make-folds`
* `fusion-train`
* `fusion-infer`

### Вариант B: по requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Запуск модулей: `python -m src.train`, `python -m src.infer`, `python -m src.make_folds`

## Данные

Ожидается структура:

```
image_relevance_challenge/
├─ data/
│  ├─ train.csv             # id, title, description, card_identifier_id, label
│  ├─ test.csv              # id, title, description, card_identifier_id
│  └─ images/
│     ├─ 1.jpg
│     ├─ 2.jpg
│     └─ ...
└─ outputs/                 # чекпойнты и сабмиты будут здесь
```

**Обязательные колонки:**

* `id` — int, имя файла изображения это `{id}.jpg`
* `title`, `description` — текст
* `card_identifier_id` — группировка объектов (для BPR и разбиения по группам)
* `label` — 0/1 (только для train)

## Разбиение на фолды

`make-folds` гарантирует групповой сплит по `card_identifier_id`:

```bash
make-folds --train_csv data/train.csv --n_splits 5
# создаст data/train_folds5.csv с колонкой fold
```

(Аналог: `python -m src.make_folds --train_csv data/train.csv --n_splits 5`)

## Тренировка

```bash
fusion-train \
  --train_csv data/train_folds5.csv \
  --img_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
  --img_dir data/images \
  --out_dir outputs/fusion_v3_bigger_img \
  --image_size 448 \
  --epochs 8 \
  --batch_size 24 \
  --freeze_epochs 2 \
  --lr_head 1e-3 \
  --lr_ft 1e-5 \
  --txt_model deepvk/USER-bge-m3 \
  --max_len 256 \
  --pair_lambda 0.01 \
  --pair_warmup_epochs 3 \
  --clip_norm 1.0 \
  --cache_text \
  --grad_accum 1 \
  --grad_checkpoint
```

Эквивалент через модуль:

```bash
python -m src.train \
  --train_csv data/train_folds5.csv \
  --img_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
  --img_dir data/images \
  --out_dir outputs/fusion_v3_bigger_img \
  --image_size 448 \
  --epochs 8 \
  --batch_size 24 \
  --freeze_epochs 2 \
  --lr_head 1e-3 \
  --lr_ft 1e-5 \
  --txt_model deepvk/USER-bge-m3 \
  --max_len 256 \
  --pair_lambda 0.01 \
  --pair_warmup_epochs 3 \
  --clip_norm 1.0 \
  --cache_text \
  --grad_accum 1 \
  --grad_checkpoint
```

**Заметки:**

* `--cache_text` ускоряет этапы, где текстовый энкодер заморожен/фиксирован.
* `--grad_checkpoint` включится только если выбранная `timm`-модель это поддерживает.
* EMA включена по умолчанию через `AveragedModel`; сохраняются EMA-веса с наилучшим AUC.

## Инференс

```bash
fusion-infer \
  --test_csv data/test.csv \
  --img_dir data/images \
  --out_dir outputs/fusion_v3_bigger_img \
  --out_file fusion_eva_qwen.csv \
  --img_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
  --txt_model deepvk/USER-bge-m3 \
  --image_size 448 \
  --batch_size 64 \
  --cache_text
```

(Или `python -m src.infer` с теми же аргументами.)

Инференс ожидает чекпойнты по путям:

```
outputs/fusion_v3_bigger_img/folds/fusion_fold{0..F-1}.pt
```

где `F` — число фолдов (`--folds` в `config.py` или через аргумент).

## Требования к окружению

* Python ≥ 3.10
* CUDA (опционально). Скрипты сами выберут `cuda`/`mps`/`cpu`.
* Некоторые модели из timm (например, EVA-02) требуют достаточно свежую версию `timm`.