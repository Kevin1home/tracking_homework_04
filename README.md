# ДЗ MLOPS — tracking_homework_04

Классификация AI-generated vs Human-generated изображений (ResNet18, transfer learning)
с интеграцией **TensorBoard**, **MinIO (S3)** и **DVC-пайплайном**.

## Структура
```
.
├── docker-compose.yml          # MinIO (S3) локально
├── params.yaml                 # все гиперпараметры обучения и пути
├── dvc.yaml                    # описание DVC-пайплайна (4 стадии)
├── requirements.txt
├── train_model.ipynb           # итоговый ноутбук со всеми шагами + выводом
├── src/
│   ├── dataset.py              # ImageDataset + transforms
│   ├── model.py                # build_model / load_weights (ResNet18)
│   ├── engine.py               # train_epoch / validate + метрики
│   ├── s3_utils.py             # клиент boto3 для MinIO
│   ├── train.py                # стадия DVC: обучение базовой модели
│   ├── evaluate_base.py        # стадия DVC: оценка базовой модели на Test_1
│   ├── finetune.py             # стадия DVC: дообучение на Train_2
│   └── evaluate_finetuned.py   # стадия DVC: оценка дообученной модели на Test_2
└── ai-vs-human-generated-dataset-hw/   # датасет (в .gitignore)
```

## Быстрый запуск (Git Bash на Windows)
```bash
# 1. виртуальное окружение и зависимости
python -m venv .venv
source .venv/Scripts/activate          # на WSL/Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. MinIO (нужен Docker Desktop)
docker compose up -d
# открыть http://localhost:9001 (minioadmin / minioadmin) -> создать bucket "models"

# 3. TensorBoard в отдельном терминале
tensorboard --logdir=logs

# 4. DVC пайплайн (обучение → оценка → дообучение → оценка)
dvc init -f
dvc repro
dvc dag
dvc metrics show
```

## Стадии DVC-пайплайна
| Стадия | Скрипт | Вход | Выход |
|---|---|---|---|
| `train` | `src/train.py` | `Train_1`, `Test_1` | `models/resnet18_base.pth`, `logs/base/`, выгрузка в S3 |
| `evaluate_base` | `src/evaluate_base.py` | `Test_1`, базовая модель | `metrics/base_metrics.json` |
| `finetune` | `src/finetune.py` | базовая модель из S3, `Train_2`, `Test_2` | `models/resnet18_finetuned.pth`, `logs/finetune/`, выгрузка в S3 |
| `evaluate_finetuned` | `src/evaluate_finetuned.py` | `Test_2`, дообученная модель | `metrics/finetune_metrics.json` |

## Метрики в TensorBoard
В каждой стадии логируется:
- `train/loss`, `train/accuracy`, `train/f1`
- `val/loss`, `val/accuracy`, `val/f1`, `val/precision`, `val/recall`
- `test/*` — финальная оценка по тестовому датасету
- `params/train`, `params/data`, `params/finetune` — текстовый снимок гиперпараметров
