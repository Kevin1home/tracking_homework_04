"""DVC-стадия 2: финальная оценка базовой модели на Test_1."""
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ImageDataset, get_test_transform
from src.engine import validate
from src.model import build_model, load_weights


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    tcfg = params["train"]
    dcfg = params["data"]
    pcfg = params["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_base] device = {device}")

    test_ds = ImageDataset(
        csv_file=dcfg["test_1_csv"],
        root_dir=dcfg["test_1_root"],
        transform=get_test_transform(tcfg["image_size"]),
        subsample=tcfg["subsample"],
        seed=tcfg["seed"],
    )
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"])
    print(f"[eval_base] test size = {len(test_ds)}")

    model = build_model(num_classes=2, pretrained=False).to(device)
    model = load_weights(model, pcfg["base_model"], device)
    criterion = nn.CrossEntropyLoss()

    loss, acc, f1, prec, rec = validate(model, test_loader, criterion, device)
    print(f"[eval_base] loss={loss:.4f} acc={acc:.4f} f1={f1:.4f} "
          f"prec={prec:.4f} rec={rec:.4f}")

    # Логирование в TensorBoard под отдельным префиксом test/
    writer = SummaryWriter(log_dir=pcfg["tb_logs_base"])
    writer.add_scalar("test/loss", loss, 0)
    writer.add_scalar("test/accuracy", acc, 0)
    writer.add_scalar("test/f1", f1, 0)
    writer.add_scalar("test/precision", prec, 0)
    writer.add_scalar("test/recall", rec, 0)
    writer.close()

    metrics = {
        "loss": loss, "accuracy": acc, "f1": f1,
        "precision": prec, "recall": rec,
        "dataset": "Test_1", "model": "base",
    }
    Path(pcfg["base_metrics"]).parent.mkdir(parents=True, exist_ok=True)
    with open(pcfg["base_metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_base] метрики -> {pcfg['base_metrics']}")


if __name__ == "__main__":
    main()
