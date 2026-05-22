"""DVC-стадия 1: обучить базовую модель на Train_1, выгрузить в S3."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ImageDataset, get_train_transform, get_test_transform
from src.engine import train_epoch, validate
from src.model import build_model
from src.s3_utils import get_s3_client, upload_file


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    tcfg = params["train"]
    dcfg = params["data"]
    pcfg = params["paths"]
    s3cfg = params["s3"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    train_ds = ImageDataset(
        csv_file=dcfg["train_1_csv"],
        root_dir=dcfg["train_1_root"],
        transform=get_train_transform(tcfg["image_size"]),
        subsample=tcfg["subsample"],
        seed=tcfg["seed"],
    )
    test_ds = ImageDataset(
        csv_file=dcfg["test_1_csv"],
        root_dir=dcfg["test_1_root"],
        transform=get_test_transform(tcfg["image_size"]),
        subsample=tcfg["subsample"],
        seed=tcfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"])
    print(f"[train] train size = {len(train_ds)}, test size = {len(test_ds)}")

    model = build_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tcfg["step_size"], gamma=tcfg["gamma"])

    Path(pcfg["tb_logs_base"]).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=pcfg["tb_logs_base"])

    # Логируем параметры обучения ДО старта
    writer.add_text("params/train", json.dumps(tcfg, indent=2))
    writer.add_text("params/data", json.dumps(dcfg, indent=2))

    train_history = []
    for epoch in range(tcfg["epochs"]):
        print(f"\n[train] Epoch {epoch + 1}/{tcfg['epochs']}")
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        # Промежуточная валидация на конце эпохи -> в TensorBoard
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(
            model, test_loader, criterion, device
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/f1", train_f1, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/f1", val_f1, epoch)
        writer.add_scalar("val/precision", val_prec, epoch)
        writer.add_scalar("val/recall", val_rec, epoch)

        print(f"  train  loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")
        print(f"  val    loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} "
              f"prec={val_prec:.4f} rec={val_rec:.4f}")
        train_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "val_precision": val_prec, "val_recall": val_rec,
        })

    writer.close()

    # Сохраняем модель локально
    Path(pcfg["base_model"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pcfg["base_model"])
    print(f"[train] модель сохранена: {pcfg['base_model']}")

    # Сохраняем историю обучения для DVC metrics
    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/train_history.json", "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)

    # Выгружаем в S3 (MinIO)
    try:
        client = get_s3_client(
            s3cfg["endpoint_url"], s3cfg["access_key"], s3cfg["secret_key"], s3cfg["region"]
        )
        upload_file(client, pcfg["base_model"], s3cfg["bucket"], s3cfg["base_model_key"])
        print(f"[train] выгружено в s3://{s3cfg['bucket']}/{s3cfg['base_model_key']}")
    except Exception as e:
        print(f"[train] WARNING: не удалось выгрузить в S3: {e}")


if __name__ == "__main__":
    main()
