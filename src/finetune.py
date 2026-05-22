"""DVC-стадия 3: скачать базовую модель из S3, дообучить на Train_2,
выложить новую версию в S3."""
import json
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
from src.model import build_model, load_weights
from src.s3_utils import download_file, get_s3_client, upload_file


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    fcfg = params["finetune"]
    dcfg = params["data"]
    pcfg = params["paths"]
    s3cfg = params["s3"]

    set_seed(fcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device = {device}")

    # 1) Скачиваем базовую модель из S3 (если не получится — берём локальную копию)
    Path(pcfg["base_model"]).parent.mkdir(parents=True, exist_ok=True)
    try:
        client = get_s3_client(
            s3cfg["endpoint_url"], s3cfg["access_key"], s3cfg["secret_key"], s3cfg["region"]
        )
        download_file(client, s3cfg["bucket"], s3cfg["base_model_key"], pcfg["base_model"])
        print(f"[finetune] скачали из s3://{s3cfg['bucket']}/{s3cfg['base_model_key']}")
    except Exception as e:
        print(f"[finetune] WARNING: не удалось скачать из S3 ({e}), беру локальный файл")

    # 2) Готовим данные Train_2 / Test_2
    train_ds = ImageDataset(
        csv_file=dcfg["train_2_csv"],
        root_dir=dcfg["train_2_root"],
        transform=get_train_transform(fcfg["image_size"]),
        subsample=fcfg["subsample"],
        seed=fcfg["seed"],
    )
    test_ds = ImageDataset(
        csv_file=dcfg["test_2_csv"],
        root_dir=dcfg["test_2_root"],
        transform=get_test_transform(fcfg["image_size"]),
        subsample=fcfg["subsample"],
        seed=fcfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=fcfg["batch_size"], shuffle=True,
                              num_workers=fcfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=fcfg["batch_size"], shuffle=False,
                             num_workers=fcfg["num_workers"])
    print(f"[finetune] train_2 size = {len(train_ds)}, test_2 size = {len(test_ds)}")

    # 3) Грузим веса базовой модели
    model = build_model(num_classes=2, pretrained=False).to(device)
    model = load_weights(model, pcfg["base_model"], device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=fcfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=fcfg["step_size"], gamma=fcfg["gamma"])

    Path(pcfg["tb_logs_finetune"]).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=pcfg["tb_logs_finetune"])
    writer.add_text("params/finetune", json.dumps(fcfg, indent=2))

    history = []
    for epoch in range(fcfg["epochs"]):
        print(f"\n[finetune] Epoch {epoch + 1}/{fcfg['epochs']}")
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_acc, val_f1, val_prec, val_rec = validate(model, test_loader, criterion, device)

        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/accuracy", tr_acc, epoch)
        writer.add_scalar("train/f1", tr_f1, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/f1", val_f1, epoch)
        writer.add_scalar("val/precision", val_prec, epoch)
        writer.add_scalar("val/recall", val_rec, epoch)

        print(f"  train  loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
        print(f"  val    loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} "
              f"prec={val_prec:.4f} rec={val_rec:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "val_precision": val_prec, "val_recall": val_rec,
        })

    writer.close()

    # 4) Сохраняем и выкладываем новую версию
    Path(pcfg["finetuned_model"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pcfg["finetuned_model"])
    print(f"[finetune] модель сохранена: {pcfg['finetuned_model']}")

    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/finetune_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    try:
        client = get_s3_client(
            s3cfg["endpoint_url"], s3cfg["access_key"], s3cfg["secret_key"], s3cfg["region"]
        )
        upload_file(client, pcfg["finetuned_model"], s3cfg["bucket"], s3cfg["finetuned_model_key"])
        print(f"[finetune] выгружено в s3://{s3cfg['bucket']}/{s3cfg['finetuned_model_key']}")
    except Exception as e:
        print(f"[finetune] WARNING: не удалось выгрузить в S3: {e}")


if __name__ == "__main__":
    main()
