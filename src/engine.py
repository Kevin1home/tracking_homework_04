"""Тренировочные / валидационные циклы + считаем метрики."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / max(n_samples, 1)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return epoch_loss, float(epoch_acc), float(epoch_f1)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / max(n_samples, 1)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    return (
        epoch_loss,
        float(epoch_acc),
        float(epoch_f1),
        float(epoch_precision),
        float(epoch_recall),
    )
