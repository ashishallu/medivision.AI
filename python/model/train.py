"""
train.py
Fine-tunes ChestXrayResNet50 on the NIH ChestX-ray14 dataset.

Expected dataset layout (download separately -- see README):
    data/chestxray14/
        images/                *.png
        Data_Entry_2017.csv    (official NIH label file)

Usage:
    python -m python.model.train
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from python.model.classifier import ChestXrayResNet50
from python.utils.preprocess import get_training_transform, get_inference_transform
from python.utils.config_loader import load_config, resolve_path

_cfg = load_config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChestXray14Dataset(Dataset):
    """Wraps the NIH ChestX-ray14 CSV + image folder into a multi-label
    PyTorch dataset."""

    def __init__(self, dataframe: pd.DataFrame, image_dir: str, class_names, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(image_path)

        labels_str = row["Finding Labels"]  # e.g. "Cardiomegaly|Effusion"
        label_vector = np.zeros(len(self.class_names), dtype=np.float32)
        if labels_str != "No Finding":
            for finding in labels_str.split("|"):
                if finding in self.class_names:
                    label_vector[self.class_names.index(finding)] = 1.0

        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label_vector)


def build_dataloaders():
    class_names = _cfg["model"]["class_names"]
    data_dir = resolve_path("data/chestxray14")
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    image_dir = os.path.join(data_dir, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}.\n"
            "Download the NIH ChestX-ray14 dataset (free, public) and place "
            "it at data/chestxray14/ before training. See README.md for the link."
        )

    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(
        df, test_size=_cfg["training"]["val_split"] + _cfg["training"]["test_split"],
        random_state=_cfg["training"]["seed"],
    )
    val_frac = _cfg["training"]["val_split"] / (
        _cfg["training"]["val_split"] + _cfg["training"]["test_split"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_frac, random_state=_cfg["training"]["seed"]
    )

    train_ds = ChestXray14Dataset(train_df, image_dir, class_names, get_training_transform())
    val_ds = ChestXray14Dataset(val_df, image_dir, class_names, get_inference_transform())
    test_ds = ChestXray14Dataset(test_df, image_dir, class_names, get_inference_transform())

    batch_size = _cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device, class_names):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aucs = {}
    for i, name in enumerate(class_names):
        if len(np.unique(all_labels[:, i])) > 1:  # AUC undefined with only one class present
            aucs[name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
    return aucs


def main():
    set_seed(_cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    class_names = _cfg["model"]["class_names"]

    model = ChestXrayResNet50().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_cfg["training"]["learning_rate"])

    best_val_auc = 0.0
    weights_path = resolve_path(_cfg["paths"]["trained_weights_file"])
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    for epoch in range(_cfg["training"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_aucs = evaluate(model, val_loader, device, class_names)
        mean_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0.0
        print(f"[train] Epoch {epoch+1}: loss={avg_loss:.4f} mean_val_auc={mean_val_auc:.4f}")

        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            torch.save(model.state_dict(), weights_path)
            print(f"[train] Saved new best model (val_auc={mean_val_auc:.4f}) to {weights_path}")

    test_aucs = evaluate(model, test_loader, device, class_names)
    print("[train] Final test AUCs per class:")
    for name, auc in test_aucs.items():
        print(f"  {name}: {auc:.4f}")


if __name__ == "__main__":
    main()
