"""
Trening modelu CNN+LSTM na segmentach wideo z data_video/<klasa>/.
Uruchomienie: python -m danger_detection.train
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from .config import (
    DATA_VIDEO_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_CHECKPOINT,
)
from .dataset import DangerVideoDataset
from .model_cnn_lstm import CNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Urządzenie: {DEVICE}")
    print(f"Dane: {DATA_VIDEO_DIR}")

    dataset = DangerVideoDataset(root=DATA_VIDEO_DIR)
    print(f"Klasy: {dataset.class_names}")
    print(f"Liczba segmentów: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = CNNLSTM(num_classes=dataset.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total if total else 0.0
        avg_loss = total_loss / len(loader) if loader else 0.0
        print(f"Epoka {epoch}/{EPOCHS}  loss: {avg_loss:.4f}  dokładność: {acc:.2f}%")

    Path(MODEL_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": dataset.class_names,
            "num_classes": dataset.num_classes,
        },
        MODEL_CHECKPOINT,
    )
    print(f"\nModel zapisany: {MODEL_CHECKPOINT}")


if __name__ == "__main__":
    main()
