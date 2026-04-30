import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import time

from .config import (
    DATA_VIDEO_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_CHECKPOINT,
)
from .dataset import DangerFeatureDataset, DangerVideoDataset
from .model_cnn_lstm import CNNLSTM, LSTMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Urządzenie: {DEVICE}")
    print(f"Dane: {DATA_VIDEO_DIR}")
    log_every = int(os.getenv("LOG_EVERY", "50"))
    save_each_epoch = os.getenv("SAVE_EACH_EPOCH", "0").strip() in {"1", "true", "True", "yes", "YES"}

    cache_dir = os.getenv("FEATURE_CACHE_DIR", "").strip()
    use_cache = bool(cache_dir)
    if use_cache:
        dataset = DangerFeatureDataset(cache_dir)
        print(f"Tryb: FEATURE CACHE ({cache_dir})")
        if dataset.class_names:
            print(f"Klasy: {dataset.class_names}")
        print(f"Liczba segmentów (cache): {len(dataset)}")
        model = LSTMClassifier(num_classes=len(dataset.class_names)).to(DEVICE)
    else:
        dataset = DangerVideoDataset(root=DATA_VIDEO_DIR)
        print("Tryb: RAW VIDEO")
        print(f"Klasy: {dataset.class_names}")
        print(f"Liczba segmentów: {len(dataset)}")
        model = CNNLSTM(num_classes=dataset.num_classes).to(DEVICE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 0) if not use_cache else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        for step, (x, y) in enumerate(loader, start=1):
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

            if log_every > 0 and (step % log_every == 0 or step == len(loader)):
                elapsed = time.time() - epoch_start
                steps_done = step
                steps_total = len(loader) if loader else 0
                steps_left = max(0, steps_total - steps_done)
                sec_per_step = elapsed / max(1, steps_done)
                eta_sec = int(steps_left * sec_per_step)
                eta_min = eta_sec // 60
                eta_s = eta_sec % 60
                running_acc = 100.0 * correct / total if total else 0.0
                running_loss = total_loss / max(1, steps_done)
                print(
                    f"  epoch {epoch}/{EPOCHS}  step {steps_done}/{steps_total}  "
                    f"loss {running_loss:.4f}  acc {running_acc:.2f}%  "
                    f"ETA {eta_min:02d}:{eta_s:02d}"
                )

        acc = 100.0 * correct / total if total else 0.0
        avg_loss = total_loss / len(loader) if loader else 0.0
        print(f"Epoka {epoch}/{EPOCHS}  loss: {avg_loss:.4f}  dokładność: {acc:.2f}%")

        if save_each_epoch:
            Path(MODEL_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
            epoch_path = Path(MODEL_CHECKPOINT).with_name(f"model_danger_epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": dataset.class_names,
                    "num_classes": dataset.num_classes,
                    "epoch": epoch,
                    "use_cache": use_cache,
                },
                epoch_path,
            )
            print(f"  Zapisano checkpoint: {epoch_path}")

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
