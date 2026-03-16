"""
Классификатор изображений с нуля
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Классы CIFAR-10
CLASS_NAMES_PL = [
    "samolot", "samochód", "ptak", "kot", "jeleń",
    "pies", "żaba", "koń", "statek", "ciężarówka",
]

# Папки в data_photos
RUS_TO_PL = {
    "кот": "kot", "кошка": "kot", "собака": "pies", "самолёт": "samolot", "машина": "samochód", "человек": "człowiek", "жаба": "żaba", "птица": "ptak", "змея": "wąż",
}

# Размер входа модели: 64×64
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# После 3 свёрток с MaxPool(2): 64→32→16→8
FEATURE_SIZE = 8
LINEAR_IN = 128 * FEATURE_SIZE * FEATURE_SIZE


# Основная сверточная сеть, которая по картинке предсказывает класс
class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LINEAR_IN, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Загружает собственные фото из папок
def get_dataloaders_photos(data_root, batch_size=32):
    """Данные из папок: data_root/cat/*.jpg, data_root/dog/*.jpg и т.д. — для обучения на своих фото."""
    from torchvision.datasets import ImageFolder
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    data_dir = Path(__file__).resolve().parent / data_root
    if not data_dir.exists():
        return None, None, None
    dataset = ImageFolder(root=str(data_dir), transform=train_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, dataset.classes, len(dataset.classes)


# Загружает датасет CIFAR-10
def get_dataloaders():
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    data_dir = Path(__file__).resolve().parent / "data"
    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


# Обучает модель на CIFAR-10
def train():
    """Обучить модель и сохранить веса."""
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    train_loader, test_loader = get_dataloaders()
    model = MyCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"Epoka {epoch+1}/{EPOCHS}, loss: {total_loss/len(train_loader):.4f}, dokładność: {acc:.2f}%")

    path = checkpoint_dir / "model_cifar10.pth"
    torch.save(model.state_dict(), path)
    print(f"Model zapisany: {path}")
    print("(Wejście 64×64, 10 epok, augmentacje — przeucz, jeśli był stary checkpoint 32×32.)")
    return model


# Обучает модель на собственных фото из data_photos
def train_from_photos(folder="data_photos", epochs=30):
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    loader, class_names, num_classes = get_dataloaders_photos(folder)
    if loader is None or num_classes < 2:
        print(f"Folder {folder} nie istnieje lub ma mniej niż 2 klasy.")
        print("Utwórz podfoldery, np. data_photos/kot/, data_photos/pies/ i wstaw tam zdjęcia (.jpg).")
        return
    n_total = len(loader.dataset)
    if n_total < 10:
        print(f"Za mało zdjęć: {n_total}. Dodaj co najmniej 10–20 zdjęć na klasę (lepiej 30–50).")
        return
    names_pl = [RUS_TO_PL.get(c, c) for c in class_names]
    print(f"Klasy: {names_pl}. Łącznie zdjęć: {n_total}")
    model = MyCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Для своих фото берём чуть выше скорость обучения чтобы модель быстрее подстроилась
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        acc = 100 * correct / total
        print(f"Epoka {epoch+1}/{epochs}, loss: {total_loss/len(loader):.4f}, dokładność: {acc:.2f}%")
    path = checkpoint_dir / "model_photos.pth"
    torch.save({"state_dict": model.state_dict(), "classes": class_names}, path)
    print(f"Model zapisany: {path}")
    return model


# Размер вырезки из большого фото перед сжатием до 32x32
CROP_SIZE = 224
# Сколько вырезок брать с фото
NUM_CROPS = 5


# Нарезает большое фото на несколько квадратных вырезок
def _get_crops(pil_img, num_crops=5):
    from PIL import Image
    w, h = pil_img.size
    if w < CROP_SIZE or h < CROP_SIZE:
        scale = max(CROP_SIZE / w, CROP_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        w, h = pil_img.size
    # одна большая сторона может быть больше CROP_SIZE — режем по центру/углам
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    to_32 = transforms.Resize((IMG_SIZE, IMG_SIZE))
    tensors = []
    if num_crops == 5:
        # центр и 4 угла
        positions = [
            ((w - CROP_SIZE) // 2, (h - CROP_SIZE) // 2),  # центр
            (0, 0),  # верх-лево
            (max(0, w - CROP_SIZE), 0),  # верх-право
            (0, max(0, h - CROP_SIZE)),  # низ-лево
            (max(0, w - CROP_SIZE), max(0, h - CROP_SIZE)),  # низ-право
        ]
    else:
        # сетка 3x3 = 9 вырезок
        step_x = max(0, (w - CROP_SIZE) // 2)
        step_y = max(0, (h - CROP_SIZE) // 2)
        positions = []
        for i in range(3):
            for j in range(3):
                x = min(i * step_x, w - CROP_SIZE) if step_x > 0 else (w - CROP_SIZE) // 2
                y = min(j * step_y, h - CROP_SIZE) if step_y > 0 else (h - CROP_SIZE) // 2
                if step_x == 0:
                    x = (w - CROP_SIZE) // 2
                if step_y == 0:
                    y = (h - CROP_SIZE) // 2
                positions.append((x, y))
    for (cx, cy) in positions:
        box = (cx, cy, cx + CROP_SIZE, cy + CROP_SIZE)
        crop = pil_img.crop(box)
        t = transforms.ToTensor()(crop)
        t = normalize(t)
        t = to_32(t)
        tensors.append(t)
    return torch.stack(tensors)


# Делает предсказание по одной картинке/тензору и возвращает top-k классов.
def predict_image(model, image_path_or_tensor, class_names=None, top_k=1, multi_crop=True):
    if class_names is None:
        class_names = CLASS_NAMES_PL
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if isinstance(image_path_or_tensor, (str, Path)):
        from PIL import Image
        img = Image.open(image_path_or_tensor).convert("RGB")
        if multi_crop:
            x = _get_crops(img, num_crops=NUM_CROPS)
        else:
            # один центр кадра (объект обычно в центре) — лучше чем сжимать всё фото в 32×32
            single_crop = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(CROP_SIZE),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            x = single_crop(img).unsqueeze(0)
    else:
        x = image_path_or_tensor
    model.eval()
    with torch.no_grad():
        x = x.to(DEVICE)
        logits = model(x)
        # если батч из нескольких вырезок — усредняем вероятности
        probs = torch.softmax(logits, dim=1)
        if probs.dim() == 2 and probs.size(0) > 1:
            probs = probs.mean(dim=0, keepdim=True)[0]
        else:
            probs = probs[0]
        top_probs, top_idx = torch.topk(probs, min(top_k, len(class_names)))
    pred = top_idx[0].item()
    prob = top_probs[0].item()
    if top_k <= 1:
        return pred, class_names[pred], prob
    top_list = [(class_names[idx], top_probs[i].item()) for i, idx in enumerate(top_idx.tolist())]
    return pred, class_names[pred], prob, top_list


# Показывает примеры работы модели на тестовом наборе CIFAR-10
def demo_from_dataset(num_examples=5):
    """Демо: картинки из тестового набора CIFAR — модель на них обучена, ответы уверенные."""
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    path = checkpoint_dir / "model_cifar10.pth"
    if not path.exists():
        print("Najpierw uruchom uczenie: python uni_project.py train")
        return
    _, test_loader = get_dataloaders()
    model = MyCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    it = iter(test_loader)
    print("Demo na obrazkach z CIFAR-10 (jak przy uczeniu):\n")
    for _ in range(num_examples):
        images, labels = next(it)
        img, true_label = images[0:1], labels[0].item()
        _, name_pl, prob = predict_image(model, img, top_k=1)
        ok = "✓" if name_pl == CLASS_NAMES_PL[true_label] else "✗"
        print(f"  Prawda: {CLASS_NAMES_PL[true_label]:12} → Predykcja: {name_pl:12} ({prob:.0%}) {ok}")
    print("\n(Na własnych zdjęciach wynik bywa gorszy: model uczył się na CIFAR, nie na prawdziwych fotkach.)")


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent
    ckpt = base / "checkpoints" / "model_cifar10.pth"

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict" and len(sys.argv) > 2:
        if not ckpt.exists():
            print("Najpierw wytrenuj: python uni_project.py train")
        else:
            model = MyCNN(num_classes=10).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            _, _, _, top_list = predict_image(model, sys.argv[2], top_k=3, multi_crop=True)
            print("Top 3:")
            for name, p in top_list:
                print(f"  {name}: {p:.1%}")
            best_name, best_prob = top_list[0]
            if best_prob < 0.5:
                print("\n(Model uczył się na CIFAR-10, nie na prawdziwych zdjęciach — często się myli. Demo CIFAR: python uni_project.py demo)")
            else:
                print(f"\nNa zdjęciu: {best_name} (pewność {best_prob:.1%})")
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_from_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "train_photos":
        train_from_photos()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict_photos" and len(sys.argv) > 2:
        ckpt_photos = base / "checkpoints" / "model_photos.pth"
        if not ckpt_photos.exists():
            print("Najpierw wytrenuj na własnych zdjęciach: python uni_project.py train_photos")
            print("(Utwórz folder data_photos z podfolderami po rosyjsku: кот, собака, самолёт, машина, поезд, человек, жаба, птица, змея — i wstaw zdjęcia.)")
        else:
            try:
                data = torch.load(ckpt_photos, map_location=DEVICE, weights_only=False)
            except TypeError:
                data = torch.load(ckpt_photos, map_location=DEVICE)
            class_names = data["classes"]
            model = MyCNN(num_classes=len(class_names)).to(DEVICE)
            model.load_state_dict(data["state_dict"])
            _, _, _, top_list = predict_image(model, sys.argv[2], class_names=class_names, top_k=3, multi_crop=True)
            print("Top 3:")
            for name, p in top_list:
                print(f"  {RUS_TO_PL.get(name, name)}: {p:.1%}")
            best_name_pl = RUS_TO_PL.get(top_list[0][0], top_list[0][0])
            print(f"\nNa zdjęciu: {best_name_pl} (pewność {top_list[0][1]:.1%})")
    else:
        print("  Uczenie na CIFAR:   python uni_project.py train; Uczenie na zdjęciach: python uni_project.py train_photos ")
        print("  Predykcja (model CIFAR): python uni_project.py predict ścieżka; Predykcja (model na zdjęciach): python uni_project.py predict_photos ścieżka ")
        print("  Demo CIFAR:         python uni_project.py demo")
