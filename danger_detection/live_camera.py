"""
Демонстрация анализа потока с веб‑камеры в реальном времени.

Использует:
- модель CNNLSTM (pozar / brak_zagrozenia и т.д.),
- параметры из config.py,
- преобразование кадров из dataset.frames_to_tensor.

Запуск:
  python -m danger_detection.live_camera

Управление:
  - окно с картинкой с камеры,
  - каждые ~2 секунды берётся сегмент из последних кадров и анализируется моделью,
  - результат печатается в консоль и отображается вверху кадра,
  - для выхода нажать клавишу 'q' в окне OpenCV.
"""

from collections import deque
from typing import Deque, Tuple
import time

import cv2
import numpy as np
import torch

from .config import IMG_SIZE, SEQ_LEN, CLASSES, CONFIDENCE_THRESHOLD
from .model_cnn_lstm import CNNLSTM
from .dataset import frames_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(checkpoint_path: str) -> Tuple[CNNLSTM, list]:
    """
    Загружает модель и классы из чекпоинта.
    Если в чекпоинте есть поле 'classes', используем его (кол-во классов может отличаться
    от config.CLASSES).
    """
    # Сначала пробуем вытащить классы из чекпоинта
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = CLASSES
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "classes" in ckpt:
            class_names = ckpt["classes"]
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

    model = CNNLSTM(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, list(class_names)


def analyze_segment(model: CNNLSTM, frames: list, class_names: list) -> Tuple[str, float]:
    """
    Анализирует список кадров, возвращает (название класса, вероятность).
    """
    if not frames:
        return "no_frames", 0.0

    tensor = frames_to_tensor(frames, SEQ_LEN, IMG_SIZE)  # (T, C, H, W)
    tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


def main():
    from .config import MODEL_CHECKPOINT

    print(f"Устройство: {DEVICE}")
    print(f"Загружаю модель из: {MODEL_CHECKPOINT}")
    model, class_names = init_model(str(MODEL_CHECKPOINT))
    print(f"Классы модели: {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб‑камеру (cv2.VideoCapture(0)).")
        return

    # Буфер кадров ~ на 10 секунд (при 8–10 FPS этого достаточно)
    frame_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN * 5)
    last_analyze_time = 0.0
    analyze_interval_sec = 2.0

    print("Старт захвата с камеры. Нажми 'q' в окне, чтобы выйти.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Камера перестала отдавать кадры.")
                break

            # Resize под размер модели
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frame_buffer.append(frame_resized)

            now = time.time()
            text = ""

            # Периодический анализ накопленного буфера
            if now - last_analyze_time >= analyze_interval_sec and len(frame_buffer) >= SEQ_LEN:
                last_analyze_time = now
                frames = list(frame_buffer)
                threat, prob = analyze_segment(model, frames, class_names)
                text = f"{threat} ({prob:.0%})"
                print(f"[{time.strftime('%H:%M:%S')}] threat={threat}, prob={prob:.3f}")

            # Отрисовка на экране (последний кадр из буфера)
            display_frame = frame_resized.copy()
            if text:
                color = (0, 255, 0)
                # если вероятность выше порога — красным
                if any(t in text for t in ["pozar", "bojka", "palenie"]) and "(" in text:
                    try:
                        p = float(text.split("(")[1].split("%")[0]) / 100.0
                        if p >= CONFIDENCE_THRESHOLD:
                            color = (0, 0, 255)
                    except Exception:
                        pass
                cv2.putText(
                    display_frame,
                    text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Визуальный апскейл (модель работает с IMG_SIZE×IMG_SIZE из config)
            disp = max(480, min(960, IMG_SIZE * 3))
            display_big = cv2.resize(display_frame, (disp, disp), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Danger Detection - Live Camera", display_big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

