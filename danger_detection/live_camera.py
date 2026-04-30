"""
Запуск:
  python -m danger_detection.live_camera
"""

from collections import deque
from typing import Deque, Tuple
import time

import cv2
import numpy as np
import torch

from .config import IMG_HEIGHT, IMG_WIDTH, SEQ_LEN, CLASSES, CONFIDENCE_THRESHOLD
from .model_cnn_lstm import CNNFeatureExtractor, CNNLSTM, LSTMClassifier, CNNExtractorLSTM
from .dataset import frames_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(checkpoint_path: str) -> Tuple[torch.nn.Module, list]:
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

    # Two checkpoint formats:
    # 1) raw_video: {"state_dict": CNNLSTM...}
    # 2) feature_cache: {"extractor_state_dict": CNNFeatureExtractor..., "lstm_state_dict": LSTMClassifier...}
    if isinstance(ckpt, dict) and "lstm_state_dict" in ckpt:
        extractor = CNNFeatureExtractor()
        if ckpt.get("extractor_state_dict") is not None:
            extractor.load_state_dict(ckpt["extractor_state_dict"])
        head = LSTMClassifier(num_classes=len(class_names))
        head.load_state_dict(ckpt["lstm_state_dict"])
        model = CNNExtractorLSTM(extractor, head)
    else:
        model = CNNLSTM(num_classes=len(class_names))
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, list(class_names)


def analyze_segment(model: torch.nn.Module, frames: list, class_names: list) -> Tuple[str, float]:
    """
    Анализирует список кадров, возвращает (название класса, вероятность).
    """
    if not frames:
        return "no_frames", 0.0

    tensor = frames_to_tensor(frames, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH)  # (T, C, H, W)
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
            # OpenCV: (width, height)
            frame_resized = cv2.resize(
                frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
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

            # Превью: уменьшаем широкий кадр до ~960 px по ширине (сохраняем пропорции)
            preview_w = min(960, IMG_WIDTH)
            preview_h = int(IMG_HEIGHT * preview_w / IMG_WIDTH)
            display_big = cv2.resize(
                display_frame, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("Danger Detection - Live Camera", display_big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

