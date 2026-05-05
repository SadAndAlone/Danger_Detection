"""
Запуск:
  python -m danger_detection.live_camera
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import threading
import time

import cv2
import numpy as np
import torch

from .config import (
    CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    SEQ_LEN,
    CONFIDENCE_THRESHOLD,
    ALERT_CLIP_BEFORE_SEC,
    ALERT_CLIP_AFTER_SEC,
    ALERT_COOLDOWN_SEC,
    ALERT_CLIPS_DIR,
    ALERT_CLIP_RECORD_WIDTH,
    ALERT_CLIP_RECORD_HEIGHT,
    LIVE_PREVIEW_MAX_WIDTH,
)
from .model_cnn_lstm import CNNFeatureExtractor, CNNLSTM, LSTMClassifier, CNNExtractorLSTM
from .dataset import frames_to_tensor
from .video_utils import write_bgr_frames_to_mp4


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nazwy klas w treningu (PL) -> typ dla API alertów (EN)
CLASS_NAME_TO_ALERT_THREAT = {
    "pozar": "fire",
    "palenie": "smoke",
    "bojka": "fight",
}


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

    with torch.inference_mode():
        logits = model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


def _pairs_to_frames_sorted(pairs: List[Tuple[float, np.ndarray]]) -> Tuple[List[np.ndarray], float]:
    """Sortuje po czasie, usuwa puste; zwraca klatki i fps wyprowadzony z rzeczywistego rozstawu czasu."""
    if not pairs:
        return [], 25.0
    pairs = sorted(pairs, key=lambda x: x[0])
    times = [t for t, _ in pairs]
    frames = [f for _, f in pairs]
    span = max(times[-1] - times[0], 1e-3)
    fps_out = max(1.0, min(60.0, len(frames) / span))
    return frames, fps_out


def _inference_worker(
    stop_event: threading.Event,
    buffer_lock: threading.Lock,
    frame_buffer: Deque[np.ndarray],
    infer_lock: threading.Lock,
    shared: Dict[str, object],
    model: torch.nn.Module,
    class_names: list,
    analyze_interval_sec: float,
) -> None:
    """GPU / model w osobnym wątku — pętla główna tylko OpenCV, płynniejszy podgląd."""
    local_last = 0.0
    while not stop_event.is_set():
        if stop_event.wait(0.02):
            break
        now = time.time()
        if now - local_last < analyze_interval_sec:
            continue
        with buffer_lock:
            if len(frame_buffer) < SEQ_LEN:
                continue
            frames = list(frame_buffer)
        threat, prob = analyze_segment(model, frames, class_names)
        with infer_lock:
            shared["threat"] = threat
            shared["prob"] = float(prob)
            shared["tick"] = int(shared["tick"]) + 1  # type: ignore[arg-type]
        local_last = now


def main():
    from .config import MODEL_CHECKPOINT
    from . import alerts_client

    print(f"Устройство: {DEVICE}")
    print(f"Загружаю модель из: {MODEL_CHECKPOINT}")
    model, class_names = init_model(str(MODEL_CHECKPOINT))
    print(f"Классы модели: {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб‑камеру (cv2.VideoCapture(0)).")
        return

    # Mniejszy bufor urządzenia = mniejsze opóźnienie obrazu (działa nie na wszystkich sterownikach).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cam_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if cam_fps < 1.0 or cam_fps > 120.0:
        cam_fps = 25.0

    # Bufor do modelu LSTM (współdzielony z wątkiem inferencji)
    buffer_lock = threading.Lock()
    frame_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN * 5)
    infer_lock = threading.Lock()
    shared: Dict[str, object] = {"threat": "", "prob": 0.0, "tick": 0}
    stop_inference = threading.Event()
    analyze_interval_sec = 2.0
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or IMG_WIDTH
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or IMG_HEIGHT
    clip_rw = min(int(ALERT_CLIP_RECORD_WIDTH), IMG_WIDTH)
    if ALERT_CLIP_RECORD_HEIGHT and int(ALERT_CLIP_RECORD_HEIGHT) > 0:
        clip_rh = int(ALERT_CLIP_RECORD_HEIGHT)
    else:
        clip_rh = max(1, int(round(clip_rw * nh / max(nw, 1))))
    clip_size = (clip_rw, clip_rh)  # (width, height) dla cv2.resize

    # Pierścieniowy bufor małych klatek (tylko pod klip) — bez pełnego frame.copy() w 1080p
    ring_max = max(int(cam_fps * (ALERT_CLIP_BEFORE_SEC + ALERT_CLIP_AFTER_SEC + 4.0)), 64)
    raw_ring: Deque[Tuple[float, np.ndarray]] = deque(maxlen=ring_max)

    last_overlay = ""
    last_overlay_prob = 0.0
    last_seen_tick = 0

    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(
            stop_inference,
            buffer_lock,
            frame_buffer,
            infer_lock,
            shared,
            model,
            class_names,
            analyze_interval_sec,
        ),
        daemon=True,
    )
    infer_thread.start()

    # Stan nagrywania „po” zdarzeniu
    capture_after_until: Optional[float] = None
    clip_pre_pairs: List[Tuple[float, np.ndarray]] = []
    clip_post_pairs: List[Tuple[float, np.ndarray]] = []
    clip_threat_api: Optional[str] = None
    last_alert_sent_time = 0.0

    ALERT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Старт захвата с камеры. Нажми 'q' в окне, чтобы выйти.")
    print(
        f"Alarm clip: {ALERT_CLIP_BEFORE_SEC}s before + {ALERT_CLIP_AFTER_SEC}s after → "
        f"{ALERT_CLIPS_DIR} (zapis {clip_rw}×{clip_rh})"
    )
    print(f"Podgląd max {LIVE_PREVIEW_MAX_WIDTH}px; inferencja w osobnym wątku co {analyze_interval_sec}s.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Камера перестала отдавать кадры.")
                break

            now = time.time()

            frame_resized = cv2.resize(
                frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
            # Klip z rozdzielczości modelowej (taniej niż z pełnej matrycy kamery)
            clip_frame = cv2.resize(
                frame_resized, clip_size, interpolation=cv2.INTER_AREA
            )
            with buffer_lock:
                frame_buffer.append(frame_resized)

            text = ""
            if capture_after_until is not None:
                clip_post_pairs.append((now, clip_frame))
                if now >= capture_after_until:
                    all_pairs = clip_pre_pairs + clip_post_pairs
                    frames_clip, fps_clip = _pairs_to_frames_sorted(all_pairs)
                    capture_after_until = None
                    clip_post_pairs = []
                    clip_pre_pairs = []
                    threat_api = clip_threat_api
                    clip_threat_api = None
                    if frames_clip and threat_api:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        out_path = ALERT_CLIPS_DIR / f"{ts}_{threat_api}.mp4"
                        try:
                            write_bgr_frames_to_mp4(frames_clip, out_path, fps_clip)
                            print(f"Zapisano klip: {out_path} ({len(frames_clip)} klatek, fps≈{fps_clip:.1f})")
                            aid = alerts_client.create_alert_with_local_clip(threat_api, out_path)
                            if aid is not None:
                                print(f"Alert wysłany, id={aid}")
                            last_alert_sent_time = time.time()
                        except Exception as exc:
                            print(f"Błąd zapisu/wysyłki klipu: {exc}")
            else:
                with infer_lock:
                    cur_tick = int(shared["tick"])
                    threat = str(shared["threat"])
                    prob = float(shared["prob"])
                if cur_tick != last_seen_tick:
                    last_seen_tick = cur_tick
                    last_overlay = threat
                    last_overlay_prob = prob
                    text = f"{threat} ({prob:.0%})"
                    print(f"[{time.strftime('%H:%M:%S')}] threat={threat}, prob={prob:.3f}")

                    api_type = CLASS_NAME_TO_ALERT_THREAT.get(threat)
                    if (
                        api_type is not None
                        and prob >= CONFIDENCE_THRESHOLD
                        and now - last_alert_sent_time >= ALERT_COOLDOWN_SEC
                    ):
                        trigger_t = now
                        clip_pre_pairs = [
                            (t, f)
                            for t, f in raw_ring
                            if t >= trigger_t - ALERT_CLIP_BEFORE_SEC
                        ]
                        clip_pre_pairs.append((trigger_t, clip_frame))
                        clip_post_pairs = []
                        clip_threat_api = api_type
                        capture_after_until = trigger_t + ALERT_CLIP_AFTER_SEC
                        print(
                            f"*** ALARM {api_type} (prob={prob:.2f}) — nagrywanie +{ALERT_CLIP_AFTER_SEC}s ***"
                        )

            raw_ring.append((now, clip_frame))

            fh, fw = frame.shape[:2]
            pv_w = min(int(LIVE_PREVIEW_MAX_WIDTH), fw)
            pv_h = max(1, int(fh * pv_w / max(fw, 1)))
            display_big = cv2.resize(
                frame, (pv_w, pv_h), interpolation=cv2.INTER_NEAREST
            )
            overlay = text or (
                f"{last_overlay} ({last_overlay_prob:.0%})" if last_overlay else ""
            )
            if overlay:
                color = (0, 255, 0)
                if last_overlay in CLASS_NAME_TO_ALERT_THREAT and last_overlay_prob >= CONFIDENCE_THRESHOLD:
                    color = (0, 0, 255)
                cv2.putText(
                    display_big,
                    overlay,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if capture_after_until is not None:
                remain = max(0.0, capture_after_until - now)
                cv2.putText(
                    display_big,
                    f"REC +{remain:.1f}s",
                    (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 128, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Danger Detection - Live Camera", display_big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_inference.set()
        infer_thread.join(timeout=8.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

