"""
Dzielenie wideo na klatki i segmenty (OpenCV).
"""
import cv2
from pathlib import Path
from typing import List, Tuple, Generator
import numpy as np


def video_to_frames(
    video_path: str | Path,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    resize: Tuple[int, int] | None = (64, 64),
) -> List[np.ndarray]:
    """
    Czyta wideo i zwraca listę klatek (BGR, numpy).
    :param video_path: ścieżka do pliku wideo
    :param every_n_frames: co którą klatkę brać (1 = wszystkie)
    :param max_frames: maksymalna liczba klatek (None = bez limitu)
    :param resize: (H, W) do przeskalowania; None = oryginalny rozmiar
    :return: lista arrayów (H, W, 3) BGR
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Plik wideo nie istnieje: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {path}")

    frames = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n_frames == 0:
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    return frames


def video_to_frame_generator(
    video_path: str | Path,
    every_n_frames: int = 1,
    resize: Tuple[int, int] | None = (64, 64),
) -> Generator[np.ndarray, None, None]:
    """
    Generator klatek (oszczędza pamięć przy długich wideo).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Plik wideo nie istnieje: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {path}")

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n_frames == 0:
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
                yield frame
            idx += 1
    finally:
        cap.release()


def get_video_info(video_path: str | Path) -> dict:
    """Zwraca fps, liczbę klatek, czas trwania (s)."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Plik wideo nie istnieje: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def extract_segments(
    video_path: str | Path,
    segment_duration_sec: float = 2.0,
    fps: float = 8.0,
    resize: Tuple[int, int] = (64, 64),
) -> List[Tuple[float, float, List[np.ndarray]]]:
    """
    Dzieli wideo na segmenty czasowe i dla każdego zwraca listę klatek.
    :return: lista (start_sec, end_sec, list_of_frames)
    """
    info = get_video_info(video_path)
    video_fps = info["fps"]
    duration_sec = info["duration_sec"]

    # Co ile klatek oryginalnego wideo brać jedną (żeby uzyskać ~fps klatek na segment)
    if video_fps <= 0:
        video_fps = 25.0
    step = max(1, int(round(video_fps / fps)))

    frames_per_segment = max(1, int(round(fps * segment_duration_sec)))
    segments = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {video_path}")

    t_sec = 0.0
    frame_interval = 1.0 / video_fps
    segment_frames = []
    segment_start = 0.0
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if segment_frames:
                    segments.append((segment_start, t_sec, segment_frames))
                break
            if idx % step == 0:
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
                if not segment_frames:
                    segment_start = t_sec
                segment_frames.append(frame)
                if len(segment_frames) >= frames_per_segment:
                    segments.append((segment_start, t_sec + frame_interval, segment_frames))
                    segment_frames = []
            t_sec += frame_interval
            idx += 1
    finally:
        cap.release()

    return segments


def extract_frames_for_segment(
    video_path: str | Path,
    segment_index: int,
    segment_duration_sec: float = 2.0,
    fps: float = 8.0,
    resize: Tuple[int, int] = (64, 64),
) -> List[np.ndarray]:
    """
    Wyciąga klatki tylko dla jednego segmentu (bez ładowania całego wideo).
    :param segment_index: który segment (0, 1, 2, ...)
    :return: lista klatek (numpy BGR) dla tego segmentu
    """
    info = get_video_info(video_path)
    video_fps = info["fps"]
    if video_fps <= 0:
        video_fps = 25.0
    step = max(1, int(round(video_fps / fps)))
    frames_per_segment = max(1, int(round(fps * segment_duration_sec)))
    start_sec = segment_index * segment_duration_sec
    if start_sec >= info["duration_sec"]:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {video_path}")

    start_ms = int(start_sec * 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    frame_interval = 1.0 / video_fps
    frames = []
    idx = 0
    t = start_sec
    end_sec = start_sec + segment_duration_sec

    try:
        while t < end_sec and len(frames) < frames_per_segment:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
            idx += 1
            t += frame_interval
    finally:
        cap.release()

    return frames
