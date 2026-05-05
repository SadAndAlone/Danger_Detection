"""
Dzielenie wideo na klatki i segmenty (OpenCV).
"""
import shutil
import subprocess
import cv2
from pathlib import Path
from typing import Any, List, Tuple, Generator
import numpy as np

from .config import IMG_HEIGHT, IMG_WIDTH

# Domyślny resize = wejście modelu; jawne ``resize=None`` = bez skalowania (oryginalna rozdzielczość).
_USE_CONFIG_IMG_SIZE = object()


def video_to_frames(
    video_path: str | Path,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    resize: Any = _USE_CONFIG_IMG_SIZE,
) -> List[np.ndarray]:
    """
    Czyta wideo i zwraca listę klatek (BGR, numpy).
    :param video_path: ścieżka do pliku wideo
    :param every_n_frames: co którą klatkę brać (1 = wszystkie)
    :param max_frames: maksymalna liczba klatek (None = bez limitu)
    :param resize: (H, W); domyślnie ``(IMG_HEIGHT, IMG_WIDTH)`` z ``config``; ``None`` = bez skalowania
    :return: lista arrayów (H, W, 3) BGR
    """
    eff_resize: Tuple[int, int] | None
    if resize is _USE_CONFIG_IMG_SIZE:
        eff_resize = (IMG_HEIGHT, IMG_WIDTH)
    else:
        eff_resize = resize  # type: ignore[assignment]

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
                if eff_resize:
                    frame = cv2.resize(frame, (eff_resize[1], eff_resize[0]), interpolation=cv2.INTER_LINEAR)
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
    resize: Any = _USE_CONFIG_IMG_SIZE,
) -> Generator[np.ndarray, None, None]:
    """
    Generator klatek (oszczędza pamięć przy długich wideo).
    Domyślnie ``(IMG_HEIGHT, IMG_WIDTH)``; ``resize=None`` = bez skalowania.
    """
    if resize is _USE_CONFIG_IMG_SIZE:
        eff_resize: Tuple[int, int] | None = (IMG_HEIGHT, IMG_WIDTH)
    else:
        eff_resize = resize  # type: ignore[assignment]

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
                if eff_resize:
                    frame = cv2.resize(frame, (eff_resize[1], eff_resize[0]), interpolation=cv2.INTER_LINEAR)
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
    resize: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
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
    resize: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
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


def _normalize_bgr_frames_even(
    frames: List[np.ndarray],
) -> Tuple[List[np.ndarray], int, int]:
    """Ujednolica rozmiar klatek; libx264/yuv420p wymaga parzystych szer./wys."""
    h0, w0 = frames[0].shape[:2]
    w, h = w0 - (w0 % 2), h0 - (h0 % 2)
    w, h = max(w, 2), max(h, 2)
    out: List[np.ndarray] = []
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.ascontiguousarray(fr, dtype=np.uint8))
    return out, w, h


def _write_mp4_ffmpeg_libx264(
    frames: List[np.ndarray],
    out_path: Path,
    fps: float,
    w: int,
    h: int,
    ffmpeg_exe: str,
) -> None:
    """
    MP4 dla przeglądarek: H.264 baseline + yuv420p + AAC (cisza) + faststart.
    Bez ścieżki audio część odtwarzaczy HTML5 zgłasza „uszkodzony / nieobsługiwany” format.
    """
    eff_fps = float(max(1.0, min(fps, 120.0)))
    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(eff_fps),
        "-i",
        "-",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise RuntimeError("ffmpeg: brak stdin pipe")
    try:
        for fr in frames:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        err_b = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        err = err_b.decode(errors="replace")
        if rc != 0:
            raise RuntimeError(f"ffmpeg zakończył się kodem {rc}: {err[:800]}")
    except Exception:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        raise


def write_bgr_frames_to_mp4(
    frames: List[np.ndarray],
    out_path: str | Path,
    fps: float,
) -> Path:
    """
    Zapisuje listę klatek BGR do MP4.

    Preferuje **ffmpeg**: **H.264 baseline** + **yuv420p** + **AAC** (cisza) + **faststart**
    — zgodne z typowym `<video>` w Chrome; OpenCV ``mp4v`` bywa „uszkodzony” dla przeglądarek.

    Jeśli ``ffmpeg`` nie jest w PATH, używa VideoWriter (``mp4v``) i parzystych wymiarów.
    """
    if not frames:
        raise ValueError("write_bgr_frames_to_mp4: pusta lista klatek")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_n, w, h = _normalize_bgr_frames_even(frames)
    eff_fps = float(max(1.0, min(fps, 120.0)))

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        try:
            _write_mp4_ffmpeg_libx264(frames_n, out_path, eff_fps, w, h, ffmpeg_exe)
            return out_path
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, eff_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Nie można utworzyć VideoWriter: {out_path}. "
            "Zainstaluj ffmpeg (https://ffmpeg.org) i dodaj do PATH — wtedy zapis będzie H.264."
        )
    try:
        for fr in frames_n:
            writer.write(fr)
    finally:
        writer.release()
    return out_path
