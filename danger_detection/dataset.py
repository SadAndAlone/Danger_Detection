"""
Dataset ładujący segmenty wideo z data_video/<klasa>/*.mp4 dla treningu CNN+LSTM.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from .config import (
    DATA_VIDEO_DIR,
    IMG_SIZE,
    SEQ_LEN,
    SEGMENT_DURATION_SEC,
    EXTRACT_FPS,
)
from .video_utils import get_video_info, extract_frames_for_segment

# Rozszerzenia plików wideo
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """BGR [0,255] -> float, normalizacja (0.5, 0.5, 0.5) jak w reszcie projektu."""
    x = frame.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x


def frames_to_tensor(frames: list, seq_len: int, img_size: int) -> torch.Tensor:
    """
    Lista klatek (H,W,3) BGR -> tensor (seq_len, 3, H, W).
    Jeśli mniej niż seq_len klatek: powtarzamy ostatnią. Jeśli więcej: obcinamy.
    """
    if not frames:
        return torch.zeros(seq_len, 3, img_size, img_size, dtype=torch.float32)

    arr = np.stack([_normalize_frame(f) for f in frames], axis=0)
    # (N, H, W, 3) -> (N, 3, H, W)
    arr = np.transpose(arr, (0, 3, 1, 2))
    T = arr.shape[0]
    if T >= seq_len:
        arr = arr[:seq_len]
    else:
        pad = np.repeat(arr[-1:], seq_len - T, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return torch.from_numpy(arr).float()


class DangerVideoDataset(Dataset):
    """
    Dataset: każdy element to jeden segment wideo (sekwencja klatek) + etykieta.
    Katalogi w data_video/ = nazwy klas (np. pożar, brak_zagrożenia).
    """

    def __init__(
        self,
        root: Path | str | None = None,
        segment_duration_sec: float = SEGMENT_DURATION_SEC,
        fps: float = EXTRACT_FPS,
        seq_len: int = SEQ_LEN,
        resize: tuple = (IMG_SIZE, IMG_SIZE),
    ):
        self.root = Path(root) if root is not None else DATA_VIDEO_DIR
        self.segment_duration_sec = segment_duration_sec
        self.fps = fps
        self.seq_len = seq_len
        self.resize = resize

        if not self.root.exists():
            raise FileNotFoundError(f"Katalog danych nie istnieje: {self.root}")

        # Klasy = nazwy podkatalogów (posortowane)
        self.class_names = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        if not self.class_names:
            raise ValueError(f"Brak podkatalogów w {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Lista (ścieżka_wideo, indeks_segmentu, class_idx)
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]
            for path in class_dir.iterdir():
                if path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                try:
                    info = get_video_info(path)
                except Exception:
                    continue
                duration = info["duration_sec"]
                n_segments = max(1, int(duration / segment_duration_sec))
                for seg_idx in range(n_segments):
                    self.samples.append((str(path), seg_idx, class_idx))

        if not self.samples:
            raise ValueError(
                f"Nie znaleziono żadnego wideo w {self.root}. "
                f"Umieść pliki .mp4 w podkatalogach, np. {self.root}/pożar/, {self.root}/brak_zagrożenia/"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path, segment_index, class_idx = self.samples[idx]
        frames = extract_frames_for_segment(
            video_path,
            segment_index,
            segment_duration_sec=self.segment_duration_sec,
            fps=self.fps,
            resize=self.resize,
        )
        x = frames_to_tensor(frames, self.seq_len, self.resize[0])
        return x, class_idx

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
