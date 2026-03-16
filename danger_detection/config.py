"""
Konfiguracja modelu wykrywania zagrożeń w wideo.
"""
#   python -m danger_detection.test_video videos\15433110_3840_2160_30fps.mp4 output_frames
#   python -m danger_detection.train

from pathlib import Path

# Klasy zagrożeń (zgodnie z wymaganiami projektu)
CLASSES = [
    "bójka",
    "palenie",
    "pożar",
    "brak zagrożenia",
]
NUM_CLASSES = len(CLASSES)

# Obraz: rozmiar klatki na wejściu do CNN (H, W)
IMG_SIZE = 64
# Długość sekwencji: ile klatek trafia do LSTM (np. ~2 s przy 8 fps)
SEQ_LEN = 16
# Co ile sekund wyciągać klatkę przy analizie wideo (segmenty)
EXTRACT_FPS = 8
# Długość segmentu w sekundach przy analizie
SEGMENT_DURATION_SEC = 2.0
# Próg pewności: powyżej tej wartości zapisujemy przedział jako zagrożenie
CONFIDENCE_THRESHOLD = 0.70

# Ścieżki (względem katalogu projektu)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_VIDEO_DIR = BASE_DIR / "data_video"
CHECKPOINT_DIR = BASE_DIR / "danger_detection" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT = CHECKPOINT_DIR / "model_danger.pth"

# Hiperparametry treningu (domyślne)
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
