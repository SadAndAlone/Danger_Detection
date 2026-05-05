"""
Konfiguracja modelu wykrywania zagrożeń w wideo.
"""
#   python -m danger_detection.test_video videos\15433110_3840_2160_30fps.mp4 output_frames
#   python -m danger_detection.train

from pathlib import Path

# Klasy zagrożeń (zgodnie z wymaganiami projektu)
CLASSES = [
    "bojka",
    "palenie",
    "pozar",
    "brak_zagrozenia",
]
NUM_CLASSES = len(CLASSES)

# Rozmiar klatki na wejściu do CNN: **720p** = 1280×720 (szerokość × wysokość w OpenCV resize).
# Uwaga: większy obraz = więcej VRAM i wolniejszy trening — przy OOM zmniejsz BATCH_SIZE.
IMG_WIDTH = 1280
IMG_HEIGHT = 720
# Długość sekwencji: ile klatek trafia do LSTM (np. ~2 s przy 8 fps)
SEQ_LEN = 16
# Co ile sekund wyciągać klatkę przy analizie wideo (segmenty)
EXTRACT_FPS = 8
# Długość segmentu w sekundach przy analizie
SEGMENT_DURATION_SEC = 2.0
# Próg pewności: powyżej tej wartości zapisujemy przedział jako zagrożenie
CONFIDENCE_THRESHOLD = 0.70

# Nagrywanie klipu przy alarmie z kamery (live_camera): sekundy przed/po wykryciu
ALERT_CLIP_BEFORE_SEC = 5.0
ALERT_CLIP_AFTER_SEC = 5.0
# Minimalny odstęp między wysłanymi alarmami (żeby nie spamować API)
ALERT_COOLDOWN_SEC = 30.0
# Katalog zapisu mp4 (nie commitujemy dużych plików — .gitignore)
ALERT_CLIPS_DIR = Path(__file__).resolve().parent / "alert_clips"
# Rozdzielczość klipu alarmowego (mniejsza = mniej kopiowania RAM i płynniejszy podgląd).
# Wysokość 0 = proporcjonalnie do szerokości i proporcji kamery.
ALERT_CLIP_RECORD_WIDTH = 640
ALERT_CLIP_RECORD_HEIGHT = 0
# Maks. szerokość okna podglądu (live_camera) — mniejsza = mniej pikseli do rysowania
LIVE_PREVIEW_MAX_WIDTH = 640

# Ścieżki (względem katalogu projektu)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_VIDEO_DIR = BASE_DIR / "data_video"
CHECKPOINT_DIR = BASE_DIR / "danger_detection" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT = CHECKPOINT_DIR / "model_danger.pth"

# Hiperparametry treningu (domyślne). Przy 720p zacznij od mniejszego batcha; na RTX 5090 możesz podnieść, jeśli VRAM pozwala.
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3


#cd d:\Uni_Project\ImageClassifier\uni_project
#python -m danger_detection.live_camera
#python -m danger_detection.alerts_client