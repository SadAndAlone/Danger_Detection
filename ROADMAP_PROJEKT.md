# Plan przejścia do projektu: System wykrywania niebezpiecznych sytuacji w wideo

## Cel projektu (z wymagań)

- **Klasy (semantyka):** bójka, palenie papierosów, pożar (ogień/dym), brak zagrożenia — **w kodzie i folderach `data_video/`:** `bojka`, `palenie`, `pozar`, `brak_zagrozenia` (bez polskich znaków)  
- **Pipeline:** Wideo → klatki → CNN (cechy) → LSTM (sekwencja) → klasyfikacja  
- **Wynik:** przedziały czasowe z prawdopodobieństwami (np. Bójka: 00:12–00:18, Pożar: 01:03–01:09)  
- **Stack:** Python, PyTorch, OpenCV, FastAPI, HTML/CSS

## Etapy realizacji (kolejność)

### Etap 1 — Przygotowanie danych i pipeline wideo

1. **Zbiór danych** — patrz sekcja **„Zbiory danych (datasety)”** poniżej.
2. **Dzielenie wideo na klatki (OpenCV)**
   - Skrypt: wczytaj plik wideo → zapisz klatki co N sekund (np. co 1–2 s) lub co K klatek.
   - Zapisywanie w katalogach: `data_video/bojka/`, `data_video/pozar/` itd. (albo jeden plik z adnotacjami).
3. **Dataset dla PyTorch**
   - Dataset ładujący **sekwencje klatek** (np. 16 klatek = ~2 s), nie pojedyncze zdjęcia.
   - Normalizacja spójna z obecnym kodem (np. 64×64 lub 224×224, Normalize(0.5, 0.5, 0.5)).

**Rezultat:** Możliwość ładowania „segmentów wideo” (batch sekwencji) do treningu.


### Etap 2 — Model CNN + LSTM

1. **CNN (ekstrakcja cech z klatki)**
   - Wykorzystać Twoją `MyCNN` bez warstwy klasyfikacyjnej (tylko `features` → wektor cech na klatkę).
   - Albo uprościć: kilka Conv2d + Pool → Flatten → wektor o stałym rozmiarze (np. 256 lub 512).
2. **LSTM**
   - Na wejście: sekwencja wektorów (jedna klatka = jeden wektor).
   - LSTM przetwarza sekwencję; na wyjściu (ostatni stan lub uśrednienie) → warstwa liniowa → 4 klasy.
3. **Trenowanie**
   - Loss: CrossEntropy. Metryki: accuracy, precision, recall dla każdej klasy.
   - Na początek możesz trenować na **klatkach z etykietą segmentu** (każda sekwencja ma jedną etykietę: bojka/pozar/palenie/brak_zagrozenia).

**Rezultat:** Wytrenowany model zapisany w `.pth`, który przyjmuje sekwencję klatek i zwraca prawdopodobieństwa dla 4 klas.


### Etap 3 — Analiza wideo w czasie (segmenty)

1. **Funkcja analizy pliku wideo**
   - Otwórz wideo (OpenCV), podziel na segmenty (np. co 2 s).
   - Dla każdego segmentu: wyciągnij klatki → zbuduj sekwencję → model → prawdopodobieństwa.
   - Jeśli max(prawdopodobieństwo) > próg (np. 70%) — zapisz przedział czasowy i klasę.
2. **Format wyniku**
   - Lista zdarzeń: `[{ "class": "Bójka", "start": "00:12", "end": "00:18", "confidence": 0.85 }, ...]`.
   - To będzie zwracane przez backend do frontendu.

**Rezultat:** Jedna funkcja w Pythonie: `analyze_video(path_to_video) -> list of time intervals with labels`.


### Etap 4 — Backend (FastAPI)

1. **Endpoint do przesyłania wideo**
   - `POST /upload` lub `POST /analyze`: odbierz plik wideo (multipart/form-data).
   - Zapisz tymczasowo na dysk (np. `temp/`), wywołaj `analyze_video(...)`.
   - Zwróć JSON z przedziałami czasowymi i etykietami (oraz opcjonalnie surowe prawdopodobieństwa per segment).
2. **Służbowy endpoint**
   - `GET /health` — czy serwer i model są gotowe.
3. **CORS** — włączyć dla frontendu na innym porcie (np. front na :3000, backend na :8000).

**Rezultat:** Działające API: wysyłasz wideo, dostajesz wynik analizy.


### Etap 5 — Frontend (strona WWW)

1. **Jedna strona**
   - Formularz: wybór pliku wideo (input type=file), przycisk „Uruchom analizę”.
   - Po wysłaniu: POST do FastAPI, oczekiwanie na odpowiedź (może być dłuższe — pokazać „Analiza w toku...”).
   - Wyświetlenie wyniku: tabela lub lista przedziałów (np. „Bójka: 00:12–00:18”, „Pożar: 01:03–01:09”).
2. **Technologie:** HTML, CSS, opcjonalnie minimalny JavaScript.

**Rezultat:** Użytkownik wchodzi na stronę, wgrywa wideo, klika „Analizuj” i widzi przedziały z zagrożeniami.

## Zbiory danych (datasety)

Do treningu i ewaluacji modelu potrzebne są nagrania wideo (lub wyciągnięte klatki) z adnotacjami dla czterech klas. Poniżej rekomendowane zbiory z podziałem na kategorię zagrożenia.

### Bójka (wykrywanie przemocy / fight detection)

| Zbiór | Opis | Dostęp |
|-------|------|--------|
| **UCF-Crime** | ~1900 nagrań CCTV, 13 kategorii zdarzeń, w tym **Fighting** i **Assault**; adnotacje na poziomie klatek (początek/koniec zdarzenia). | [CRCV UCF](https://www.crcv.ucf.edu/projects/real-world/), [Dropbox (oficjalny)](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0), [Hugging Face (klatki)](https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset), [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) |
| **CCTV-Fights (NTU ROSE)** | ~1000 wideo z bójkami (CCTV + mobilne), adnotacje na poziomie klatek. | [ROSE Lab NTU](https://rose1.ntu.edu.sg/dataset/cctvFights/) — rejestracja akademicka |
| **RWF-2000** | Duży, otwarty zbiór wideo do wykrywania przemocy (violence detection). | Publikacja IEEE, zbiory powiązane często na Kaggle / GitHub |
| **VFD-2000** | Ponad 2000 wideo (m.in. ulica, plaża, restauracje), scenariusze przemocy. | Papers With Code / powiązane repozytoria |

**Uwaga:** Dla klasy „bójka” można użyć kategorii **Fighting** i **Assault** z UCF-Crime oraz zbiorów typu CCTV-Fights lub RWF-2000. Model uczy się rozpoznawać charakterystyczne ruchy i pozy ciała (uderzenia, pchnięcia, walka).


### Pożar i dym (ogień / smoke detection)

| Zbiór | Opis | Dostęp |
|-------|------|--------|
| **FiSmo** | Kompilacja zbiorów: 5556 obrazów + 97 wideo (pożar, dym, segmentacja), adnotacje i ROI. | Artykuł / Academia.edu (FiSmo: Fire and Smoke Analysis) |
| **Fire and Smoke Detection Videos** | 85 wideo do wykrywania ognia i dymu. | [Kaggle](https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset) |
| **Fire-and-Smoke-Dataset (DataCluster Labs)** | Obrazy i wideo pod wczesne wykrywanie ognia i dymu. | [GitHub](https://github.com/datacluster-labs/Fire-and-Smoke-Dataset) |
| **D-Fire Dataset** | Zbiór obrazów do detekcji ognia i dymu. | [GitHub](https://github.com/gaiasd/DFireDataset) |

**Uwaga:** Dym z pożaru i dym z papierosów mają podobną teksturę (mgła, rozproszenie); model może uczyć się najpierw na „dymie w ogóle”, a dopiero potem rozróżniać kontekst (pożar vs palenie), jeśli dodamy zbiory z paleniem.


### Palenie papierosów (dym z papierosa / cigarette smoke)

| Zbiór / podejście | Opis | Dostęp |
|-------------------|------|--------|
| **Własne nagrania** | Krótkie fragmenty wideo z osobą palącą (widoczny dym, ewentualnie papieros). Najbardziej dopasowane do klasy „palenie”. | Nagrania z kamer / screeny z filmów (zgodnie z prawem i regulaminem uczelni) |
| **Zbiory „smoke” / „smoking”** | Część zbiorów ogólnych „smoke detection” zawiera też dym nie z ognia; wyszukiwanie po słowach: *smoking detection dataset*, *cigarette smoke video*. | Kaggle, GitHub, Roboflow (np. zestawy do detekcji dymu) |
| **Wykorzystanie zbiorów dymu/ognia** | Klatki z samym dymem (bez płomieni) z FiSmo lub Fire-and-Smoke można etykietować jako „dym”; część da się traktować jako bliską „paleniu” wizualnie. | Jak wyżej |

**Uwaga:** Gotowych, dużych zbiorów „tylko palenie” jest mniej niż do bójek czy pożarów. Rozsądna strategia: (1) trenować detekcję „dymu” na FiSmo / Fire-and-Smoke, (2) dołożyć własne lub znalezione klipy z paleniem i etykietować je jako „palenie papierosów”.


### Brak zagrożenia (sceny normalne)

| Zbiór / podejście | Opis | Dostęp |
|-------------------|------|--------|
| **UCF-Crime** | Kategoria **Normal** — nagrania bez zdarzeń kryminalnych. | Ten sam link co przy UCF-Crime |
| **UCF-101 / HMDB** | Klipy z codziennych czynności (spacer, rozmowa, sport bez przemocy). | UCF-101, HMDB-51 — standardowe zbiory do action recognition |
| **Własne nagrania** | Korytarze, ulica, pomieszczenia bez ognia, dymu i bójek. | — |

**Uwaga:** Ważne, żeby w danych „brak zagrożenia” było wystarczająco dużo różnorodnych scen (inaczej model może nadmiernie przypisywać tę klasę).


### Podsumowanie wyboru danych

- **Bójka:** UCF-Crime (Fighting, Assault) + ewentualnie CCTV-Fights lub RWF-2000.
- **Pożar / dym:** FiSmo lub Fire-and-Smoke (Kaggle/GitHub); ewentualnie D-Fire dla obrazów.
- **Palenie:** zbiory „smoke” + własne lub znalezione klipy z paleniem.
- **Brak zagrożenia:** UCF-Crime Normal + ewentualnie fragmenty UCF-101 / własne.

Po zebraniu plików wideo: podział na klatki (OpenCV), zapis w katalogach `data_video/bojka/`, `data_video/pozar/`, `data_video/palenie/`, `data_video/brak_zagrozenia/` lub jeden plik CSV/JSON z ścieżkami i etykietami (oraz przedziałami czasowymi, jeśli są).


## Proponowana struktura katalogów (docelowo)

uni_project/
├── uni_project.py          # bez zmian — demo CIFAR + zdjęcia
├── requirements.txt        # rozszerzyć: opencv-python, fastapi, uvicorn
├── ROADMAP_PROJEKT.md      # ten plik
│
├── danger_detection/       # nowy moduł projektu „niebezpieczne sytuacje”
│   ├── __init__.py
│   ├── config.py           # stałe: klasy, rozmiar klatki, ścieżki, próg %
│   ├── video_utils.py      # wideo → klatki, wideo → segmenty (OpenCV)
│   ├── dataset.py          # Dataset sekwencji klatek (PyTorch)
│   ├── model_cnn_lstm.py   # CNN + LSTM + klasyfikator
│   ├── train.py            # pętla treningu (albo skrypt do uruchomienia)
│   ├── analyze.py          # analyze_video() → przedziały czasowe
│   └── checkpoints/        # model_danger.pth
│
├── backend/                # FastAPI
│   ├── main.py             # app, endpoint POST /analyze, GET /health
│   └── ...
│
└── frontend/               # strona WWW
    ├── index.html
    ├── style.css
    └── (opcjonalnie) app.js