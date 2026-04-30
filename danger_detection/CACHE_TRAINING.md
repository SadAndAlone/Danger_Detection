## Fast training at 720p: feature cache

Training end-to-end from raw mp4 at **1280×720** is very slow because each dataset sample
does: open video → seek → decode frames. This file describes the faster workflow.

### Step 0: make sure CUDA works (optional but recommended)

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### Step 1: precompute cached features (one-time)

From project root:

```powershell
python -m danger_detection.precompute_feature_cache --cache_dir cache_features
```

This creates files:

```
cache_features/<class>/seg_00000000.pt
cache_features/<class>/seg_00000001.pt
...
```

Each file stores compact tensor `x: (SEQ_LEN, 256)` and label `y`.

### Step 2: train from cache (fast)

Set env var and run training:

```powershell
$env:FEATURE_CACHE_DIR="cache_features"
python -m danger_detection.train
```

`train.py` will automatically switch into cache mode and train only the LSTM+classifier.

### Notes
- Cache size is small compared to caching full 720p frames.
- If you change class folders or FPS/SEGMENT settings, rebuild the cache.
- This mode is meant to speed up experiments. For best final quality you can later
  fine-tune end-to-end on a smaller subset (or a smaller resolution).

