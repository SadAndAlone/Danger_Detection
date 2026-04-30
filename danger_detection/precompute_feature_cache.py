"""
Precompute cache of CNN features for each video segment.

Why: training from raw mp4 at 720p is very slow because each sample requires
open/seek/decode. This script runs decoding once and stores compact tensors:
  x: (SEQ_LEN, 256) features per frame
  y: class index

Usage (from project root):
  python -m danger_detection.precompute_feature_cache --cache_dir cache_features

Then train from cache:
  set FEATURE_CACHE_DIR=cache_features  (PowerShell: $env:FEATURE_CACHE_DIR="cache_features")
  python -m danger_detection.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import DATA_VIDEO_DIR, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH
from .dataset import DangerVideoDataset, frames_to_tensor
from .model_cnn_lstm import CNNFeatureExtractor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_VIDEO_DIR),
        help="Root directory with class folders (data_video/...).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_features",
        help="Where to store cached feature tensors.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for cached features.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of segments to cache (0 = no limit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for CNNFeatureExtractor init (must match between cache and inference).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Data:   {data_dir}")
    print(f"Cache:  {cache_dir}")
    print(f"Input:  {IMG_WIDTH}x{IMG_HEIGHT}, SEQ_LEN={SEQ_LEN}")

    dataset = DangerVideoDataset(root=data_dir)
    print(f"Classes: {dataset.class_names}")
    print(f"Segments: {len(dataset)}")

    torch.manual_seed(args.seed)
    extractor = CNNFeatureExtractor().to(DEVICE)
    extractor.eval()

    # Save extractor weights used to produce the cache (needed for live inference)
    extractor_path = cache_dir / "extractor_state_dict.pth"
    torch.save({"state_dict": extractor.state_dict(), "seed": int(args.seed)}, extractor_path)
    print(f"Saved extractor: {extractor_path}")

    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    n = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    for i in range(n):
        x_img, y = dataset[i]  # (T, 3, H, W) float32 normalized
        x_img = x_img.to(DEVICE)  # (T, 3, H, W)
        feats = extractor(x_img)  # (T, 256)
        feats = feats.to(dtype=out_dtype).cpu()

        # Store in cache/<class>/<idx>.pt (idx stable for current dataset ordering)
        class_name = dataset.class_names[y]
        out_dir = cache_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seg_{i:08d}.pt"
        torch.save(
            {
                "x": feats,
                "y": int(y),
                "classes": dataset.class_names,
            },
            out_path,
        )

        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"Cached {i+1}/{n} segments...")

    print("Done.")


if __name__ == "__main__":
    main()

