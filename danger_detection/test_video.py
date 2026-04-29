"""
Prosty test: wczytaj wideo, wyciągnij klatki, wypisz liczbę klatek i informacje.
Opcjonalnie zapisz klatki do folderu, żeby je podejrzeć.

Uruchomienie:
  python -m danger_detection.test_video ścieżka/do/wideo.mp4
  python -m danger_detection.test_video ścieżka/do/wideo.mp4 output_frames
"""
import sys
from pathlib import Path

import cv2

from danger_detection.video_utils import video_to_frames, get_video_info, extract_segments
from danger_detection.config import IMG_HEIGHT, IMG_WIDTH, SEGMENT_DURATION_SEC, EXTRACT_FPS


def main():
    if len(sys.argv) < 2:
        print("Użycie: python -m danger_detection.test_video <ścieżka_do_wideo> [folder_na_klatki]")
        print("Przykład: python -m danger_detection.test_video film.mp4")
        print("          python -m danger_detection.test_video film.mp4 output_frames  <- zapisze klatki do folderu")
        return

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Plik nie istnieje: {video_path}")
        return

    save_frames_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Wideo: {video_path}")
    info = get_video_info(video_path)
    print(f"  FPS: {info['fps']:.1f}, klatek: {info['frame_count']}, czas: {info['duration_sec']:.1f} s")

    frames = video_to_frames(
        video_path,
        every_n_frames=max(1, int(info["fps"] // EXTRACT_FPS)),
        resize=(IMG_HEIGHT, IMG_WIDTH),
    )
    print(
        f"  Wyciągnięte klatki (co ~{EXTRACT_FPS} fps, rozmiar {IMG_WIDTH}x{IMG_HEIGHT}): {len(frames)}"
    )

    if save_frames_dir is not None:
        save_frames_dir = Path(save_frames_dir)
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            path = save_frames_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
        print(f"  Zapisano {len(frames)} klatek do: {save_frames_dir.absolute()}")

    segments = extract_segments(
        video_path,
        segment_duration_sec=SEGMENT_DURATION_SEC,
        fps=EXTRACT_FPS,
        resize=(IMG_HEIGHT, IMG_WIDTH),
    )
    print(f"  Segmenty (po {SEGMENT_DURATION_SEC} s): {len(segments)}")
    if segments:
        start, end, seg_frames = segments[0]
        print(f"  Pierwszy segment: {start:.1f}s - {end:.1f}s, klatek w segmencie: {len(seg_frames)}")

    print("\nPipeline wideo -> klatki działa poprawnie.")


if __name__ == "__main__":
    main()
