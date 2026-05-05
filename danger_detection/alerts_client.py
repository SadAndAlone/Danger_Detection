"""
Клиент для общения с бекендом алертов (FastAPI Особы 2).

Используется для:
- создания записи об угрозе (alert) на сервере,
- получения списка FCM‑токенов устройств.

Пример запуска теста:
  python -m danger_detection.alerts_client

Проверка загрузки видео (ответ сервера, статус, тело):
  python -m danger_detection.alerts_client upload path\\to\\clip.mp4

Полная цепочка как в live_camera (upload файла + POST /api/alerts):
  python -m danger_detection.alerts_client sendclip path\\to\\clip.mp4 [fire|fight|smoke]
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional

import requests


# Базовый URL бекенда.
#
# Сейчас для примера стоит локальный адрес. На реальной защите,
# когда вы будете в одной сети, сюда надо будет подставить IP ноутбука,
# на котором запущен FastAPI Особы 2, например:
#   BASE_URL = "http://192.168.0.15:8000"
BASE_URL = "https://threatalertsbackend.onrender.com"

# API token (tenant). Не хардкодим в репозиторий — задавайте через переменную окружения.
# PowerShell (пример):
#   $env:THREAT_ALERTS_API_TOKEN="..."; python -m danger_detection.alerts_client
API_TOKEN = os.getenv("THREAT_ALERTS_API_TOKEN", "").strip()


def _auth_headers() -> dict:
    if not API_TOKEN:
        raise RuntimeError(
            "Не задан THREAT_ALERTS_API_TOKEN. "
            "Нужен заголовок Authorization: Bearer <token>, чтобы бекенд привязал alert к tenant."
        )
    return {"Authorization": f"Bearer {API_TOKEN}"}


def create_alert(threat_type: str, video_path: str) -> int:
    """
    Создаёт алерт на бекенде и возвращает его id.

    :param threat_type: тип угрозы: "fire", "fight" или "smoke"
    :param video_path: относительный путь к mp4 на сервере,
                       например "alerts/2026-03-16/14-32-05_fire_10s.mp4"
    """
    detected_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "threat_type": threat_type,
        "detected_at": detected_at,
        "video_path": video_path,
    }

    resp = requests.post(
        f"{BASE_URL}/api/alerts",
        json=payload,
        headers=_auth_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def diagnose_video_upload(local_path: Path) -> None:
    """
    Wypisuje status odpowiedzi serwera przy uploadzie (do debugowania API).
    Uruchomienie: ``python -m danger_detection.alerts_client upload <ścieżka.mp4>``
    """
    if not API_TOKEN:
        print("THREAT_ALERTS_API_TOKEN is not set.")
        return
    local_path = Path(local_path)
    if not local_path.is_file():
        print("File not found:", local_path)
        return
    upload_url = os.getenv("ALERTS_VIDEO_UPLOAD_URL", "").strip() or f"{BASE_URL}/api/alerts/upload"
    print("POST", upload_url)
    print("field: file =", local_path.name)
    try:
        with open(local_path, "rb") as f:
            resp = requests.post(
                upload_url,
                files={"file": (local_path.name, f, "video/mp4")},
                headers=_auth_headers(),
                timeout=180,
            )
        print("status:", resp.status_code)
        ct = resp.headers.get("content-type", "")
        print("content-type:", ct)
        body = resp.text
        if len(body) > 800:
            print("body (truncated):", body[:800], "...")
        else:
            print("body:", body)
        if resp.ok:
            try:
                data = resp.json()
                ref = data.get("video_path") or data.get("path") or data.get("url")
                print("parsed server ref:", ref)
            except ValueError:
                print("(response is not JSON)")
    except requests.RequestException as exc:
        print("request error:", exc)


def try_upload_alert_video(local_path: Path) -> Optional[str]:
    """
    Próbuje wysłać plik mp4 na backend (multipart).
    URL: zmienna ALERTS_VIDEO_UPLOAD_URL lub domyślnie ``{BASE_URL}/api/alerts/upload``.
    Oczekiwana odpowiedź JSON z polem ``video_path`` lub ``path`` lub ``url``.
    """
    if not API_TOKEN:
        return None
    local_path = Path(local_path)
    upload_url = os.getenv("ALERTS_VIDEO_UPLOAD_URL", "").strip() or f"{BASE_URL}/api/alerts/upload"
    try:
        with open(local_path, "rb") as f:
            resp = requests.post(
                upload_url,
                files={"file": (local_path.name, f, "video/mp4")},
                headers=_auth_headers(),
                timeout=180,
            )
        if resp.status_code not in (200, 201):
            return None
        data = resp.json()
        ref = data.get("video_path") or data.get("path") or data.get("url")
        return str(ref) if ref else None
    except (OSError, requests.RequestException, ValueError):
        return None


def create_alert_with_local_clip(threat_type: str, clip_path: Path) -> Optional[int]:
    """
    Zapisany lokalnie klip: próba uploadu, potem ``create_alert`` z ścieżką serwera lub ``client_clip:...``.
    Bez ``THREAT_ALERTS_API_TOKEN`` tylko log — bez wyjątku.
    """
    clip_path = Path(clip_path)
    if not clip_path.is_file():
        raise FileNotFoundError(str(clip_path))
    if not API_TOKEN:
        print("THREAT_ALERTS_API_TOKEN not set — clip saved locally, API skipped.")
        return None
    server_ref = try_upload_alert_video(clip_path)
    if server_ref:
        print(f"[alerts] upload OK — video_path в алерте: {server_ref}")
    else:
        print(
            "[alerts] upload не удался или сервер не вернул path — в алерт уйдёт заглушка; "
            f"GET .../video даст 404 (файла clips/... на сервере нет): client_clip:{clip_path.name}"
        )
    video_path_str = server_ref if server_ref else f"client_clip:{clip_path.name}"
    return create_alert(threat_type, video_path_str)


def get_device_tokens() -> List[str]:
    """
    Возвращает список FCM‑токенов устройств из бекенда.
    """
    resp = requests.get(
        f"{BASE_URL}/api/device/tokens",
        headers=_auth_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    tokens = resp.json()
    return list(tokens)


if __name__ == "__main__":
    import sys

    # python -m danger_detection.alerts_client upload path\to\clip.mp4
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "upload":
        diagnose_video_upload(Path(sys.argv[2]))
        sys.exit(0)

    # python -m danger_detection.alerts_client sendclip clip.mp4 [fire|fight|smoke]
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "sendclip":
        clip = Path(sys.argv[2])
        threat = sys.argv[3].lower() if len(sys.argv) >= 4 else "fire"
        if threat not in ("fire", "fight", "smoke"):
            print("threat_type must be fire, fight, or smoke")
            sys.exit(1)
        try:
            aid = create_alert_with_local_clip(threat, clip)
            if aid is None:
                print("No alert id (missing token or upload+alert failed); see messages above.")
            else:
                print("Alert created, id =", aid)
        except Exception as exc:
            print("Error:", exc)
        sys.exit(0)

    # Простой тест без модели и камеры.
    # Важно: к этому моменту бекенд Особы 2 должен быть запущен
    # и доступен по адресу BASE_URL.
    try:
        fake_video_path = "alerts/test_fire_10s.mp4"
        alert_id = create_alert("fire", fake_video_path)
        print("Создан alert_id =", alert_id)
        print(
            "(Без загрузки файла: в БД только путь-заглушка. "
            "Чтобы отправить реальное видео: python -m danger_detection.alerts_client sendclip путь\\к.mp4)"
        )

        tokens = get_device_tokens()
        print("FCM tokens:", tokens)
    except Exception as exc:
        print("Ошибка при обращении к бекенду алертов:", exc)

