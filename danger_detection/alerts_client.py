"""
Клиент для общения с бекендом алертов (FastAPI Особы 2).

Используется для:
- создания записи об угрозе (alert) на сервере,
- получения списка FCM‑токенов устройств.

Пример запуска теста:
  python -m danger_detection.alerts_client
"""

from datetime import datetime, timezone
import os
from typing import List

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
    # Простой тест без модели и камеры.
    # Важно: к этому моменту бекенд Особы 2 должен быть запущен
    # и доступен по адресу BASE_URL.
    try:
        fake_video_path = "alerts/test_fire_10s.mp4"
        alert_id = create_alert("fire", fake_video_path)
        print("Создан alert_id =", alert_id)

        tokens = get_device_tokens()
        print("FCM tokens:", tokens)
    except Exception as exc:
        print("Ошибка при обращении к бекенду алертов:", exc)

