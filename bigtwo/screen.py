"""大老二用的截圖與圖片讀寫（與麻將的 perception/ 分開，見 CLAUDE.md）。

座標以錄影畫面為準：明星3缺1 視窗模式、1920x1080 螢幕、視窗在左上角（標題列 y 0-23、
遊戲畫面約 y 90-965、下方是 Windows 工作列）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread 在 Windows 讀不了中文路徑，改用 imdecode。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"無法讀取圖片：{path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"無法編碼圖片：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def grab_screen(monitor: int = 1) -> np.ndarray:
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    with factory() as sct:
        shot = np.asarray(sct.grab(sct.monitors[monitor]))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def video_frame(path: str | Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"讀不到 {path} 第 {seconds} 秒")
    return frame
