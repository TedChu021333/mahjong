"""螢幕擷取與圖片讀寫（BGR numpy 陣列）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """讀取圖片；cv2.imread 在 Windows 無法處理中文路徑，改用 imdecode。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
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
    """擷取整個螢幕（預設主螢幕），回傳 BGR 圖片。"""
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss  # 新版 mss 改名為 MSS
    with factory() as sct:
        shot = np.asarray(sct.grab(sct.monitors[monitor]))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
