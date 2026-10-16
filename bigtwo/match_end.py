"""大老二：一場（4 局）打完後的畫面，自動模式按掉才能連續打。

- 社團廣告（「閃耀卡社團換卡片」，右上紅色 X）：蓋在大結算上，先按 X 關掉。
- 大結算右下「再玩一局 (倒數)」：只比對「再玩一局」四個字（倒數會變），認到就按。
都用參考圖辨識、不盲點座標（`train/bigtwo/screens/`）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .screen import read_image

SCREEN_DIR = Path(__file__).resolve().parent.parent / "train" / "bigtwo" / "screens"
# 名稱 → (參考圖, 搜尋區域, 點擊點)
TARGETS = {
    "廣告X": ("廣告X.png", (1480, 90, 1650, 250), (1561, 166)),
    "再玩一局": ("再玩一局.png", (1400, 870, 1740, 1005), (1600, 937)),
}
ORDER = ("廣告X", "再玩一局")      # 廣告蓋著時先關廣告
THRESHOLD = 0.85


@lru_cache(maxsize=4)
def _template(name: str, directory: Path = SCREEN_DIR) -> np.ndarray:
    return read_image(directory / TARGETS[name][0])


def score(image: np.ndarray, name: str, directory: Path = SCREEN_DIR) -> float:
    _, (x0, y0, x1, y1), _ = TARGETS[name]
    area, template = image[y0:y1, x0:x1], _template(name, directory)
    if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED).max())


def find_continue(image: np.ndarray, directory: Path = SCREEN_DIR) -> tuple[str, tuple[int, int]] | None:
    """要按的（名稱, 點擊點）；沒有就 None。"""
    for name in ORDER:
        if score(image, name, directory) >= THRESHOLD:
            return name, TARGETS[name][2]
    return None
