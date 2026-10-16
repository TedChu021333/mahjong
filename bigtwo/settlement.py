"""大老二：認結算畫面（Google Play 版）。

每局結束的小結算列出四家剩的牌與輸贏（第一列固定是自己），右邊有「存到名片」相機按鈕，
用它判斷（`train/bigtwo/screens/結算.png`）。輸贏金額之後由戰績統計讀。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .screen import read_image

SCREEN_DIR = Path(__file__).resolve().parent.parent / "train" / "bigtwo" / "screens"
SETTLEMENT_REGION = (1700, 620, 1910, 830)     # 「存到名片」按鈕附近
THRESHOLD = 0.85


@lru_cache(maxsize=2)
def _template(directory: Path = SCREEN_DIR) -> np.ndarray:
    return read_image(directory / "結算.png")


def is_settlement(image: np.ndarray, directory: Path = SCREEN_DIR) -> bool:
    x0, y0, x1, y1 = SETTLEMENT_REGION
    score = cv2.matchTemplate(image[y0:y1, x0:x1], _template(directory), cv2.TM_CCOEFF_NORMED).max()
    return float(score) >= THRESHOLD
