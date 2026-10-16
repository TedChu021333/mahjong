"""大老二：畫面上的按鈕（依錄影量的位置與顏色）。

輪到自己時手牌上方出現「刁牌／重選／PASS／出牌」：能按的是藍色（重選、PASS）或橘色（出牌），
不能按的是暗色底、灰字（例如還沒選牌時的出牌）。開局換三張時有「不換」（橘）與「換牌」（選了牌才變橘）。
"""
from __future__ import annotations

import numpy as np

# 名稱 → (x0, y0, x1, y1)，點擊點取中心
BUTTONS = {
    "刁牌": (365, 600, 560, 660),
    "重選": (655, 600, 850, 660),
    "PASS": (950, 600, 1140, 660),
    "出牌": (1240, 600, 1460, 660),
    "不換": (610, 545, 830, 630),
    "換牌": (1090, 545, 1310, 630),
}
ON_SHARE = 0.25      # 亮著的按鈕：橘／藍色像素比例（實測 0.33~0.66，其他畫面 ≤0.09）
OFF_GRAY = 0.06      # 暗掉的按鈕：灰字比例（出牌 0.11，其他畫面 ≤0.03）


def center(name: str) -> tuple[int, int]:
    x0, y0, x1, y1 = BUTTONS[name]
    return (x0 + x1) // 2, (y0 + y1) // 2


def _shares(image: np.ndarray, box) -> tuple[float, float, float]:
    x0, y0, x1, y1 = box
    p = image[y0:y1, x0:x1].astype(np.int16)
    b, g, r = p[..., 0], p[..., 1], p[..., 2]
    orange = ((r > 180) & (g > 70) & (g < 190) & (b < 90)).mean()
    blue = ((b > 150) & (r < 120) & (b > g + 30)).mean()
    gray = ((abs(r - g) < 18) & (abs(g - b) < 18) & (r > 110) & (r < 200)).mean()
    return float(orange), float(blue), float(gray)


def button_state(image: np.ndarray, name: str) -> str | None:
    """"on"（可以按）、"off"（出現但暗掉）、None（沒出現）。"""
    orange, blue, gray = _shares(image, BUTTONS[name])
    if max(orange, blue) > ON_SHARE:
        return "on"
    if gray > OFF_GRAY:
        return "off"
    return None


def my_turn(image: np.ndarray) -> bool:
    return button_state(image, "出牌") is not None


def swap_phase(image: np.ndarray) -> bool:
    return button_state(image, "不換") == "on"
