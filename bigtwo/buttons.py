"""大老二：畫面上的按鈕（依錄影量的位置）。

輪到自己時手牌上方出現「刁牌／重選／PASS／出牌」：能按的是藍色（重選、PASS）或橘色（出牌），
不能按的是暗色底、灰字（例如還沒選牌時的出牌）。開局換三張時有「不換」（橘）與「換牌」（選了牌才變橘）。

用按鈕整塊（含文字）的樣板比對，`train/bigtwo/buttons/<名稱>_<on|off>.png`。原本只看顏色比例，
錄影裡打牌途中的大動畫（大老二、老二對子）有 6 次被當成換三張、結算畫面被當成輪到自己——
自動模式會因此亂點牌（10/16 20:11 實戰換三張被打亂）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .screen import read_image

BUTTON_DIR = Path(__file__).resolve().parent.parent / "train" / "bigtwo" / "buttons"

# 名稱 → (x0, y0, x1, y1)，點擊點取中心
BUTTONS = {          # Google Play 版（全螢幕 1920x1080）
    "刁牌": (225, 625, 480, 715),
    "重選": (577, 625, 840, 715),
    "PASS": (933, 625, 1200, 715),
    "出牌": (1287, 625, 1590, 715),
    "不換": (570, 570, 873, 672),
    "換牌": (1050, 570, 1350, 672),
}
SHIFT = 8              # 比對時容許位置偏差（px）
THRESHOLD = 0.8


def center(name: str) -> tuple[int, int]:
    x0, y0, x1, y1 = BUTTONS[name]
    return (x0 + x1) // 2, (y0 + y1) // 2


@lru_cache(maxsize=2)
def load_buttons(directory: Path = BUTTON_DIR) -> dict[str, list[tuple[str, np.ndarray]]]:
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for path in sorted(directory.glob("*.png")):
        name, state = path.stem.rsplit("_", 1)
        state = state.rstrip("0123456789")        # on2：游標停在上面時顏色比較淺
        out.setdefault(name, []).append((state, read_image(path)))
    return out


def button_scores(image: np.ndarray, name: str, directory: Path = BUTTON_DIR) -> dict[str, float]:
    x0, y0, x1, y1 = BUTTONS[name]
    area = image[max(0, y0 - SHIFT):y1 + SHIFT, max(0, x0 - SHIFT):x1 + SHIFT]
    scores = {}
    for state, template in load_buttons(directory).get(name, []):
        if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
            continue
        score = float(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED).max())
        scores[state] = max(score, scores.get(state, -1.0))
    return scores


def button_state(image: np.ndarray, name: str, directory: Path = BUTTON_DIR) -> str | None:
    """"on"（可以按）、"off"（出現但暗掉）、None（沒出現）。"""
    scores = button_scores(image, name, directory)
    if not scores:
        return None
    state, score = max(scores.items(), key=lambda kv: kv[1])
    return state if score >= THRESHOLD else None


def my_turn(image: np.ndarray) -> bool:
    return button_state(image, "出牌") is not None


def swap_phase(image: np.ndarray) -> bool:
    return button_state(image, "不換") == "on"
