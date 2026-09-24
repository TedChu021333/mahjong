"""手牌以外的畫面資訊：吃碰槓聽胡按鈕、可吃碰的那張牌、金牌面板、換三張。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.config import (
    ACTION_BUTTON_CENTERS,
    CANCEL_BUTTON_REGION,
    CHOW_PANEL_ROWS,
    CHOW_PANEL_WIDTH,
    CLAIM_TILE_MARGINS,
    CLAIM_TILE_POSITIONS,
    CLAIM_TILE_SIZE,
    GOLD_MAX_SLOTS,
    GOLD_SLOT_ORIGIN,
    GOLD_SLOT_PITCH,
    GOLD_SLOT_SIZE,
    SWAP_CONFIRM_BUTTON,
    SWAP_KEEP_BUTTON,
)
from perception.hand_reader import TileClassifier

BUTTON_HALF_SIZE = (25, 28)
BUTTON_LIT_VALUE = 120
"""亮的按鈕亮度遠高於此值，變暗的按鈕在此之下（影片量測：亮 1.0、暗 ≤0.16）。"""
CANCEL_MIN_RED = 0.42
TILE_FACE_MIN_RATIO = 0.6


def _hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _face_ratio(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    hsv = _hsv(image[top:bottom, left:right])
    return float(((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).mean())


def read_buttons(image: np.ndarray) -> frozenset[str] | None:
    """按鈕列沒出現時回傳 None；否則回傳可以按的動作。"""
    left, top, right, bottom = CANCEL_BUTTON_REGION
    ring = _hsv(image[top:bottom, left:right])
    red = ((ring[..., 0] < 10) | (ring[..., 0] > 170)) & (ring[..., 1] > 150) & (ring[..., 2] > 150)
    if float(red.mean()) < CANCEL_MIN_RED:
        return None
    half_x, half_y = BUTTON_HALF_SIZE
    lit = set()
    for name, (x, y) in ACTION_BUTTON_CENTERS.items():
        value = _hsv(image[y - half_y:y + half_y, x - half_x:x + half_x])[..., 2]
        if float((value > BUTTON_LIT_VALUE).mean()) > 0.5:
            lit.add(name)
    # 全亮代表不是按鈕列（例如配桌畫面同位置剛好有紅色按鈕）
    return frozenset(lit) if len(lit) < len(ACTION_BUTTON_CENTERS) else None


@dataclass(frozen=True)
class ClaimTile:
    tile: int | None
    score: float = field(compare=False)
    """比對分數每幀有雜訊，不參與相等比較（提示模式靠相等判斷畫面是否穩定）。"""
    source: str = "left"
    """"left"（上家）或 "right"（下家）。"""


def read_claim_tile(image: np.ndarray, classifier: TileClassifier) -> ClaimTile | None:
    """讀出圓圈裡放大顯示、可以吃碰胡的那張牌。"""
    width, height = CLAIM_TILE_SIZE
    pad_left, pad_top, pad_right, pad_bottom = CLAIM_TILE_MARGINS
    for source, (x, y) in CLAIM_TILE_POSITIONS.items():
        if _face_ratio(image, (x + 10, y + 10, x + width - 10, y + 30)) < TILE_FACE_MIN_RATIO:
            continue
        crop = image[y - pad_top:y + height + pad_bottom, x - pad_left:x + width + pad_right]
        tile, score = classifier.classify(crop)
        return ClaimTile(tile, score, source)
    return None


def gold_slot_box(index: int) -> tuple[int, int, int, int]:
    x0, y0 = GOLD_SLOT_ORIGIN
    pitch_x, pitch_y = GOLD_SLOT_PITCH
    width, height = GOLD_SLOT_SIZE
    left = round(x0 + pitch_x * (index % 4))
    top = round(y0 + pitch_y * (index // 4))
    return (left, top, left + width, top + height)


def read_gold_tiles(image: np.ndarray, gold_classifier: TileClassifier) -> list[int | None]:
    """讀出金牌面板；gold_classifier 需以 GOLD_TEMPLATE_BOX 建立。"""
    tiles = []
    for index in range(GOLD_MAX_SLOTS):
        box = gold_slot_box(index)
        left, top, right, bottom = box
        # 牌面上方留白處檢查有沒有牌
        if _face_ratio(image, (left + 6, top + 2, right - 6, top + 8)) < TILE_FACE_MIN_RATIO:
            break
        tile, _ = gold_classifier.classify(image[top:bottom, left:right])
        tiles.append(tile)
    return tiles


def _orange_ratio(image: np.ndarray, center: tuple[int, int]) -> float:
    """量按鈕左側（避開中間的白字）的橘色比例。"""
    x, y = center
    hsv = _hsv(image[y - 40:y + 40, x - 120:x - 70])
    orange = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
    return float(orange.mean())


def read_swap_prompt(image: np.ndarray) -> bool | None:
    """換三張畫面：不在選牌時回傳 None；否則回傳「換牌」按鈕是否可按。"""
    if _orange_ratio(image, SWAP_KEEP_BUTTON) < 0.5:
        return None
    return _orange_ratio(image, SWAP_CONFIRM_BUTTON) >= 0.5


def read_chow_panels(image: np.ndarray) -> tuple[tuple[int, int], ...]:
    """多種吃法的選項框中心，由左到右；沒有選項框時回傳空 tuple。

    以「一段連續的白色牌面欄位」找框：框內三張牌之間的縫很窄，框與框之間
    隔著較寬的深色間距。"""
    top, bottom = CHOW_PANEL_ROWS
    hsv = _hsv(image[top:bottom])
    columns = (((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).mean(axis=0) > 0.5).tolist()
    panels = []
    start = end = None
    gap = 0
    for x, white in enumerate(columns + [False] * 16):
        if white:
            start = x if start is None else start
            end, gap = x, 0
        elif start is not None:
            gap += 1
            if gap > 15:
                if CHOW_PANEL_WIDTH[0] <= end - start <= CHOW_PANEL_WIDTH[1]:
                    panels.append(((start + end) // 2, (top + bottom) // 2))
                start = None
    return tuple(panels)
