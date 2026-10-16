"""大老二：讀自己的手牌（畫面下方）。

每張牌只看左上角的點數與花色（「墨水」遮罩：深色或紅色的像素），不管牌怎麼排：
一般排法間距 131.4px、置中；選起的牌上移 53px（上緣 y 695，平常 748）；
「刁牌」（自己拖曳整理）時間距不固定，所以用點數模板在整條手牌區滑動找，不寫死格線。

模板放 `train/bigtwo/templates/rank/<點數>/*.png`、`suit/<花色>/*.png`，用
`python -m bigtwo.hand_reader <截圖> "<由左到右的手牌>"` 從一般排法的截圖裁出。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .cards import RANK_CHARS, SUIT_CHARS, card, card_code, parse_cards
from .screen import read_image, write_image

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "train" / "bigtwo" / "templates"

SPACING = 131.4
CARD_WIDTH = 150
TOP_NORMAL = 748
TOP_RAISED = 695
RANK_BOX = (2, 8, 60, 66)      # 相對牌左上角 (x0, y0, x1, y1)
SUIT_BOX = (4, 70, 52, 112)
SEARCH_Y = (TOP_RAISED + RANK_BOX[1] - 8, TOP_NORMAL + RANK_BOX[1] + 8)
RANK_THRESHOLD = 0.8             # 正常 ≥0.87（變暗的 2）；被動畫蓋一半的梅花 J 誤認成 5 是 0.71
UNSURE_THRESHOLD = 0.6           # 介於兩者之間、旁邊又沒有認得的牌：可能是被蓋住的牌，整手不採用
SUIT_THRESHOLD = 0.85            # 正常 ≥0.97；被游標蓋住的梅花 J 0.61
MIN_GAP = 40                    # 兩張牌左上角至少差幾 px（刁牌時約 70）


def ink(image: np.ndarray) -> np.ndarray:
    """比附近牌面暗很多的像素＝1（以 BGR 最小值比較，紅字與黑字都算）。
    輪到自己時不能出的牌會整張變暗（白 255→109），所以和周圍最亮處相比、不用固定門檻。"""
    darkest = image.min(axis=2).astype(np.float32)
    background = cv2.dilate(darkest, np.ones((21, 21), np.uint8))
    return ((darkest < 0.6 * background) & (background > 80)).astype(np.float32)


def is_red(image: np.ndarray) -> bool:
    p = image.astype(np.int16)
    b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    glyph = ink(image) > 0
    red = (glyph & (r > 2 * np.maximum(g, b)) & (r - g > 30)).sum()
    return red > 0.5 * glyph.sum()


def crop(image, x: int, top: int, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[top + y0:top + y1, x + x0:x + x1]


def uniform_positions(image: np.ndarray) -> list[tuple[int, int]]:
    """一般排法（置中、等距）時每張牌的左上角；排法不符合時回傳 []。"""
    white = image.min(axis=2) > 200
    band = white[760:880].mean(axis=0)
    cols = np.where(band > 0.4)[0]
    if len(cols) == 0:
        return []
    left, right = int(cols.min()), int(cols.max())
    n = (right - left + 1 - CARD_WIDTH) / SPACING + 1
    if abs(n - round(n)) > 0.15 or not 1 <= round(n) <= 13:
        return []
    out = []
    for i in range(round(n)):
        x = int(round(left + SPACING * i))
        raised = white[TOP_RAISED + 5:TOP_RAISED + 45, x + 10:x + 120].mean() > 0.8
        out.append((x, TOP_RAISED if raised else TOP_NORMAL))
    return out


@lru_cache(maxsize=4)
def load_templates(directory: Path = TEMPLATE_DIR):
    ranks, suits = [], []
    for kind, out in (("rank", ranks), ("suit", suits)):
        for folder in sorted((directory / kind).glob("*")):
            for path in sorted(folder.glob("*.png")):
                out.append((folder.name, ink(read_image(path))))
    return ranks, suits


@dataclass(frozen=True)
class HandCard:
    card: int
    x: int
    raised: bool
    score: float
    suit_score: float = 1.0


def _best(patch: np.ndarray, templates) -> tuple[str | None, float]:
    best, name = -1.0, None
    for label, template in templates:
        if patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]:
            continue
        score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED).max())
        if score > best:
            best, name = score, label
    return name, best


def _looks_like_card(image: np.ndarray, x: int, y: int) -> bool:
    """點數框位置在牌面的白底上（上緣剛好是平常或選起的高度）。"""
    top = y - RANK_BOX[1]
    if min(abs(top - TOP_NORMAL), abs(top - TOP_RAISED)) > 4:
        return False
    face = image[top + 2:top + 70, x + 60:x + 110]
    return face.size > 0 and (face.min(axis=2) > 90).mean() > 0.5   # 變暗的牌面約 109


def read_hand(image: np.ndarray, directory: Path = TEMPLATE_DIR) -> list[HandCard] | None:
    """由左到右的手牌；有認不出的牌時回傳 None。"""
    ranks, suits = load_templates(directory)
    if not ranks or not suits:
        raise FileNotFoundError(f"沒有大老二模板：{directory}")
    y0, y1 = SEARCH_Y
    th = ranks[0][1].shape[0]
    strip = ink(image[y0:y1 + th + 4])
    best = best_label = None
    for label, template in ranks:
        result = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        rank = RANK_CHARS.index(label)
        if best is None:
            best = result.copy()
            best_label = np.full(result.shape, rank, dtype=np.int16)
            continue
        better = result > best
        best[better] = result[better]
        best_label[better] = rank
    column = best.max(axis=0)
    rows = best.argmax(axis=0)
    peaks = []
    for x in np.argsort(-column):
        score = float(column[x])
        if score < UNSURE_THRESHOLD:
            break
        if all(abs(int(x) - px) >= MIN_GAP for px, *_ in peaks):
            if score >= RANK_THRESHOLD:
                peaks.append((int(x), int(rows[x]), score))
            elif _looks_like_card(image, int(x), y0 + int(rows[x])):
                return None
    cards = []
    for px, py, score in sorted(peaks):
        left = px - RANK_BOX[0]
        top = y0 + py - RANK_BOX[1]
        rank = int(best_label[py, px])
        suit_patch = image[top + SUIT_BOX[1] - 6:top + SUIT_BOX[3] + 6,
                           left + SUIT_BOX[0] - 6:left + SUIT_BOX[2] + 6]
        red = is_red(crop(image, left, top, RANK_BOX))
        candidates = [(s, t) for s, t in suits if (s in "dh") == red]
        suit, suit_score = _best(ink(suit_patch), candidates)
        if suit is None or suit_score < SUIT_THRESHOLD:
            return None
        cards.append(HandCard(card(rank, SUIT_CHARS.index(suit)), left,
                              abs(top - TOP_RAISED) < abs(top - TOP_NORMAL), score, suit_score))
    return cards


def extract_templates(image: np.ndarray, labels: str, directory: Path = TEMPLATE_DIR,
                      tag: str = "") -> int:
    cards = parse_cards(labels)
    positions = uniform_positions(image)
    if len(positions) != len(cards):
        raise ValueError(f"畫面上找到 {len(positions)} 張，標註 {len(cards)} 張")
    for (x, top), c in zip(positions, cards):
        code = card_code(c)
        write_image(directory / "rank" / code[0] / f"{code}{tag}.png", crop(image, x, top, RANK_BOX))
        write_image(directory / "suit" / code[1] / f"{code}{tag}.png", crop(image, x, top, SUIT_BOX))
    load_templates.cache_clear()
    return len(cards)


if __name__ == "__main__":
    path, labels = sys.argv[1], sys.argv[2]
    n = extract_templates(read_image(path), labels, tag="_" + Path(path).stem)
    print(f"裁出 {n} 張的點數與花色模板")
