"""大老二：讀牌桌上四家最近打出的牌。

桌上的牌是手牌的 0.9 倍，五張牌型會散開成扇形（每張轉 −10°～+10°），所以點數模板縮小並轉幾個角度去找，
花色在點數下方（依角度轉過的位置）另外比對。四家的區域由錄影量出（自己的出牌在桌面正中間）。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np

from .cards import RANK_CHARS, SUIT_CHARS, card
from .combos import Combo, classify
from .hand_reader import RANK_BOX, SUIT_BOX, TEMPLATE_DIR, ink, is_red, load_templates

SCALE = 0.9
WORK = 0.5                       # 比對在半解析度做（約快 4 倍）
ANGLES = (-12, -8, -4, 0, 4, 8, 12)
RANK_THRESHOLD = 0.85            # 真的牌 ≥0.92；雜訊（其他圖案）最高 0.81
SUIT_THRESHOLD = 0.75
SURE_RANK = 0.9                  # 點數這麼像、花色卻認不出：整區回報讀不到
MIN_GAP = 22

# 座位：0 自己、1 下家（右）、2 對家（上）、3 上家（左）；(x0, y0, x1, y1)
PLAY_REGIONS = {
    0: (700, 410, 1250, 690),
    1: (1200, 330, 1720, 620),
    2: (700, 190, 1250, 410),
    3: (200, 330, 720, 620),
}

# 點數框中心到花色框中心的位移（手牌尺寸）
_RANK_CENTER = ((RANK_BOX[0] + RANK_BOX[2]) / 2, (RANK_BOX[1] + RANK_BOX[3]) / 2)
_SUIT_CENTER = ((SUIT_BOX[0] + SUIT_BOX[2]) / 2, (SUIT_BOX[1] + SUIT_BOX[3]) / 2)
_SUIT_OFFSET = (_SUIT_CENTER[0] - _RANK_CENTER[0], _SUIT_CENTER[1] - _RANK_CENTER[1])


@lru_cache(maxsize=4)
def _table_templates(directory=TEMPLATE_DIR):
    ranks, suits = load_templates(directory)

    def scale(items):
        return [(label, cv2.resize(mask, None, fx=SCALE * WORK, fy=SCALE * WORK,
                                          interpolation=cv2.INTER_AREA))
                for label, mask in items]

    return scale(ranks), scale(suits)


@dataclass(frozen=True)
class TableCard:
    card: int
    x: int          # 點數框中心（整張畫面座標）
    y: int
    angle: int
    score: float


def _best_map(mask: np.ndarray, templates):
    best = label = None
    for i, (_, template) in enumerate(templates):
        if mask.shape[0] < template.shape[0] or mask.shape[1] < template.shape[1]:
            return None, None
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        if best is None:
            best, label = result, np.zeros(result.shape, dtype=np.int16)
            continue
        better = result > best
        best = np.where(better, result, best)
        label[better] = i
    return best, label


def find_cards(image: np.ndarray, region, directory=TEMPLATE_DIR) -> list[TableCard] | None:
    """區域裡所有牌（由左到右）；有一張的花色認不出時回傳 None。

    把區域轉 ANGLES 各個角度（牌轉正）後用沒轉的模板比對，找到的位置再轉回原畫面。"""
    ranks, suits = _table_templates(directory)
    x0, y0, x1, y1 = region
    area = image[y0:y1, x0:x1]
    if (area.min(axis=2) > 200).mean() < 0.03:     # 沒有白色牌面
        return []
    area = cv2.resize(area, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_AREA)
    h, w = area.shape[:2]
    th, tw = ranks[0][1].shape
    candidates = []
    rotated = {}
    for angle in ANGLES:
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        turned = cv2.warpAffine(area, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        rotated[angle] = (turned, cv2.invertAffineTransform(matrix))
        best, label = _best_map(ink(turned), ranks)
        if best is None:
            return []
        ys, xs = np.where(best >= RANK_THRESHOLD)
        for y, x in zip(ys, xs):
            candidates.append((float(best[y, x]), int(x), int(y), angle, int(label[y, x])))
    candidates.sort(reverse=True)
    picked = []
    for score, x, y, angle, i in candidates:
        inverse = rotated[angle][1]
        cx, cy = x + tw / 2, y + th / 2
        ox = inverse[0, 0] * cx + inverse[0, 1] * cy + inverse[0, 2]
        oy = inverse[1, 0] * cx + inverse[1, 1] * cy + inverse[1, 2]
        if all((ox - px) ** 2 + (oy - py) ** 2 >= (MIN_GAP * WORK) ** 2 for px, py, *_ in picked):
            picked.append((ox, oy, score, angle, i, cx, cy))
    cards = []
    for ox, oy, score, angle, i, cx, cy in sorted(picked):
        turned = rotated[angle][0]
        sx, sy = cx + SCALE * WORK * _SUIT_OFFSET[0], cy + SCALE * WORK * _SUIT_OFFSET[1]
        sh, sw = suits[0][1].shape
        pad = 4
        window = turned[max(0, int(sy - sh / 2 - pad)):int(sy + sh / 2 + pad),
                        max(0, int(sx - sw / 2 - pad)):int(sx + sw / 2 + pad)]
        glyph = turned[int(cy - th / 2):int(cy + th / 2), int(cx - tw / 2):int(cx + tw / 2)]
        red = is_red(glyph)
        suit_mask = ink(window)
        suit, suit_score = None, -1.0
        for s, t in suits:
            if (s in "dh") != red or suit_mask.shape[0] < t.shape[0] or suit_mask.shape[1] < t.shape[1]:
                continue
            v = float(cv2.matchTemplate(suit_mask, t, cv2.TM_CCOEFF_NORMED).max())
            if v > suit_score:
                suit, suit_score = s, v
        if suit is None or suit_score < SUIT_THRESHOLD:
            if score < SURE_RANK:
                continue        # 點數也不太像：當成雜訊
            return None
        rank = ranks[i][0]
        cards.append(TableCard(card(RANK_CHARS.index(rank), SUIT_CHARS.index(suit)),
                               int(x0 + ox / WORK), int(y0 + oy / WORK), angle, score))
    return cards


def read_plays(image: np.ndarray, directory=TEMPLATE_DIR) -> dict[int, Combo | None | str]:
    """四家桌上的牌：Combo、None（沒有牌）或 "?"（有牌但讀不出／不是合法牌型）。"""
    out: dict[int, Combo | None | str] = {}
    for seat, region in PLAY_REGIONS.items():
        found = find_cards(image, region, directory)
        if found is None:
            out[seat] = "?"
        elif not found:
            out[seat] = None
        else:
            combo = classify([c.card for c in found])
            out[seat] = combo if combo is not None else "?"
    return out
