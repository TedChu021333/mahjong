"""大老二：讀牌桌上四家最近打出的牌。

桌上的牌是手牌的 0.9 倍，五張牌型會散開成扇形（每張轉 −10°～+10°），所以點數模板縮小並轉幾個角度去找，
花色在點數下方（依角度轉過的位置）另外比對。四家的區域由 Google Play 版錄影量出（自己的出牌在桌面正中間）。
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
WORK = 0.75                      # 比對解析度（0.5 時 Google Play 版的 6 會被認成 8、3）
ANGLES = (-12, -8, -4, 0, 4, 8, 12)
RANK_THRESHOLD = 0.85            # 真的牌 ≥0.92；雜訊（其他圖案）最高 0.81
SUIT_THRESHOLD = 0.75
SURE_RANK = 0.9                  # 點數這麼像、花色卻認不出：整區回報讀不到
MIN_GAP = 22

# 整張牌桌一起找，再把相鄰的牌分成一手、看離哪家最近：五張牌散開的扇形常伸進別家的範圍
# （下家的扇形左緣到 x 1110，和對家的出牌區重疊），固定切四塊會把一手牌切開。
TABLE_REGION = (120, 30, 1800, 740)              # 下緣不含自己選起的手牌（上緣 745）
TABLE_MASKS = (
    (1200, 10, 1600, 112),     # 右上角 A／2 記牌器（也是牌面）
    (120, 30, 330, 340),       # 左上「代打」、上家頭像
    (1620, 140, 1800, 340),    # 下家頭像
    (120, 600, 490, 740),      # 聊天、刁牌按鈕
)
BLOB_MIN_AREA = 6000                            # 一張桌上的牌約 130x230
GROUP_GAP = 130                                  # 扇形裡相鄰兩張的點數框約差 70~80px
# 座位：0 自己、1 下家（右）、2 對家（上）、3 上家（左）；各家出牌的點數框大約中心
SEAT_ANCHORS = {0: (960, 500), 1: (1420, 320), 2: (960, 130), 3: (430, 420)}

# 點數框中心到花色框中心的位移（手牌尺寸）
_RANK_CENTER = ((RANK_BOX[0] + RANK_BOX[2]) / 2, (RANK_BOX[1] + RANK_BOX[3]) / 2)
_SUIT_CENTER = ((SUIT_BOX[0] + SUIT_BOX[2]) / 2, (SUIT_BOX[1] + SUIT_BOX[3]) / 2)
_SUIT_OFFSET = (_SUIT_CENTER[0] - _RANK_CENTER[0], _SUIT_CENTER[1] - _RANK_CENTER[1])


@lru_cache(maxsize=4)
def _table_templates(directory=TEMPLATE_DIR):
    ranks, suits = load_templates(directory)

    def scale(items, one_each):
        seen, out = set(), []
        for label, mask in items:          # 點數每種只留一份（整塊區域比對，次數少 4 倍）
            if one_each and label in seen:
                continue
            seen.add(label)
            out.append((label, cv2.resize(mask, None, fx=SCALE * WORK, fy=SCALE * WORK,
                                          interpolation=cv2.INTER_AREA)))
        return out

    return scale(ranks, True), scale(suits, False)


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


def find_cards(image: np.ndarray, region=TABLE_REGION, masks=TABLE_MASKS,
               directory=TEMPLATE_DIR) -> list[TableCard]:
    """區域裡所有牌（由左到右）；花色認不出、點數又很像的牌 card 記成 -1。

    把區域轉 ANGLES 各個角度（牌轉正）後用沒轉的模板比對，找到的位置再轉回原畫面。"""
    ranks, suits = _table_templates(directory)
    x0, y0, x1, y1 = region
    area = image[y0:y1, x0:x1].copy()
    for mx0, my0, mx1, my1 in masks:                  # 右上角 A／2 記牌器也是牌面
        area[max(0, my0 - y0):max(0, my1 - y0), max(0, mx0 - x0):max(0, mx1 - x0)] = (60, 120, 40)
    if (area.min(axis=2) > 200).mean() < 0.01:       # 沒有白色牌面
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

    def back(angle, px, py):
        inverse = rotated[angle][1]
        return (inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2],
                inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2])

    off = (SCALE * WORK * _SUIT_OFFSET[0], SCALE * WORK * _SUIT_OFFSET[1])
    picked = []
    for score, x, y, angle, i in candidates:
        cx, cy = x + tw / 2, y + th / 2
        ox, oy = back(angle, cx, cy)
        if all((ox - p[0]) ** 2 + (oy - p[1]) ** 2 >= (MIN_GAP * WORK) ** 2 for p in picked):
            picked.append((ox, oy, score, angle, i, cx, cy, back(angle, cx + off[0], cy + off[1])))
    # 花色有時也被當成點數（例如方塊被認成 3）：在別張牌花色位置上的去掉
    picked = [p for p in picked
              if not any(q is not p and (p[0] - q[7][0]) ** 2 + (p[1] - q[7][1]) ** 2 < (MIN_GAP * WORK) ** 2
                         for q in picked)]
    cards = []
    for ox, oy, score, angle, i, cx, cy, _ in sorted(picked):
        turned = rotated[angle][0]
        sx, sy = cx + off[0], cy + off[1]
        sh, sw = suits[0][1].shape
        pad = 8
        window = turned[max(0, int(sy - sh / 2 - pad)):int(sy + sh / 2 + pad),
                        max(0, int(sx - sw / 2 - pad)):int(sx + sw / 2 + pad)]
        glyph = turned[max(0, int(cy - th / 2)):int(cy + th / 2), max(0, int(cx - tw / 2)):int(cx + tw / 2)]
        red = is_red(glyph) if glyph.size else False
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
            value = -1          # 像是一張牌，但花色讀不出
        else:
            value = card(RANK_CHARS.index(ranks[i][0]), SUIT_CHARS.index(suit))
        cards.append(TableCard(value, int(x0 + ox / WORK), int(y0 + oy / WORK), angle, score))
    return cards


def group_cards(cards: list[TableCard]) -> list[list[TableCard]]:
    """相鄰（點數框中心距離 < GROUP_GAP）的牌是同一手。"""
    groups: list[list[TableCard]] = []
    for c in sorted(cards, key=lambda c: c.x):
        near = [g for g in groups if any((c.x - d.x) ** 2 + (c.y - d.y) ** 2 < GROUP_GAP ** 2 for d in g)]
        merged = [c] + [d for g in near for d in g]
        groups = [g for g in groups if g not in near] + [merged]
    return groups


def seat_of(group: list[TableCard]) -> int:
    gx = sum(c.x for c in group) / len(group)
    gy = sum(c.y for c in group) / len(group)
    return min(SEAT_ANCHORS, key=lambda s: (SEAT_ANCHORS[s][0] - gx) ** 2 + (SEAT_ANCHORS[s][1] - gy) ** 2)


def card_blobs(image: np.ndarray, region=TABLE_REGION, masks=TABLE_MASKS) -> list[tuple[int, int, int, int]]:
    """桌上白色牌面的區塊（一手牌的牌互相疊著，通常是一塊），只在這些地方比對。"""
    x0, y0, x1, y1 = region
    white = (image[y0:y1, x0:x1].min(axis=2) > 200).astype(np.uint8)
    for mx0, my0, mx1, my1 in masks:
        white[max(0, my0 - y0):max(0, my1 - y0), max(0, mx0 - x0):max(0, mx1 - x0)] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= BLOB_MIN_AREA and h >= 100 and w >= 70:
            pad = 30
            boxes.append((max(x0, x0 + x - pad), max(y0, y0 + y - pad),
                          min(x1, x0 + x + w + pad), min(y1, y0 + y + h + pad)))
    return boxes


def read_plays(image: np.ndarray, directory=TEMPLATE_DIR) -> dict[int, Combo | None | str]:
    """四家桌上的牌：Combo、None（沒有牌）或 "?"（有牌但讀不出／不是合法牌型）。"""
    out: dict[int, Combo | None | str] = {seat: None for seat in SEAT_ANCHORS}
    cards = [c for box in card_blobs(image) for c in find_cards(image, box, directory=directory)]
    for group in group_cards(cards):
        seat = seat_of(group)
        if out[seat] is not None or any(c.card < 0 for c in group):
            out[seat] = "?"
            continue
        combo = classify([c.card for c in group])
        out[seat] = combo if combo is not None else "?"
    return out
