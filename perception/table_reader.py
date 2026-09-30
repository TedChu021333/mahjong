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
    AUTOPLAY_BUTTON_REGION,
    DEALER_BADGE_REGIONS,
    DECLARE_MIN_SCORE,
    DECLARE_REGIONS,
    GOLD_MAX_SLOTS,
    GOLD_SLOT_ORIGIN,
    GOLD_SLOT_PITCH,
    GOLD_SLOT_SIZE,
    STREAK_TEMPLATE_DIR,
    SWAP_CONFIRM_BUTTON,
    SWAP_KEEP_BUTTON,
)
from perception.hand_reader import TileClassifier

BUTTON_HALF_SIZE = (25, 28)
BUTTON_LIT_VALUE = 120
"""亮的按鈕亮度遠高於此值，變暗的按鈕在此之下（影片量測：亮 1.0、暗 ≤0.16）。"""
CANCEL_MIN_RED = 0.42
TILE_FACE_MIN_RATIO = 0.6
CIRCLE_PROBE_OFFSETS = (62, 95)
"""圓框判斷的取樣帶：距牌中心水平 62～95 像素（牌半寬約 53、圓框半徑約 105）。"""
CIRCLE_PROBE_HALF_HEIGHT = 40
CIRCLE_DARK_MAX = 40
CIRCLE_MIN_DARK_RATIO = 0.6
RING_RADII = (104, 124, 2)
RING_ANGLE_STEP = 5
RING_MIN_RATIO = 0.8
"""有圓框的畫面外圈比例 1.0；其他 60 張截圖（結算、牌河、讀不到手牌）最高 0.65。"""
CLAIM_MIN_SCORE = 0.6
CLAIM_MIN_MARGIN = 0.1
CLAIM_STRONG_SCORE = 0.8
CLAIM_STRONG_MARGIN = 0.2
"""沒偵測到圓框也採用的門檻（21.12.08 錄影 543 幀：除了圓框縮小動畫中的真實出牌，沒有誤判）。"""
"""放大的牌：最高分 ≥0.6 且領先次高 0.1 才採用（21.12.08 影片 17 次出牌：正確牌最低 0.70，
與次高差最小 0.12；縮放動畫中的幀分數 <0.6 或差距 <0.05）。"""
GOLD_MIN_FACE_RATIO = 0.3
STREAK_REGION = (-18, -33, 16, 14)
"""連莊數的搜尋範圍，相對「莊」方塊右下角 (左, 上, 右, 下)。"""
STREAK_WHITE_MIN = 190
STREAK_DIGIT_MIN_X = 10
STREAK_DIGIT_MIN_AREA = 50
STREAK_DIGIT_SIZE = (12, 20)
STREAK_MIN_IOU = 0.7


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


def _ring_ratio(image: np.ndarray, center_x: int, center_y: int) -> float:
    """圓框外緣淡紫色細圈（半徑約 113）：各方向在 RING_RADII 範圍內有淡紫色像素的比例。
    外圈畫在牌河上面，不怕被壓住；白色牌面、綠色桌面都不是這個顏色。"""
    angles = np.deg2rad(np.arange(0, 360, RING_ANGLE_STEP))[:, None]
    radii = np.arange(*RING_RADII)[None, :]
    xs = np.clip((center_x + radii * np.cos(angles)).astype(int), 0, image.shape[1] - 1)
    ys = np.clip((center_y + radii * np.sin(angles)).astype(int), 0, image.shape[0] - 1)
    pixels = image[ys, xs].astype(int)
    blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    lavender = (blue > 190) & (blue - green > 18) & (red - green > -8)
    return float(lavender.any(axis=1).mean())


def _in_discard_circle(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """放大的牌外面有半透明的暗色圓框；牌左右兩側任一側夠暗就算（另一側可能壓著牌河，
    透出來較亮）。綠色桌面、牌河、牌面都比這亮得多。上家牌河排到第二欄後兩側都壓著牌河
    （2026-09-30 七次吃碰沒處理有六次是這樣），改看圓框的淡紫色外圈。"""
    left, top, right, bottom = box
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    rows = slice(center_y - CIRCLE_PROBE_HALF_HEIGHT, center_y + CIRCLE_PROBE_HALF_HEIGHT)
    near, far = CIRCLE_PROBE_OFFSETS
    sides = (image[rows, center_x - far:center_x - near],
             image[rows, center_x + near:center_x + far])
    if max(float((side.max(axis=2) < CIRCLE_DARK_MAX).mean()) for side in sides) \
            >= CIRCLE_MIN_DARK_RATIO:
        return True
    return _ring_ratio(image, center_x, center_y) >= RING_MIN_RATIO


def read_claim_tile(image: np.ndarray, classifier: TileClassifier) -> ClaimTile | None:
    """讀出圓框裡放大顯示的那張牌（別家剛打出、有人可能吃碰胡時顯示約 0.65 秒；
    能吃碰時會停到按鈕按下）。沒有放大的牌回傳 None；有圓框但認不出時 tile 為 None。

    圓框判斷靠牌左右的暗色內部，但圓框蓋在牌河上時兩側都是亮色的牌、判斷會失敗
    （實戰 7 次吃碰沒處理有 6 次是這樣，都在上家的位置）；牌本身讀得很清楚
    （≥CLAIM_STRONG_SCORE 且領先 ≥CLAIM_STRONG_MARGIN）時不需要圓框。"""
    width, height = CLAIM_TILE_SIZE
    pad_left, pad_top, pad_right, pad_bottom = CLAIM_TILE_MARGINS
    for source, (x, y) in CLAIM_TILE_POSITIONS.items():
        circle = _in_discard_circle(image, (x, y, x + width, y + height))
        crop = image[y - pad_top:y + height + pad_bottom, x - pad_left:x + width + pad_right]
        ranked = sorted(classifier.scores(crop).items(), key=lambda item: -item[1])
        (tile, score), (_, second) = ranked[0], ranked[1]
        strong = score >= CLAIM_STRONG_SCORE and score - second >= CLAIM_STRONG_MARGIN
        if not circle and not strong:
            continue
        known = score >= CLAIM_MIN_SCORE and score - second >= CLAIM_MIN_MARGIN
        return ClaimTile(tile if known else None, score, source)
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
        # 空格是藍色面板底色；有牌時整格白色比例高（圖案可能貼近邊緣，不能只看上緣）
        if _face_ratio(image, box) < GOLD_MIN_FACE_RATIO:
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


def read_declared(image: np.ndarray, template: np.ndarray | None) -> frozenset[str]:
    """哪幾家（left/top/right）旁邊有「聽」標記；template 為灰階模板，沒有模板時回傳空集合。"""
    if template is None:
        return frozenset()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seats = set()
    for seat, (left, top, right, bottom) in DECLARE_REGIONS.items():
        region = gray[top:bottom, left:right]
        if cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max() >= DECLARE_MIN_SCORE:
            seats.add(seat)
    return frozenset(seats)


def _dealer_badges(image: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """各位置找到的「莊」方塊（座位, (左, 上, 寬, 高)）：實心的紅色方塊；橫幅上的紅字、
    牌上的紅色筆畫都太細，不會被當成方塊。"""
    found = []
    for seat, (left, top, right, bottom) in DEALER_BADGE_REGIONS.items():
        hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red = (((hue < 8) | (hue > 172)) & (saturation > 150) & (value > 150)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(red)
        for x, y, w, h, area in stats[1:count]:
            if 44 <= w <= 62 and 44 <= h <= 62 and area > 1200:
                found.append((seat, (left + int(x), top + int(y), int(w), int(h))))
                break
    return found


def read_dealer(image: np.ndarray) -> str | None:
    """誰是莊家（me/left/top/right）；認不出來時回傳 None。"""
    found = _dealer_badges(image)
    return found[0][0] if len(found) == 1 else None


def streak_digit_mask(image: np.ndarray) -> np.ndarray | None:
    """「莊」方塊右下角的連莊數（白字紅邊，連 1 起才出現）：回傳縮放成 STREAK_DIGIT_SIZE 的
    字形遮罩；沒有數字（或認不出莊家）回傳 None。"""
    found = _dealer_badges(image)
    if len(found) != 1:
        return None
    _, (x, y, w, h) = found[0]
    left, top = x + w + STREAK_REGION[0], y + h + STREAK_REGION[1]
    region = image[top:y + h + STREAK_REGION[3], left:x + w + STREAK_REGION[2]]
    white = (region.min(axis=2) > STREAK_WHITE_MIN).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    # 「莊」字右下的筆畫也在區域左側，數字從方塊右緣附近開始
    digits = [index for index in range(1, count)
              if stats[index][0] >= STREAK_DIGIT_MIN_X and stats[index][4] >= STREAK_DIGIT_MIN_AREA]
    if not digits:
        return None
    index = max(digits, key=lambda i: stats[i][4])
    dx, dy, dw, dh, _ = stats[index]
    mask = (labels[dy:dy + dh, dx:dx + dw] == index).astype(np.uint8) * 255
    return cv2.resize(mask, STREAK_DIGIT_SIZE, interpolation=cv2.INTER_AREA)


def load_streak_templates(directory=STREAK_TEMPLATE_DIR) -> dict[int, np.ndarray]:
    """連莊數字模板：<目錄>/<數字>.png（streak_digit_mask 存下的遮罩）。"""
    templates = {}
    for path in sorted(directory.glob("*.png")) if directory.exists() else ():
        if path.stem.isdigit():
            mask = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
            templates[int(path.stem)] = cv2.resize(mask, STREAK_DIGIT_SIZE)
    return templates


def read_dealer_streak(image: np.ndarray, templates: dict[int, np.ndarray]) -> int | None:
    """連莊數：沒有數字為 0；有數字但和模板都不像（還沒收集到的 3 以上）回傳 None。"""
    mask = streak_digit_mask(image)
    if mask is None:
        return 0
    best, best_iou = None, 0.0
    shape = mask > 127
    for number, template in templates.items():
        other = template > 127
        iou = float((shape & other).sum()) / max(1, int((shape | other).sum()))
        if iou > best_iou:
            best, best_iou = number, iou
    return best if best_iou >= STREAK_MIN_IOU else None


def autoplay_active(image: np.ndarray) -> bool:
    """聽牌後遊戲在代打（手牌上蓋著「取消代打」按鈕）：此時讀不到手牌是正常的。"""
    left, top, right, bottom = AUTOPLAY_BUTTON_REGION
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    orange = (hsv[..., 0] < 20) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
    return float(orange.mean()) > 0.6
