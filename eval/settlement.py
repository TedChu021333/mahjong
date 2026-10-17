"""讀小結算畫面：四家的手牌、胡的那張、莊家、胡牌者有沒有宣告聽牌。

畫面四列由上到下固定是東南西北（見 eval.stats）。每列的牌從 x=778 起、間距約 55.75 排 16 張
（副露也混在裡面、看不出分組），胡牌者胡的那張另外放在右邊有框的位置 (1701)。牌比手牌小，
用結算畫面專用的模板 train/settlement_templates/<牌>/*.png（由 bootstrap 從手牌模板自動收集，
見 python -m eval.settlement --bootstrap）；沒有時退回手牌模板只取牌面。

    python -m eval.settlement result/20261005_010658.png
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eval.stats import ROW_TOPS
from game.tiles import tile_code, tile_name
from perception.capture import read_image, write_image
from perception.config import TEMPLATE_DIR
from perception.hand_reader import TileClassifier

TILE_PITCH = 55.75
FIRST_TILE_X = 778
SLOTS = 16
WIN_TILE_X = 1701
TILE_WIDTH = 52
TILE_Y = (86, 160)
"""牌面相對於列頂的上下範圍（只取上方牌面，下緣的金牌燈泡、底座不比）。"""
HAND_TEMPLATE_BOX = (2, 22, 100, 160)
"""沒有結算模板時，手牌模板取這塊縮放比對（10/5 兩張截圖 50 張牌全對，但一萬與六萬只差 0.01）。"""
SETTLEMENT_TEMPLATE_DIR = Path("train/settlement_templates")
MIN_SCORE = 0.5
MIN_LEAD = 0.05
FACE_MIN_RATIO = 0.35
"""牌位上半部白色牌面的比例；空位（自己放槍後只剩 16 張）是藍底。"""
TING_TEMPLATE = Path(__file__).with_name("tai_ting.png")
TING_REGION = (150, 200, 520, 760)
"""左邊台數明細的範圍；「聽牌」「地聽」都有「聽」字（有 ≥0.93、沒有 ≤0.45）。"""
TING_MIN_SCORE = 0.8
DU_TEMPLATE = Path(__file__).with_name("tai_du.png")
"""「獨聽」的「獨」：獨聽的「聽」字也會配上（0.90），前一個字是「獨」的不算宣告（10/18 00:09 誤判）。"""
DU_MIN_SCORE = 0.8
CHAR_PITCH = 40
"""台數明細一個字的寬度（獨聽的「獨」在「聽」左邊 40 像素）。"""
DEALER_BOX = (695, 83, 750, 133)
"""各列風位標籤下方「莊」的紅色方塊（相對於列頂）。"""
DEALER_MIN_RED = 0.4


@dataclass(frozen=True)
class SettlementRow:
    tiles: tuple[int | None, ...]
    """16 個牌位（空位不列；認不出的為 None）。"""
    win_tile: int | None
    """有框的胡牌那張；不是胡牌者為 None。"""
    dealer: bool


@dataclass(frozen=True)
class Settlement:
    rows: tuple[SettlementRow, ...]
    """東南西北四列。"""
    winner_declared: bool
    """台數明細有「聽」（聽牌、地聽）。"""


def slot_box(top: int, index: int) -> tuple[int, int, int, int]:
    x = WIN_TILE_X if index == SLOTS else round(FIRST_TILE_X + TILE_PITCH * index)
    return (x + 1, top + TILE_Y[0], x + 1 + TILE_WIDTH, top + TILE_Y[1])


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return image[top:bottom, left:right]


def is_face(region: np.ndarray) -> bool:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).mean()) >= FACE_MIN_RATIO


class SettlementReader:
    def __init__(self, classifier: TileClassifier, ting: np.ndarray | None,
                 du: np.ndarray | None = None) -> None:
        self.classifier = classifier
        self.ting = ting
        self.du = du

    @classmethod
    def load(cls, directory: Path = SETTLEMENT_TEMPLATE_DIR) -> "SettlementReader":
        if directory.exists() and any(directory.glob("*/*.png")):
            classifier = TileClassifier.from_directory(directory, min_score=0)
        else:
            classifier = TileClassifier.from_directory(TEMPLATE_DIR, template_box=HAND_TEMPLATE_BOX,
                                                       min_score=0)
        ting = None
        if TING_TEMPLATE.exists():
            ting = cv2.cvtColor(read_image(TING_TEMPLATE), cv2.COLOR_BGR2GRAY)
        du = None
        if DU_TEMPLATE.exists():
            du = cv2.cvtColor(read_image(DU_TEMPLATE), cv2.COLOR_BGR2GRAY)
        return cls(classifier, ting, du)

    def classify(self, region: np.ndarray) -> tuple[int | None, float, float]:
        """(牌, 分數, 領先第二名多少)；分數或領先不夠時牌為 None。"""
        scores = self.classifier.scores(region)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        lead = scores[best] - scores[second]
        ok = scores[best] >= MIN_SCORE and lead >= MIN_LEAD
        return (best if ok else None), scores[best], lead

    def read_row(self, image: np.ndarray, top: int) -> SettlementRow:
        tiles = []
        for index in range(SLOTS):
            region = crop(image, slot_box(top, index))
            if is_face(region):
                tiles.append(self.classify(region)[0])
        win_region = crop(image, slot_box(top, SLOTS))
        win_tile = self.classify(win_region)[0] if is_face(win_region) else None
        return SettlementRow(tuple(tiles), win_tile, is_dealer(image, top))

    def read(self, image: np.ndarray) -> Settlement:
        return Settlement(tuple(self.read_row(image, top) for top in ROW_TOPS),
                          self.winner_declared(image))

    def winner_declared(self, image: np.ndarray) -> bool:
        if self.ting is None:
            return False
        left, top, right, bottom = TING_REGION
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(gray, self.ting, cv2.TM_CCOEFF_NORMED)
        height, width = self.ting.shape
        for y, x in zip(*np.nonzero(scores >= TING_MIN_SCORE)):
            if not self._is_du(gray, x - CHAR_PITCH, y, width, height):
                return True  # 聽牌、地聽
        return False

    def _is_du(self, gray: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
        """(x, y) 附近是不是「獨」字（容許幾個像素的偏差）。"""
        if self.du is None:
            return False
        pad = 4
        area = gray[max(0, y - pad):y + height + pad, max(0, x - pad):x + width + pad]
        if area.shape[0] < self.du.shape[0] or area.shape[1] < self.du.shape[1]:
            return False
        return float(cv2.matchTemplate(area, self.du, cv2.TM_CCOEFF_NORMED).max()) >= DU_MIN_SCORE


def is_dealer(image: np.ndarray, top: int) -> bool:
    left, box_top, right, bottom = DEALER_BOX
    region = image[top + box_top:top + bottom, left:right].astype(int)
    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    return float(((red > 150) & (red - green > 80) & (red - blue > 80)).mean()) >= DEALER_MIN_RED


def bootstrap(paths: list[Path], per_tile: int = 12, min_lead: float = 0.12,
              output: Path = SETTLEMENT_TEMPLATE_DIR, seed: int = 0) -> dict[int, int]:
    """用手牌模板認結算畫面，領先夠多的牌存成結算模板（每種牌隨機取 per_tile 張）。"""
    reader = SettlementReader(
        TileClassifier.from_directory(TEMPLATE_DIR, template_box=HAND_TEMPLATE_BOX, min_score=0),
        None)
    found: dict[int, list[np.ndarray]] = {}
    for path in paths:
        image = read_image(path)
        for top in ROW_TOPS:
            for index in range(SLOTS + 1):
                region = crop(image, slot_box(top, index))
                if not is_face(region):
                    continue
                tile, score, lead = reader.classify(region)
                if tile is not None and lead >= min_lead:
                    found.setdefault(tile, []).append(region.copy())
    rng = random.Random(seed)
    for tile, regions in found.items():
        rng.shuffle(regions)
        for index, region in enumerate(regions[:per_tile]):
            write_image(output / tile_code(tile) / f"boot_{index:02d}.png", region)
    return {tile: len(regions) for tile, regions in found.items()}


def format_settlement(settlement: Settlement) -> str:
    lines = []
    for wind, row in zip("東南西北", settlement.rows):
        text = "".join(tile_name(t) if t is not None else "?" for t in row.tiles)
        if row.win_tile is not None:
            text += f"　胡 {tile_name(row.win_tile)}"
        lines.append(f"{wind}{'（莊）' if row.dealer else ''}：{text}")
    lines.append(f"胡牌者宣告聽牌：{'是' if settlement.winner_declared else '否'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="讀小結算畫面的四家手牌")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--bootstrap", nargs="*", type=Path,
                        help="由這些結算截圖自動收集結算模板")
    args = parser.parse_args()
    if args.bootstrap is not None:
        counts = bootstrap(args.bootstrap)
        print("收集到：" + "、".join(f"{tile_name(t)} {n}" for t, n in sorted(counts.items())))
        return
    reader = SettlementReader.load()
    for path in args.images:
        print(path)
        print(format_settlement(reader.read(read_image(path))))


if __name__ == "__main__":
    main()
