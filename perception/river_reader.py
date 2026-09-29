"""辨識四家的牌河（打出的牌）。

牌河的格位排列還沒完整量到，所以不用固定格線，而是在各家區域內找牌面上緣
（見 config.RIVER_REGIONS 的說明），再用牌河專用模板辨識。

    python -m perception.river_reader <截圖>                         # 印出辨識結果
    python -m perception.river_reader <截圖> --label top=西1p... me=...  # 切出模板

標註依偵測順序：由上而下、同一列由左而右（格式同 game.tiles.parse，z 為字牌）。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from game.tiles import parse, tile_code, tile_name
from perception.capture import read_image, write_image
from perception.config import (
    RIVER_FACE_HEIGHT,
    RIVER_FACE_MIN,
    RIVER_FACE_PAD,
    RIVER_FACE_WIDTH,
    RIVER_GAP_MAX,
    RIVER_REGIONS,
    RIVER_ROW_TOLERANCE,
    RIVER_ROW_TOPS,
    RIVER_TEMPLATE_DIR,
)
from perception.hand_reader import Box, TileClassifier

SEATS = ("me", "right", "top", "left")
"""自己、下家、對家、上家（出牌順序）。"""
MIN_WHITE_RATIO = 0.6
"""上緣區段中真正白色（不含箭頭）的比例下限。"""
MERGE_DISTANCE = 4
"""同一張牌的上緣可能在相鄰幾列都被偵測到。"""


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2].tolist(), changes[1::2].tolist()))


def _arrow_mask(region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (hue >= 10) & (hue <= 30) & (saturation > 120) & (value > 150)
    # 箭頭外圍有一圈深色描邊，一併算進去
    return cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def locate_river_tiles(image: np.ndarray) -> dict[str, list[Box]]:
    """各家牌河每張牌的牌面位置，依由上而下、由左而右排序。"""
    low, high = RIVER_FACE_WIDTH
    result = {}
    for seat in SEATS:
        x0, y0, x1, y1 = RIVER_REGIONS[seat]
        region = image[y0:y1, x0:x1]
        gray = region.min(axis=2)
        # 最新打出的牌上方有上下跳動的橘色箭頭，尖端可能壓到牌面上緣：
        # 箭頭像素在牌面列算牌面、在上方列算暗縫；箭頭本身則靠「真正白色要過半」排除
        arrow = _arrow_mask(region)
        white = gray > RIVER_FACE_MIN
        face = white | arrow
        gap = (gray < RIVER_GAP_MAX) | arrow
        edge = face[2:-1] & face[3:] & (gap[:-3] | gap[1:-2])
        tops: list[tuple[int, int, int]] = []
        for row in range(edge.shape[0]):
            y = y0 + row + 2
            if all(abs(y - top) > RIVER_ROW_TOLERANCE for top in RIVER_ROW_TOPS[seat]):
                continue
            for left, right in _runs(edge[row]):
                if not low <= right - left <= high:
                    continue
                if white[row + 2, left:right].mean() < MIN_WHITE_RATIO:
                    continue
                left, right = x0 + left, x0 + right
                if any(abs(y - ty) <= MERGE_DISTANCE and min(right, tr) - max(left, tl) > 20
                       for tl, ty, tr in tops):
                    continue
                tops.append((left, y, right))
        height = RIVER_FACE_HEIGHT[seat]
        result[seat] = [(left - RIVER_FACE_PAD, y, right + RIVER_FACE_PAD, y + height)
                        for left, y, right in sorted(tops, key=lambda t: (t[1], t[0]))]
    return result


def crop_face(image: np.ndarray, box: Box) -> np.ndarray:
    left, top, right, bottom = box
    return image[top:bottom, left:right]


@dataclass(frozen=True)
class RiverTile:
    tile: int | None
    score: float
    box: Box


class RiverReader:
    def __init__(self, classifier: TileClassifier | None) -> None:
        self.classifier = classifier

    @classmethod
    def from_directory(cls, directory: Path = RIVER_TEMPLATE_DIR,
                       min_score: float = 0.7) -> "RiverReader":
        if not directory.exists() or not any(directory.iterdir()):
            return cls(None)
        return cls(TileClassifier.from_directory(directory, min_score=min_score))

    def read(self, image: np.ndarray) -> dict[str, list[RiverTile]]:
        rivers = {}
        for seat, boxes in locate_river_tiles(image).items():
            tiles = []
            for box in boxes:
                if self.classifier is None:
                    tiles.append(RiverTile(None, 0.0, box))
                    continue
                tile, score = self.classifier.classify(crop_face(image, box))
                tiles.append(RiverTile(tile, score, box))
            rivers[seat] = tiles
        return rivers


def extract_templates(image: np.ndarray, labels: dict[str, list[int]], name: str,
                      output_dir: Path = RIVER_TEMPLATE_DIR) -> list[Path]:
    located = locate_river_tiles(image)
    paths = []
    for seat, tiles in labels.items():
        boxes = located[seat]
        if len(boxes) != len(tiles):
            raise ValueError(f"{seat} 偵測到 {len(boxes)} 張牌，但標註有 {len(tiles)} 張")
        for index, (box, tile) in enumerate(zip(boxes, tiles)):
            path = output_dir / tile_code(tile) / f"{name}_{seat}{index:02d}.png"
            write_image(path, crop_face(image, box))
            paths.append(path)
    return paths


def format_rivers(rivers: dict[str, list[RiverTile]]) -> str:
    lines = []
    for seat in SEATS:
        parts = [f"{tile_name(t.tile) if t.tile is not None else '??'}({t.score:.2f})"
                 for t in rivers[seat]]
        lines.append(f"{seat:5s} {len(parts):2d} 張：" + " ".join(parts))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="辨識牌河或切出牌河模板")
    parser.add_argument("image", type=Path)
    parser.add_argument("--label", nargs="+", metavar="座位=牌",
                        help="例如 top=3z1p6z me=5z7m；座位為 me、right、top、left")
    args = parser.parse_args()
    image = read_image(args.image)
    if args.label:
        labels = {}
        for item in args.label:
            seat, _, tiles = item.partition("=")
            if seat not in SEATS:
                raise SystemExit(f"未知的座位：{seat}")
            labels[seat] = parse(tiles)
        paths = extract_templates(image, labels, args.image.stem)
        print(f"已輸出 {len(paths)} 張牌河模板到 {RIVER_TEMPLATE_DIR}")
        return
    print(format_rivers(RiverReader.from_directory().read(image)))


if __name__ == "__main__":
    main()
