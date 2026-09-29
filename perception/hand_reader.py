"""自己暗手牌的定位與牌面辨識（OpenCV 模板比對）。

定位：暗手牌排在間距固定的格線上，逐格檢查一條橫帶是否為白色牌面，
張數由畫面決定，不依副露數推算；副露牌位置較低，不會被誤算成暗手牌。
摸進的第 17 張在右側獨立位置。被選取而上移的牌由牌面下緣量出位移後再裁切。

辨識：每格與 train/templates/<牌代碼>/ 的模板做 TM_CCOEFF_NORMED 比對，
取最高分；此分數對亮度的線性變化不敏感，吃碰提示時的變暗遮罩與高亮牌
都能用同一組模板。低於門檻的格子回傳 None，交給人工確認。

用法：
    python -m perception.hand_reader train/完整正常畫面.png
    python -m perception.hand_reader --screen
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from game.tiles import NUM_TILE_TYPES, parse, tile_code, tile_name
from perception.capture import grab_screen, read_image
from perception.config import (
    HAND_FACE_PROBE_ROWS,
    HAND_GRID_RIGHT,
    HAND_TILE_BOTTOM,
    HAND_TILE_PITCH,
    HAND_TILE_TOP,
    DRAWN_TILE_CROP_X,
    MAX_HAND_TILES,
    TILE_RAISE,
    TEMPLATE_DIR,
)

Box = tuple[int, int, int, int]  # (left, top, right, bottom)

# 牌面像素：低飽和度、夠亮（吃碰提示的變暗遮罩下亮度約 140，桌面約 45）
FACE_MAX_SATURATION = 60
FACE_MIN_VALUE = 90
FACE_MIN_RATIO = 0.6

RAISE_TOLERANCE = 8

TILE_SIZE = (40, 60)  # 比對前統一縮放的 (寬, 高)
TEMPLATE_MARGIN = 2    # 模板四邊內縮，讓比對可容忍幾個像素的位移
HIGH_PASS_SIGMA = 3.0
MIN_MATCH_SCORE = 0.78  # 13 張截圖留一驗證：正確牌最低 0.832，錯誤種類最高 0.736


def _cell_box(index_from_right: int) -> Box:
    right = HAND_GRID_RIGHT - HAND_TILE_PITCH * index_from_right
    left = right - HAND_TILE_PITCH
    return (round(left), HAND_TILE_TOP, round(right), HAND_TILE_BOTTOM)


def _face_mask(region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < FACE_MAX_SATURATION) & (hsv[..., 2] > FACE_MIN_VALUE)


def _is_tile_face(image: np.ndarray, box: Box) -> bool:
    """檢查牌面上緣的空白橫帶；橫帶跟著牌的上移量一起移動。"""
    left, box_top, right, _ = box
    if left < 0 or right > image.shape[1]:
        return False
    shift = HAND_TILE_TOP - box_top
    top, bottom = (row - shift for row in HAND_FACE_PROBE_ROWS)
    inset = (right - left) // 10
    face = _face_mask(image[top:bottom, left + inset:right - inset])
    return float(face.mean()) >= FACE_MIN_RATIO


def _raise_offset(image: np.ndarray, box: Box) -> int:
    """被選取的牌會上移；由牌面最下緣與正常位置的差距求出位移（0 或約 48）。"""
    left, _, right, bottom = box
    inset = (right - left) // 5
    region = image[bottom - TILE_RAISE - 2 * RAISE_TOLERANCE:bottom,
                   left + inset:right - inset]
    rows = np.where(_face_mask(region).mean(axis=1) >= FACE_MIN_RATIO)[0]
    if len(rows) == 0:
        return 0
    offset = region.shape[0] - 1 - int(rows.max())
    return offset if abs(offset - TILE_RAISE) <= RAISE_TOLERANCE else 0


def _shift_up(box: Box, offset: int) -> Box:
    left, top, right, bottom = box
    return (left, top - offset, right, bottom - offset)


def drawn_tile_box() -> Box:
    left, right = DRAWN_TILE_CROP_X
    return (left, HAND_TILE_TOP, right, HAND_TILE_BOTTOM)


def locate_hand_tiles(image: np.ndarray) -> list[Box]:
    """回傳暗手牌每張的框，由左到右；若有摸進的第 17 張，放在最後。"""
    cells = [_cell_box(index) for index in reversed(range(MAX_HAND_TILES))]
    cells = [_shift_up(box, _raise_offset(image, box)) for box in cells]
    found = [_is_tile_face(image, box) for box in cells]
    # 暗手牌一定佔連續的格子；槓子第 4 張疊在副露中間會單獨被偵測到，
    # 所以只保留最長的連續段（同長取右邊）。
    best_start, best_length, start = 0, 0, None
    for index, hit in enumerate(found + [False]):
        if hit and start is None:
            start = index
        elif not hit and start is not None:
            if index - start >= best_length:
                best_start, best_length = start, index - start
            start = None
    boxes = cells[best_start:best_start + best_length]
    drawn = _shift_up(drawn_tile_box(), _raise_offset(image, drawn_tile_box()))
    if _is_tile_face(image, drawn):
        boxes.append(drawn)
    return boxes


def crop_tile(image: np.ndarray, box: Box) -> np.ndarray:
    left, top, right, bottom = box
    return image[top:bottom, left:right].copy()


def _normalize(tile_image: np.ndarray) -> np.ndarray:
    """縮放後做高通濾波：去掉變暗遮罩與高亮造成的亮度漸層，只留筆畫結構。"""
    resized = cv2.resize(tile_image, TILE_SIZE, interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)
    return resized - cv2.GaussianBlur(resized, (0, 0), HIGH_PASS_SIGMA)


def _unit_rows(patches: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) 影像塊：各通道減去平均後攤平並正規化成單位向量，
    兩兩內積即為 TM_CCOEFF_NORMED 分數。"""
    centered = patches - patches.mean(axis=(1, 2), keepdims=True)
    flat = centered.reshape(len(patches), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norms, 1e-6)


class TileClassifier:
    """以模板比對辨識單張牌；templates 為 (牌編碼, 圖片) 列表。"""

    def __init__(self, templates: list[tuple[int, np.ndarray]],
                 min_score: float = MIN_MATCH_SCORE) -> None:
        self.min_score = min_score
        if not templates:
            raise ValueError("沒有任何牌面模板")
        m = TEMPLATE_MARGIN
        self.templates = []
        for tile, image in templates:
            if not 0 <= tile < NUM_TILE_TYPES:
                raise ValueError(f"模板牌編碼無效：{tile}")
            self.templates.append((tile, _normalize(image)[m:-m, m:-m]))
        self._tiles = np.array([tile for tile, _ in self.templates])
        middle = self.templates[0][1].shape[0] // 2
        self._halves = (
            _unit_rows(np.stack([t[:middle] for _, t in self.templates])),
            _unit_rows(np.stack([t[middle:] for _, t in self.templates])),
        )

    @classmethod
    def from_directory(cls, directory: str | Path = TEMPLATE_DIR,
                       template_box: Box | None = None,
                       min_score: float = MIN_MATCH_SCORE) -> "TileClassifier":
        """讀取 <directory>/<牌代碼>/*.png。

        template_box 只取模板中的一塊（left, top, right, bottom），用來辨識
        畫面上其他位置、比例不同的正面牌（例如金牌面板只有牌面、沒有牌邊）。
        """
        directory = Path(directory)
        templates = []
        for tile_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            codes = parse(tile_dir.name)
            if len(codes) != 1:
                raise ValueError(f"模板資料夾名稱必須是單張牌代碼：{tile_dir.name}")
            templates.extend((codes[0], read_image(path))
                             for path in sorted(tile_dir.glob("*.png")))
        if template_box is not None:
            left, top, right, bottom = template_box
            templates = [(tile, image[top:bottom, left:right]) for tile, image in templates]
        return cls(templates, min_score)

    @property
    def known_tiles(self) -> set[int]:
        return {tile for tile, _ in self.templates}

    def scores(self, tile_image: np.ndarray) -> dict[int, float]:
        """每種牌的最佳比對分數（-1 ~ 1）。

        上下半張分開比對取較低者：萬子下半部的「萬」字都相同，整張比對
        會被它拉高，分開後由上半的數字決定分數。每半只在原位置上下左右
        TEMPLATE_MARGIN 內搜尋；分數等同 TM_CCOEFF_NORMED，但一次對所有模板
        以矩陣運算計算（逐一呼叫 matchTemplate 一幀要 3 秒以上）。
        """
        target = _normalize(tile_image)
        m = TEMPLATE_MARGIN
        middle = (target.shape[0] - 2 * m) // 2
        regions = (target[:middle + 2 * m], target[middle:])
        per_template = None
        for region, templates in zip(regions, self._halves):
            height = region.shape[0] - 2 * m
            width = region.shape[1] - 2 * m
            windows = np.lib.stride_tricks.sliding_window_view(region, (height, width, 3))
            windows = _unit_rows(windows.reshape(-1, height, width, 3))
            # einsum 不經過 OpenBLAS；多執行緒 BLAS 處理這種小矩陣在 Windows 上慢約 60 倍，
            # 而環境變數只有在 numpy 載入前設定才有效，改用 einsum 就不受載入順序影響
            best = np.einsum("ij,kj->ik", windows, templates, optimize=False).max(axis=0)
            per_template = best if per_template is None else np.minimum(per_template, best)
        result: dict[int, float] = {}
        for tile, score in zip(self._tiles.tolist(), per_template.tolist()):
            if score > result.get(tile, -1.0):
                result[tile] = score
        return result

    def classify(self, tile_image: np.ndarray) -> tuple[int | None, float]:
        scores = self.scores(tile_image)
        tile = max(scores, key=scores.__getitem__)
        score = scores[tile]
        return (tile if score >= self.min_score else None), score


@dataclass(frozen=True)
class HandReading:
    boxes: tuple[Box, ...]
    tiles: tuple[int | None, ...]
    scores: tuple[float, ...]

    @property
    def complete(self) -> bool:
        return all(tile is not None for tile in self.tiles)

    def known_tiles(self) -> tuple[int, ...]:
        """只在全部辨識成功時回傳，避免把缺牌的手牌餵給規則引擎。"""
        if not self.complete:
            raise ValueError("手牌中有無法辨識的牌")
        return tuple(tile for tile in self.tiles if tile is not None)


def read_hand(image: np.ndarray, classifier: TileClassifier) -> HandReading:
    boxes = locate_hand_tiles(image)
    tiles, scores = [], []
    for box in boxes:
        tile, score = classifier.classify(crop_tile(image, box))
        tiles.append(tile)
        scores.append(score)
    return HandReading(tuple(boxes), tuple(tiles), tuple(scores))


def format_reading(reading: HandReading) -> str:
    parts = [
        f"{tile_name(tile)}({score:.2f})" if tile is not None else f"??({score:.2f})"
        for tile, score in zip(reading.tiles, reading.scores)
    ]
    return f"{len(reading.tiles)} 張：" + " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="辨識自己的暗手牌")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("image", nargs="?", type=Path)
    source.add_argument("--screen", action="store_true", help="直接擷取目前螢幕")
    parser.add_argument("--templates", type=Path, default=TEMPLATE_DIR)
    args = parser.parse_args()
    image = grab_screen() if args.screen else read_image(args.image)
    reading = read_hand(image, TileClassifier.from_directory(args.templates))
    print(format_reading(reading))
    if reading.complete:
        print("代碼：" + " ".join(tile_code(tile) for tile in reading.known_tiles()))


if __name__ == "__main__":
    main()
