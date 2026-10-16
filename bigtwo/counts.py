"""大老二：讀三家頭像旁「剩幾張」的數字（暗底白字，0～13）。

數字模板放 `train/bigtwo/digits/<數字>/*.png`（白字遮罩），由 `extract_digits` 從已知張數的畫面裁出。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .screen import read_image, write_image

DIGIT_DIR = Path(__file__).resolve().parent.parent / "train" / "bigtwo" / "digits"

# 座位 1 下家（右）、2 對家（上）、3 上家（左）
BADGES = {1: (1555, 225, 1632, 312), 2: (1033, 65, 1102, 148), 3: (303, 225, 378, 312)}   # Google Play 版
DIGIT_SIZE = (20, 30)
DIGIT_THRESHOLD = 0.8


def _components(image: np.ndarray, box) -> list[np.ndarray]:
    x0, y0, x1, y1 = box
    white = (image[y0:y1, x0:x1].min(axis=2) > 200).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    parts = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h >= 18 and area >= 25:
            glyph = (labels[y:y + h, x:x + w] == i).astype(np.float32)
            parts.append((x, cv2.resize(glyph, DIGIT_SIZE, interpolation=cv2.INTER_AREA)))
    return [g for _, g in sorted(parts, key=lambda p: p[0])]


@lru_cache(maxsize=2)
def load_digits(directory: Path = DIGIT_DIR):
    out = []
    for folder in sorted(directory.glob("*")):
        for path in sorted(folder.glob("*.png")):
            out.append((int(folder.name), read_image(path)[:, :, 0].astype(np.float32) / 255))
    return out


def _classify(glyph: np.ndarray, templates) -> tuple[int | None, float]:
    best, digit = -1.0, None
    for d, t in templates:
        score = float(cv2.matchTemplate(glyph, t, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best:
            best, digit = score, d
    return digit, best


def read_count(image: np.ndarray, seat: int, directory: Path = DIGIT_DIR) -> int | None:
    templates = load_digits(directory)
    parts = _components(image, BADGES[seat])
    if not parts or len(parts) > 2:
        return None
    value = 0
    for glyph in parts:
        digit, score = _classify(glyph, templates)
        if digit is None or score < DIGIT_THRESHOLD:
            return None
        value = value * 10 + digit
    return value if value <= 13 else None


def read_counts(image: np.ndarray, directory: Path = DIGIT_DIR) -> dict[int, int | None]:
    return {seat: read_count(image, seat, directory) for seat in BADGES}


def extract_digits(image: np.ndarray, seat: int, value: int, tag: str, directory: Path = DIGIT_DIR) -> int:
    parts = _components(image, BADGES[seat])
    digits = [int(c) for c in str(value)]
    if len(parts) != len(digits):
        raise ValueError(f"座位 {seat} 找到 {len(parts)} 個數字，標註 {value}")
    for i, (glyph, d) in enumerate(zip(parts, digits)):
        write_image(directory / str(d) / f"{tag}_{seat}_{i}.png", (glyph * 255).astype(np.uint8))
    load_digits.cache_clear()
    return len(parts)
