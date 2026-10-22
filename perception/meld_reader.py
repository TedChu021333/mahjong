"""對手的副露：上家、下家側躺疊成一排、對家倒著擺成一列，用牌河模板轉向後滑動比對讀出來。

    python -m perception.meld_reader <截圖>...          # 印出三家讀到的副露

上家、下家：用 MELD_COLUMN_CORNERS（使用者在亮牌截圖標的頭尾兩張牌面的角）把整排透視拉直成直條（副露之間有空隙，
不能平均切），牌河模板轉 90 度、縮成直條上一格的大小，沿直條上下滑動比對；對家：模板轉 180 度沿那列左右滑動。
分數高的位置就是一張正面朝上的牌，立著的手牌（藍色牌背）與空隙分數很低。
10/22 驗證：亮牌兩排 32 張錯 2 張（側躺的八萬讀成一萬），實戰的碰白、碰一筒、吃四五六筒、碰發全對，
沒有副露的畫面什麼都沒讀到。
速度：全部模板（每種 5 張 × 3 種大小）掃整排一次讀三家要 7.6 秒；改成每種 3 張 × 2 種大小、先只掃副露那一端
（SEARCH_TILES 張），一半解析度則會變不準（試過）。實戰在背景執行緒讀（OpenCV 比對時放開 GIL），只在有人吃碰後讀。
先找「一般牌面」再交給 TileClassifier 認牌的做法也試過：平均起來的牌面太糊，位置會偏半張，認不出來。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from game.rules import Meld
from game.tiles import parse, tile_name
from perception.capture import read_image
from perception.config import (MELD_COLUMN_CORNERS, MELD_STRIP_MARGIN, MELD_STRIP_SIZE, RIVER_TEMPLATE_DIR,
                               TOP_MELD_ROW)

SIDE_ROTATION = {"left": cv2.ROTATE_90_CLOCKWISE, "right": cv2.ROTATE_90_COUNTERCLOCKWISE}
SIDE_HEIGHTS = (27, 31)
"""直條上一張牌可能的高度（透視拉直後近似，不同截圖差幾像素）。"""
TOP_WIDTHS = (50, 55)
TOP_ASPECT = 0.75
"""對家牌面的高／寬：倒著、從遠處斜看，比正面扁。10/22 已知答案的 8 張截圖（33 張牌）：0.9 時對 22、漏 11、錯 8，
0.75 時對 31、漏 2、錯 2（0.7 以下又變差）。"""
SIDE_MIN_SCORE = 0.45
TOP_MIN_SCORE = 0.55
SIDE_SPACING = 24
"""直條上兩張牌至少相隔幾像素（一張約 27～32）。原本 20：10/22 收集的一萬模板（大部分是白的）把最後一張牌下方 21 像素
的白色牌身讀成一萬（0.46）。"""
TOP_SPACING = 45
"""對家那列兩張牌至少相隔幾像素（一張約 54；亮牌時四萬和北之間的空隙曾在 34 像素處多抓一張）。"""
TEMPLATES_PER_TILE = 3
MELD_TEMPLATE_DIR = Path("train/meld_templates")
"""實戰收集的副露專用模板：<座位 left/right/top>/<牌代碼>/*.png，已經是直條／那一列上的方向（hint.meld_watch 收集）。"""
SEAT_TEMPLATES_PER_TILE = 4
SEARCH_TILES = 8
"""先只掃副露那一端這麼多張的範圍；讀到最後一格附近才掃整排。"""
SIDE_PITCH = 32
TOP_PITCH = 54
GUIDED_PITCHES = {"side": range(24, 39, 2), "top": range(48, 63, 2)}
"""引導搜尋吃時，三張牌之間可能的間距。"""
GUIDED_MIN_SCORE = 0.42
"""引導搜尋：三張平均分數到這個以上才採用（單張讀不到的對家倒著的吃，19:34 四五六筒單張約 0.5）。"""
GUIDED_MARGIN = 0.02
"""最好的吃要比其他可能的吃（不同的三連號）平均高這麼多，分不出來就不採用。"""


@dataclass(frozen=True)
class MeldTile:
    tile: int
    score: float
    position: int
    """在直條（上家、下家）或那一列（對家）上的位置，用來排順序。"""
    crop: np.ndarray | None = field(default=None, compare=False, repr=False)
    """這張牌在直條／那一列上的切圖（轉向後、和模板同方向），收集專用模板用。"""


def _strip_transform(seat: str) -> np.ndarray:
    first_left, first_right, last_right, last_left = MELD_COLUMN_CORNERS[seat]
    width, height = MELD_STRIP_SIZE
    src = np.float32([first_left, first_right, last_right, last_left])
    dst = np.float32([[0, MELD_STRIP_MARGIN], [width, MELD_STRIP_MARGIN],
                      [width, height + MELD_STRIP_MARGIN], [0, height + MELD_STRIP_MARGIN]])
    return cv2.getPerspectiveTransform(src, dst)


def straighten(image: np.ndarray, seat: str) -> np.ndarray:
    """上家或下家那一排拉直成直條（上 = 離自己遠的那端）。"""
    width, height = MELD_STRIP_SIZE
    return cv2.warpPerspective(image, _strip_transform(seat), (width, height + 2 * MELD_STRIP_MARGIN))


def load_seat_templates(directory: Path = MELD_TEMPLATE_DIR,
                        per_tile: int = SEAT_TEMPLATES_PER_TILE) -> dict[str, list[tuple[int, np.ndarray]]]:
    """實戰收集的副露模板（每座位每種牌取最新的幾張）。"""
    result: dict[str, list[tuple[int, np.ndarray]]] = {}
    if not directory.exists():
        return result
    for seat_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for folder in sorted(path for path in seat_dir.iterdir() if path.is_dir()):
            for path in sorted(folder.glob("*.png"))[-per_tile:]:
                image = read_image(path)
                if image is not None:
                    result.setdefault(seat_dir.name, []).append((parse(folder.name)[0], image))
    return result


def _peaks(best: np.ndarray, label: np.ndarray, min_score: float, spacing: int,
           offset: int = 0) -> list[MeldTile]:
    taken = np.zeros(len(best), bool)
    hits = []
    for index in np.argsort(-best):
        if best[index] < min_score:
            break
        if taken[max(0, index - spacing):index + spacing].any():
            continue
        taken[index] = True
        hits.append(MeldTile(int(label[index]), float(best[index]), int(index) + offset))
    return sorted(hits, key=lambda hit: hit.position)


class Scan:
    """沿直條（axis=1，每一列）或那一列（axis=0，每一欄）每個位置最好的比對：分數、牌、哪個模板、另一個方向的偏移。"""

    def __init__(self, area: np.ndarray, templates: list[tuple[int, np.ndarray]], axis: int) -> None:
        length = area.shape[0] if axis == 1 else area.shape[1]
        self.area, self.templates, self.axis = area, templates, axis
        self.best = np.full(length, -1.0)
        self.label = np.full(length, -1)
        self.which = np.zeros(length, int)
        self.cross = np.zeros(length, int)
        for number, (tile, template) in enumerate(templates):
            if template.shape[0] > area.shape[0] or template.shape[1] > area.shape[1]:
                continue
            result = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
            scores, cross = result.max(axis=axis), result.argmax(axis=axis)
            index = np.flatnonzero(scores > self.best[:len(scores)])
            self.best[index] = scores[index]
            self.label[index] = tile
            self.which[index] = number
            self.cross[index] = cross[index]

    def crop(self, position: int) -> np.ndarray:
        height, width = self.templates[self.which[position]][1].shape[:2]
        other = self.cross[position]
        if self.axis == 1:
            return self.area[position:position + height, other:other + width]
        return self.area[other:other + height, position:position + width]


NUMBER_HALF = {"left": "right", "right": "left", "top": "bottom"}
"""萬子的數字在牌面上半；轉向後在哪一邊。"""


def _half(image: np.ndarray, side: str) -> np.ndarray:
    height, width = image.shape[:2]
    return {"right": image[:, width // 2:], "left": image[:, :width // 2],
            "bottom": image[height // 2:], "top": image[:height // 2]}[side]


def _refine_manzu(scan: Scan, hits: list[MeldTile], offset: int, seat: str) -> list[MeldTile]:
    """萬子只比數字那一半：下半部的「萬」字每張都一樣、佔大半張，整張比對時一萬、六萬、八萬容易混
    （10/22 對家碰六萬讀成一萬、亮牌時側躺的八萬讀成一萬）。"""
    refined = []
    for hit in hits:
        position = hit.position - offset
        crop = scan.crop(position)
        if hit.tile >= 9:
            refined.append(MeldTile(hit.tile, hit.score, hit.position, crop))
            continue
        shape = crop.shape
        number = _half(crop, NUMBER_HALF[seat])
        best_tile, best_score = hit.tile, -1.0
        for tile, template in scan.templates:
            if tile >= 9 or template.shape != shape:
                continue
            score = float(cv2.matchTemplate(number, _half(template, NUMBER_HALF[seat]), cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_tile, best_score = tile, score
        refined.append(MeldTile(best_tile, hit.score, hit.position, crop))
    return refined


class MeldReader:
    def __init__(self, templates: list[tuple[int, np.ndarray]],
                 seat_templates: dict[str, list[tuple[int, np.ndarray]]] | None = None) -> None:
        """templates：正立的牌河模板（轉向後用）；seat_templates：實戰收集、已經同方向的副露模板（優先）。"""
        width, _ = MELD_STRIP_SIZE
        seat_templates = seat_templates or {}

        def sized(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        self.side = {seat: [(tile, sized(image, (width - 6, height)))
                            for tile, image in seat_templates.get(seat, []) for height in SIDE_HEIGHTS]
                     + [(tile, sized(cv2.rotate(image, rotation), (width - 6, height)))
                        for tile, image in templates for height in SIDE_HEIGHTS]
                     for seat, rotation in SIDE_ROTATION.items()}
        self.top = [(tile, sized(image, (width_, round(width_ * TOP_ASPECT))))
                    for tile, image in seat_templates.get("top", []) for width_ in TOP_WIDTHS] \
            + [(tile, sized(cv2.rotate(image, cv2.ROTATE_180), (width_, round(width_ * TOP_ASPECT))))
               for tile, image in templates for width_ in TOP_WIDTHS]

    @classmethod
    def from_directory(cls, directory: Path = RIVER_TEMPLATE_DIR, per_tile: int = TEMPLATES_PER_TILE,
                       seat_dir: Path = MELD_TEMPLATE_DIR) -> "MeldReader | None":
        """每種牌取幾張牌河模板（人工標註的優先）；沒有模板時回傳 None。"""
        if not directory.exists():
            return None
        templates = []
        for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
            manual = sorted(folder.glob("manual_*.png"))
            others = sorted(path for path in folder.glob("*.png") if not path.name.startswith("manual_"))
            for path in (manual[:1] + others)[:per_tile]:
                image = read_image(path)
                if image is not None:
                    templates.append((parse(folder.name)[0], image))
        if not templates:
            return None
        return cls(templates, load_seat_templates(seat_dir))

    def _side(self, image: np.ndarray, seat: str) -> list[MeldTile]:
        strip = straighten(image, seat)
        reach = MELD_STRIP_MARGIN + SEARCH_TILES * SIDE_PITCH
        # 上家從直條上端（遠端）開始擺、下家從下端（近端）開始擺
        start = 0 if seat == "left" else max(0, strip.shape[0] - reach)
        scan = Scan(strip[start:start + reach], self.side[seat], axis=1)
        hits = _peaks(scan.best, scan.label, SIDE_MIN_SCORE, SIDE_SPACING, start)
        near_end = hits and (hits[-1].position - start > reach - 2 * SIDE_PITCH if seat == "left"
                             else hits[0].position - start < 2 * SIDE_PITCH)
        if near_end:  # 副露很多：整排都掃
            start = 0
            scan = Scan(strip, self.side[seat], axis=1)
            hits = _peaks(scan.best, scan.label, SIDE_MIN_SCORE, SIDE_SPACING)
        hits = _refine_manzu(scan, hits, start, seat)
        return hits if seat == "left" else hits[::-1]

    def _top(self, image: np.ndarray) -> list[MeldTile]:
        left, top, right, bottom = TOP_MELD_ROW
        start = max(left, right - SEARCH_TILES * TOP_PITCH)  # 對家從右邊開始擺
        scan = Scan(image[top:bottom, start:right], self.top, axis=0)
        hits = _peaks(scan.best, scan.label, TOP_MIN_SCORE, TOP_SPACING, start)
        if hits and hits[0].position - start < TOP_PITCH:
            start = left
            scan = Scan(image[top:bottom, left:right], self.top, axis=0)
            hits = _peaks(scan.best, scan.label, TOP_MIN_SCORE, TOP_SPACING, left)
        return _refine_manzu(scan, hits, start, "top")[::-1]

    def guided_chow(self, image: np.ndarray, seat: str, tile: int) -> list[MeldTile] | None:
        return self.guided_meld(image, seat, tile, ("chow",))

    def guided_meld(self, image: np.ndarray, seat: str, tile: int,
                    kinds: tuple[str, ...] = ("chow", "pung")) -> list[MeldTile] | None:
        """已知這家吃或碰了 tile（牌河看到被拿走的那張）：可能的組（碰＝三張 tile；吃＝包含 tile 的三連號、三張任意順序）
        在不同間距下整組比對，最像的位置與牌（分數不夠、或有兩種組合分不出時 None）。位置的單位同 read()。
        10/22 對家倒著的吃（四五六筒）、碰一條（倒著的鳥）單張分數都只有 0.5 左右，整組比對才找得到。"""
        runs: list[tuple[int, ...]] = []
        if "pung" in kinds:
            runs.append((tile,) * 3)
        if "chow" in kinds and tile < 27:
            suit, rank = divmod(tile, 9)
            runs += [tuple(suit * 9 + r for r in range(start, start + 3))
                     for start in range(rank - 2, rank + 1) if 0 <= start and start + 2 <= 8]
        if not runs:
            return None
        if seat == "top":
            left, top, right, bottom = TOP_MELD_ROW
            area, templates, axis, offset = image[top:bottom, left:right], self.top, 0, left
            pitches = GUIDED_PITCHES["top"]
        else:
            area, templates, axis, offset = straighten(image, seat), self.side[seat], 1, 0
            pitches = GUIDED_PITCHES["side"]
        scans = {t: Scan(area, [item for item in templates if item[0] == t], axis)
                 for t in {t for run in runs for t in run}}
        per_run: dict[tuple[int, ...], tuple[float, list[MeldTile]]] = {}
        for run in runs:
            for order in set(permutations(run)):
                profiles = [scans[t].best for t in order]
                length = min(len(profile) for profile in profiles)
                for pitch in pitches:
                    span = length - 2 * pitch
                    if span <= 0:
                        continue
                    total = profiles[0][:span] + profiles[1][pitch:pitch + span] + profiles[2][2 * pitch:2 * pitch + span]
                    start = int(np.argmax(total))
                    average = float(total[start]) / 3
                    if average > per_run.get(run, (-1.0, []))[0]:
                        hits = [MeldTile(t, float(scans[t].best[start + i * pitch]), start + i * pitch + offset,
                                         scans[t].crop(start + i * pitch)) for i, t in enumerate(order)]
                        per_run[run] = (average, hits)
        if not per_run:
            return None
        ranked = sorted(per_run.values(), key=lambda item: -item[0])
        best_average, hits = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_average < GUIDED_MIN_SCORE or best_average - runner_up < GUIDED_MARGIN:
            return None
        return hits

    def read(self, image: np.ndarray) -> dict[str, list[MeldTile]]:
        """三家正面朝上的牌，依擺放順序（從副露起點那端開始）。"""
        return {"left": self._side(image, "left"), "right": self._side(image, "right"), "top": self._top(image)}


def meld_spans(tiles: list[int]) -> list[tuple[str, int, int]]:
    """依擺放順序分組：四張一樣＝槓、三張一樣＝碰、同花色連續三張＝吃；回傳每組 (種類, 起點, 張數)，分不出的牌跳過。"""
    spans = []
    index = 0
    while index < len(tiles):
        rest = tiles[index:]
        if len(rest) >= 4 and len(set(rest[:4])) == 1:
            spans.append(("kong", index, 4))
            index += 4
        elif len(rest) >= 3 and len(set(rest[:3])) == 1:
            spans.append(("pung", index, 3))
            index += 3
        elif len(rest) >= 3 and _is_run(rest[:3]):
            spans.append(("chow", index, 3))
            index += 3
        else:
            index += 1
    return spans


def group_melds(tiles: list[int]) -> tuple[list[Meld], list[int]]:
    """meld_spans 分好的組（Meld）與分不出的牌。"""
    melds, used = [], set()
    for kind, start, length in meld_spans(tiles):
        group = tiles[start:start + length]
        melds.append(Meld(kind, tuple(sorted(group)) if kind == "chow" else tuple(group), from_player=-1))
        used.update(range(start, start + length))
    return melds, [tile for index, tile in enumerate(tiles) if index not in used]


def _is_run(tiles: list[int]) -> bool:
    ordered = sorted(tiles)
    return ordered[0] < 27 and ordered[0] // 9 == ordered[2] // 9 \
        and ordered[1] == ordered[0] + 1 and ordered[2] == ordered[0] + 2


def describe(read: dict[str, list[MeldTile]]) -> str:
    names = {"left": "上家", "top": "對家", "right": "下家"}
    parts = []
    for seat in ("left", "top", "right"):
        melds, leftover = group_melds([hit.tile for hit in read.get(seat, [])])
        if melds or leftover:
            text = " ".join("".join(tile_name(t) for t in meld.tiles) for meld in melds)
            if leftover:
                text += f"（另有 {''.join(tile_name(t) for t in leftover)}）"
            parts.append(f"{names[seat]} {text.strip()}")
    return "；".join(parts) or "沒有副露"


def main() -> None:
    parser = argparse.ArgumentParser(description="讀三家的副露")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    reader = MeldReader.from_directory()
    if reader is None:
        raise SystemExit(f"找不到牌河模板：{RIVER_TEMPLATE_DIR}")
    for path in args.images:
        image = read_image(path)
        if image is None:
            print(f"{path}：讀不到圖")
            continue
        read = reader.read(image)
        detail = "；".join(f"{seat} " + " ".join(f"{tile_name(h.tile)}{h.score:.2f}" for h in hits)
                          for seat, hits in read.items())
        print(f"{path.name}：{describe(read)}\n  {detail}")


if __name__ == "__main__":
    main()
