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
from dataclasses import dataclass, field, replace
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
    stacked: bool = field(default=False, compare=False)
    """這張是槓的一部分（看到第四張疊在中間那張上，見 _side_kongs、_top_kongs）。"""


TOP_SKEW = 0.001
TOP_SKEW_CENTER = (960, 185)
"""對家那排的透視：牌的直邊往畫面中央上方收，越靠兩端越斜（最右的起點那張上緣比下緣偏左約 14 像素）。
拉正：那一列 y 的 x 以 TOP_SKEW_CENTER 為中心縮放 1 + TOP_SKEW × (y − 185)，y＝185 那一列不動（位置照舊）。
10/29 18:43 對家最右那組八筒碰，最右那張只有 0.47（讀成一萬 0.57）湊不成碰；已知答案 9 張截圖 39 張牌：
不拉正對 37 錯 2，TOP_SKEW 0.0005 對 38 錯 0、0.0008～0.0013 全對（負的方向變差）。"""


def top_row(image: np.ndarray) -> np.ndarray:
    """拉正後的對家那排（TOP_MELD_ROW 的範圍，x 從 TOP_MELD_ROW 左緣起算）。"""
    left, top, right, bottom = TOP_MELD_ROW
    center_x, center_y = TOP_SKEW_CENTER
    ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
    map_x = center_x + (xs - center_x) * (1 + TOP_SKEW * (ys - center_y))
    return cv2.remap(image, map_x, ys, cv2.INTER_LINEAR)


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
    """實戰收集的副露模板（每座位每種牌：人工標註的 manual_* 全部，再補最新的幾張）。"""
    result: dict[str, list[tuple[int, np.ndarray]]] = {}
    if not directory.exists():
        return result
    for seat_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for folder in sorted(path for path in seat_dir.iterdir() if path.is_dir()):
            manual = sorted(folder.glob("manual_*.png"))
            others = sorted(path for path in folder.glob("*.png") if not path.name.startswith("manual_"))
            for path in manual + others[-max(0, per_tile - len(manual)):] if per_tile > len(manual) else manual:
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

    def around(self, position: int, margin: int) -> np.ndarray:
        """crop 四周各多 margin 像素（萬子數字比對時容許位置差幾像素）。"""
        height, width = self.templates[self.which[position]][1].shape[:2]
        other = self.cross[position]
        if self.axis == 1:
            top, left = position - margin, other - margin
        else:
            top, left = other - margin, position - margin
        top, left = max(0, top), max(0, left)
        return self.area[top:top + height + 2 * margin, left:left + width + 2 * margin]

    def crop(self, position: int) -> np.ndarray:
        height, width = self.templates[self.which[position]][1].shape[:2]
        other = self.cross[position]
        if self.axis == 1:
            return self.area[position:position + height, other:other + width]
        return self.area[other:other + height, position:position + width]


KONG_BAND_SHARE = 0.6
KONG_BAND_ROWS = 3
KONG_PAIR_SPAN = 75
"""上家、下家的槓：第四張疊在中間那張上、往畫面上方偏，拉直後兩張一樣的牌之間露出一段藍色牌側（一般的碰三張牌面相連、
沒有藍色）。兩張一樣的牌相距不超過 KONG_PAIR_SPAN、中間有 KONG_BAND_ROWS 列以上藍色超過 KONG_BAND_SHARE 就是槓。
10/28 槓截圖：上家八萬槓、東槓抓到；上家九筒槓在直條起點、第一張被切掉，抓不到；已知答案的截圖沒有誤判。"""
TOP_STACK_RISE = 8
"""對家的槓：疊起來的中間那張牌面上緣比兩旁高（10/28 對家八萬槓 141 對 159；一般的碰、吃 41 張都在 158～159）。"""
NUMBER_HALF = {"left": "right", "right": "left", "top": "bottom"}
MANZU_SHIFT = 6
"""萬子數字比對時，允許和定位的位置差幾像素。"""
"""萬子的數字在牌面上半；轉向後在哪一邊。"""


def _half(image: np.ndarray, side: str) -> np.ndarray:
    height, width = image.shape[:2]
    return {"right": image[:, width // 2:], "left": image[:, :width // 2],
            "bottom": image[height // 2:], "top": image[:height // 2]}[side]


def _refine_manzu(scan: Scan, hits: list[MeldTile], offset: int, seat: str,
                  templates: list[tuple[int, np.ndarray]] | None = None) -> list[MeldTile]:
    """萬子只比數字那一半：下半部的「萬」字每張都一樣、佔大半張，整張比對時一萬、六萬、八萬容易混
    （10/22 對家碰六萬讀成一萬、亮牌時側躺的八萬讀成一萬）。

    templates：數字只和牌河模板（轉向）比。實戰收集的副露模板是同一個位置切的、分數一定比較高，但只有部分萬子有：
    10/28 對家沒有四萬的專用模板、二萬有十幾張，碰四萬最右那張就讀成二萬；只用牌河模板時已知答案的 9 張萬子全對。"""
    refined = []
    for hit in hits:
        position = hit.position - offset
        crop = scan.crop(position)
        if hit.tile >= 9:
            refined.append(MeldTile(hit.tile, hit.score, hit.position, crop))
            continue
        shape = crop.shape
        # 位置可能偏幾像素（10/28 對家碰四萬最右那張用專用模板定位、偏了，數字只比到一半而讀成一萬）：在稍大的範圍裡找數字
        area = scan.around(position, MANZU_SHIFT)
        best_tile, best_score = hit.tile, -1.0
        for tile, template in (templates if templates is not None else scan.templates):
            if tile >= 9 or template.shape != shape:
                continue
            number = _half(template, NUMBER_HALF[seat])
            if number.shape[0] > area.shape[0] or number.shape[1] > area.shape[1]:
                continue
            score = float(cv2.matchTemplate(area, number, cv2.TM_CCOEFF_NORMED).max())
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

        self.side_base = {seat: [(tile, sized(cv2.rotate(image, rotation), (width - 6, height)))
                                 for tile, image in templates for height in SIDE_HEIGHTS]
                          for seat, rotation in SIDE_ROTATION.items()}
        """只有牌河模板（萬子數字比對用，見 _refine_manzu）。"""
        self.side = {seat: [(tile, sized(image, (width - 6, height)))
                            for tile, image in seat_templates.get(seat, []) for height in SIDE_HEIGHTS]
                     + self.side_base[seat] for seat in SIDE_ROTATION}
        self.top_base = [(tile, sized(cv2.rotate(image, cv2.ROTATE_180), (width_, round(width_ * TOP_ASPECT))))
                         for tile, image in templates for width_ in TOP_WIDTHS]
        self.top = [(tile, sized(image, (width_, round(width_ * TOP_ASPECT))))
                    for tile, image in seat_templates.get("top", []) for width_ in TOP_WIDTHS] + self.top_base

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
        hits = _refine_manzu(scan, hits, start, seat, self.side_base[seat])
        hits = _side_kongs(strip, hits)
        return hits if seat == "left" else hits[::-1]

    def _top(self, image: np.ndarray) -> list[MeldTile]:
        left, _, right, _ = TOP_MELD_ROW
        row = top_row(image)
        start = max(left, right - SEARCH_TILES * TOP_PITCH)  # 對家從右邊開始擺
        scan = Scan(row[:, start - left:], self.top, axis=0)
        hits = _peaks(scan.best, scan.label, TOP_MIN_SCORE, TOP_SPACING, start)
        if hits and hits[0].position - start < TOP_PITCH:
            start = left
            scan = Scan(row, self.top, axis=0)
            hits = _peaks(scan.best, scan.label, TOP_MIN_SCORE, TOP_SPACING, left)
        return _top_kongs(image, _refine_manzu(scan, hits, start, "top", self.top_base))[::-1]

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
            area, templates, axis, offset = top_row(image), self.top, 0, TOP_MELD_ROW[0]
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

    @staticmethod
    def concealed_kongs(image: np.ndarray) -> dict[str, int]:
        return concealed_kongs(image)

    def read(self, image: np.ndarray) -> dict[str, list[MeldTile]]:
        """三家正面朝上的牌，依擺放順序（從副露起點那端開始）。"""
        return {"left": self._side(image, "left"), "right": self._side(image, "right"), "top": self._top(image)}


FACE_DOWN_SIDE_SHARE = 0.8
"""直條上一列（橫跨整個寬度）藍色牌背至少佔這麼多＝平躺的牌背（暗槓）。立著的手牌只佔一邊、另一邊露出深綠桌面，
明牌副露是白的。"""
FACE_DOWN_SIDE_MIN = 30
"""直條上連著的平躺牌背（間隔不到 FACE_DOWN_SIDE_GAP 的合併）至少這麼長才算一組暗槓（一張約 30）。"""
FACE_DOWN_SIDE_GAP = 30
TOP_FLAT_ROWS = ((146, 157), (178, 204))
"""對家那排：平躺的牌背在下面那段是藍的、上面那段不是（立著的手牌兩段都藍、疊在中間那張也是）。"""
TOP_FLAT_WIDTH = (35, 65)
TOP_FLAT_PAIR = (90, 130)
"""對家的暗槓：兩段平躺牌背（各約 50 寬）、中間隔一張疊高的，兩段起點相距約 107。"""


def _blue_backs(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.int16)
    blue, red = pixels[..., 0], pixels[..., 2]
    return (blue > 150) & (red < 90) & (blue - red > 100)


def _runs(flags: np.ndarray, min_length: int, merge_gap: int = 0) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for index, flag in enumerate(flags):
        if not flag:
            continue
        if runs and index - runs[-1][1] <= merge_gap:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1])
    return [(start, end) for start, end in runs if end - start >= min_length]


def concealed_kongs(image: np.ndarray) -> dict[str, int]:
    """三家桌邊的暗槓（三張牌背平躺、第四張疊在中間）有幾組。牌背朝上讀不到是哪張，只數組數。
    10/29 槓截圖：對家 5 張、上家下家 7 張看得到暗槓的都抓到；其他 60 多張牌局截圖沒有誤判（結算等牌局外畫面不算）。"""
    counts = {}
    for seat in ("left", "right"):
        cover = _blue_backs(straighten(image, seat)).mean(axis=1)
        counts[seat] = len(_runs(cover >= FACE_DOWN_SIDE_SHARE, FACE_DOWN_SIDE_MIN, FACE_DOWN_SIDE_GAP))
    left, _, right, _ = TOP_MELD_ROW
    blue = _blue_backs(image[:, left:right])
    (upper_top, upper_bottom), (lower_top, lower_bottom) = TOP_FLAT_ROWS
    flat = (blue[lower_top:lower_bottom].mean(axis=0) > 0.8) & (blue[upper_top:upper_bottom].mean(axis=0) < 0.2)
    pieces = [run for run in _runs(flat, 1) if TOP_FLAT_WIDTH[0] <= run[1] - run[0] <= TOP_FLAT_WIDTH[1]]
    pairs, used = 0, set()
    for index, (start, _) in enumerate(pieces):
        partner = next((other for other in range(index + 1, len(pieces)) if other not in used
                        and TOP_FLAT_PAIR[0] <= pieces[other][0] - start <= TOP_FLAT_PAIR[1]), None)
        if index not in used and partner is not None:
            pairs += 1
            used |= {index, partner}
    counts["top"] = pairs
    return counts


def _band_rows(strip: np.ndarray) -> np.ndarray:
    """直條每一列有多少比例是藍色牌側（紅色幾乎沒有、藍明顯比綠多；牌面、桌布都不是）。"""
    pixels = strip.astype(np.int16)
    blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return ((blue > 130) & (red < 40) & (blue - green > 25)).mean(axis=1)


def _side_kongs(strip: np.ndarray, hits: list[MeldTile]) -> list[MeldTile]:
    """上家、下家：兩張一樣的牌之間有藍色牌側＝槓。中間讀到的牌（疊起來那張的側面常被讀成別的）換成一張同樣的牌，
    三張都標成 stacked。"""
    ordered = sorted(hits, key=lambda hit: hit.position)
    band = _band_rows(strip)
    index = 0
    while index < len(ordered):
        a = ordered[index]
        for other in range(index + 1, len(ordered)):
            b = ordered[other]
            if b.position - a.position > KONG_PAIR_SPAN:
                break
            between = band[a.position + 20:b.position + 4]
            if b.tile != a.tile or (between > KONG_BAND_SHARE).sum() < KONG_BAND_ROWS:
                continue
            middle = MeldTile(a.tile, min(a.score, b.score), (a.position + b.position) // 2, None, True)
            ordered[index:other + 1] = [replace(a, stacked=True), middle, replace(b, stacked=True)]
            index += 2
            break
        index += 1
    return ordered


def _face_top(image: np.ndarray, hit: MeldTile) -> int | None:
    """對家那張牌（hit.position 是牌的左緣）牌面上緣的 y。"""
    _, top, _, bottom = TOP_MELD_ROW
    area = image[top - 40:bottom, hit.position + 12:hit.position + 40]
    white = (area.min(axis=2) > 200).mean(axis=1)
    rows = np.flatnonzero(white > 0.6)
    return int(rows[0]) + top - 40 if len(rows) else None


def _top_kongs(image: np.ndarray, hits: list[MeldTile]) -> list[MeldTile]:
    """對家：碰的中間那張牌面比兩旁高 TOP_STACK_RISE 以上＝槓（第四張疊在上面），三張標成 stacked。"""
    ordered = sorted(hits, key=lambda hit: hit.position)
    for kind, start, length in meld_spans([hit.tile for hit in ordered]):
        if kind != "pung":
            continue
        tops = [_face_top(image, hit) for hit in ordered[start:start + length]]
        if None not in tops and tops[1] <= min(tops[0], tops[2]) - TOP_STACK_RISE:
            ordered[start:start + length] = [replace(hit, stacked=True) for hit in ordered[start:start + length]]
    return ordered


def meld_spans(tiles: list[int]) -> list[tuple[str, int, int]]:
    """依擺放順序分組：三張一樣＝碰、同花色連續三張＝吃；回傳每組 (種類, 起點, 張數)，分不出的牌跳過。
    槓不會四張並排（第四張疊在中間那張上，由 MeldTile.stacked 標出）：四張一樣並排是多讀了一張（10/28 對家
    九萬碰旁邊的七萬碰讀成九萬，被當成九萬槓、七萬碰不見），當成碰加一張分不出的牌。"""
    spans = []
    index = 0
    while index < len(tiles):
        rest = tiles[index:]
        if len(rest) >= 3 and len(set(rest[:3])) == 1:
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
