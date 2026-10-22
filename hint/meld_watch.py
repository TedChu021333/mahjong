"""提示迴圈用的對手副露：在背景執行緒讀（一次約 1.8 秒，OpenCV 比對時放開 GIL，不擋出牌），
用牌河推算的碰校正萬子數字，對得上的牌切下來存成專用模板。

- 什麼時候讀：提示迴圈在有人吃碰後 1.5 秒、或場上有副露時每 8 秒叫 request（上一次還沒讀完就略過）。
- 萬子校正：讀到某家碰了萬子 X，牌河那邊推算他碰的萬子卻是 Y（被拿走的那張從牌河讀，較準）、而且 Y 沒有對上
  任何讀到的碰，就改成 Y（10/22 對家碰六萬讀成一萬、亮牌時側躺的八萬讀成一萬）。
- 收集模板：讀到的碰和牌河推算的同一種牌（含校正後的），三張的切圖存到 train/meld_templates/<座位>/<牌>/，
  每座位每種牌最多 MAX_TEMPLATES 張；perception.meld_reader 下次啟動時優先用。
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from game.rules import Meld
from game.tiles import tile_code, tile_name
from perception.capture import write_image
from perception.meld_reader import MELD_TEMPLATE_DIR, MeldReader, MeldTile, _is_run, group_melds, meld_spans

SEATS = ("left", "top", "right")
SEAT_NAMES = {"left": "上家", "top": "對家", "right": "下家"}
MAX_TEMPLATES = 20
"""每座位每種牌最多收集幾張專用模板。"""


def correct_manzu(tiles: list[int], known: list[int]) -> list[int]:
    """讀到的牌（依擺放順序）用牌河推算校正萬子：known 是牌河推算這家副露裡的牌（碰的話是那種牌兩張）。
    推算他碰了萬子 Y、讀到的卻沒有三張 Y 時，找連續三張萬子、而且本身湊不成吃或推算裡有的碰，改成三張 Y
    （10/22 對家碰六萬，讀成一萬一萬一萬、或一萬六萬六萬）。"""
    counts = Counter(known)
    unmatched = [tile for tile, count in counts.items() if count >= 2 and tile < 9 and tiles.count(tile) < 3]
    result = list(tiles)
    index = 0
    while unmatched and index + 3 <= len(result):
        window = result[index:index + 3]
        confirmed_pung = len(set(window)) == 1 and window[0] in counts
        if all(tile < 9 for tile in window) and not _is_run(window) and not confirmed_pung:
            result[index:index + 3] = [unmatched.pop(0)] * 3
            index += 3
        else:
            index += 1
    return result


def describe(melds: dict[str, tuple[Meld, ...]]) -> str:
    parts = [f"{SEAT_NAMES[seat]} " + " ".join("".join(tile_name(t) for t in meld.tiles) for meld in group)
             for seat in SEATS if (group := melds.get(seat))]
    return "；".join(parts) or "沒有副露"


class MeldWatch:
    def __init__(self, reader: MeldReader, template_dir: Path | None = MELD_TEMPLATE_DIR,
                 save: Callable[[Path, np.ndarray], object] = write_image, background: bool = True) -> None:
        self.reader = reader
        self.template_dir = template_dir
        self.save = save
        self.background = background
        self.melds: dict[str, tuple[Meld, ...]] = {}
        """各家讀到、能完整成組的副露（萬子已校正）。"""
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending: tuple[int, dict[str, list[MeldTile]]] | None = None
        self._generation = 0
        """換局時加一：換局前送出的讀取結果作廢。"""
        self.saved_templates = 0
        self.reads = 0

    def new_hand(self) -> None:
        with self._lock:
            self._generation += 1
            self._pending = None
        self.melds = {}

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request(self, frame: np.ndarray) -> bool:
        """開始讀這一幀（上一次還沒讀完就不讀）。"""
        if self.busy():
            return False
        generation = self._generation
        image = frame.copy()

        def work() -> None:
            read = self.reader.read(image)
            with self._lock:
                if generation == self._generation:
                    self._pending = (generation, read)

        if self.background:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
        else:
            work()
        return True

    def collect(self, known: dict[str, list[int]]) -> bool:
        """讀完的結果套用進 self.melds（known：牌河推算的各家副露裡的牌）；有變化回傳 True。"""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return False
        self.reads += 1
        read = pending[1]
        result: dict[str, tuple[Meld, ...]] = {}
        for seat in SEATS:
            hits = read.get(seat, [])
            tiles = correct_manzu([hit.tile for hit in hits], known.get(seat, []))
            melds, _ = group_melds(tiles)
            if melds:
                result[seat] = tuple(melds)
            self._harvest(seat, hits, tiles, known.get(seat, []))
        changed = result != self.melds
        self.melds = result
        return changed

    def _harvest(self, seat: str, hits: list[MeldTile], tiles: list[int], known: list[int]) -> None:
        """讀到（校正後）的碰和牌河推算的對得上：那幾張切圖存成專用模板（標註用校正後的牌）。"""
        if self.template_dir is None:
            return
        confirmed = {tile for tile, count in Counter(known).items() if count >= 2}
        for kind, start, length in meld_spans(tiles):
            tile = tiles[start]
            if kind not in ("pung", "kong") or tile not in confirmed:
                continue
            folder = self.template_dir / seat / tile_code(tile)
            existing = len(list(folder.glob("*.png"))) if folder.exists() else 0
            for number, hit in enumerate(hits[start:start + length]):
                if hit.crop is None or existing >= MAX_TEMPLATES:
                    break
                self.save(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{number}.png", hit.crop)
                existing += 1
                self.saved_templates += 1


SUIT_NAMES = "萬筒條"
RANKS = "一二三四五六七八九"


def _short(meld: Meld) -> str:
    """碰／槓寫一張（「白」「五萬」），吃寫三個數字加花色（「七八九筒」）。"""
    if meld.kind == "chow":
        tiles = sorted(meld.tiles)
        return "".join(RANKS[tile % 9] for tile in tiles) + SUIT_NAMES[tiles[0] // 9]
    return tile_name(meld.tiles[0])


def seat_labels(melds: dict[str, tuple[Meld, ...]], counts: dict[str, int]) -> dict[str, str | None]:
    """三家旁邊小標籤的文字（沒有副露時 None）：「碰 白　吃 七八九筒　槓 二筒」；counts 是推算的副露組數，
    比讀到的多時補「另有 N 組沒讀到」。"""
    labels: dict[str, str | None] = {}
    words = {"chow": "吃", "pung": "碰", "kong": "槓"}
    for seat in SEATS:
        group = melds.get(seat, ())
        parts = [f"{words[meld.kind]} {_short(meld)}" for meld in group]
        missing = counts.get(seat, 0) - len(group)
        if missing > 0:
            parts.append(f"另有 {missing} 組沒讀到")
        labels[seat] = f"{SEAT_NAMES[seat]}　" + "　".join(parts) if parts else None
    return labels
