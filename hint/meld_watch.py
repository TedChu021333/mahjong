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
HARVEST_MIN_SCORE = 0.5
"""引導搜尋找到的牌，單張分數到這個以上才收成模板。"""
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


CONFUSABLE_MANZU = {0: (5, 7), 5: (0, 7), 7: (0, 5)}
"""側躺、倒著時數字容易認錯的萬子：一萬、六萬、八萬（10/22 八萬讀成一萬好幾次、六萬讀成一萬）。"""


def fix_confusable_manzu(tiles: list[int]) -> list[int]:
    """連續三張萬子湊不成吃或碰時，把其中一張一／六／八萬換成另外兩種之一；只有一種換法湊得成才採用
    （20:08 對家七九八萬讀成七萬九萬一萬）。"""
    result = list(tiles)
    index = 0
    while index + 3 <= len(result):
        window = result[index:index + 3]
        if all(tile < 9 for tile in window) and not _is_run(window) and len(set(window)) != 1:
            options = set()
            for position, tile in enumerate(window):
                for other in CONFUSABLE_MANZU.get(tile, ()):
                    changed = list(window)
                    changed[position] = other
                    if _is_run(changed) or len(set(changed)) == 1:
                        options.add(tuple(changed))
            if len(options) == 1:
                result[index:index + 3] = list(options.pop())
                index += 3
                continue
        index += 1
    return result


def fix_odd_middle(tiles: list[int]) -> list[int]:
    """連續三張同花色的數牌、兩張一樣、另一張只差 1～2（「九七九條」）：吃不可能有重複的牌，當成三張一樣的碰。
    槓的第四張疊在中間那張上，那張最常讀錯（21:04 對家九條槓讀成九條七條九條、整組被丟掉）；
    倒著、側躺的七八九條、一六八萬也常互相讀錯。"""
    result = list(tiles)
    index = 0
    while index + 3 <= len(result):
        window = result[index:index + 3]
        counts = Counter(window)
        if len(counts) == 2 and all(tile < 27 for tile in window):
            (major, major_count), (minor, _) = counts.most_common()
            if major_count == 2 and major // 9 == minor // 9 and abs(major - minor) <= 2 and not _is_run(window):
                result[index:index + 3] = [major] * 3
                index += 3
                continue
        index += 1
    return result


def fill_gaps(hits: list[MeldTile]) -> list[MeldTile]:
    """兩張一樣的牌中間剛好空出一張牌的位置（另一張分數不夠沒讀到），補上中間那張（21:04 下家條子碰只讀到兩張）。
    一張牌的間距用相鄰讀到的牌估。"""
    if len(hits) < 2:
        return list(hits)
    ordered = sorted(hits, key=lambda hit: hit.position)
    gaps = [b.position - a.position for a, b in zip(ordered, ordered[1:])]
    pitch = sorted(gaps)[len(gaps) // 2]
    if len(gaps) < 3:  # 讀到的太少，間距估不準：用比較小的那個
        pitch = min(gaps)
    filled = [ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        if a.tile == b.tile and 1.6 * pitch <= b.position - a.position <= 2.4 * pitch:
            filled.append(MeldTile(a.tile, 0.0, (a.position + b.position) // 2))
        filled.append(b)
    return filled


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
        self._pending: tuple[int, dict[str, list[MeldTile]], list[tuple[str, list[MeldTile]]]] | None = None
        """背景讀完的 (換局編號, 讀到的牌, 引導搜尋找到的吃)。"""
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

    def request(self, frame: np.ndarray, hints: dict[str, list] | None = None) -> bool:
        """開始讀這一幀（上一次還沒讀完就不讀）。hints：各家吃碰了、還沒讀到的牌（牌河推算，(牌, "chow"／"pung"／None)
        或只有牌＝可能是吃），用 MeldReader.guided_meld 引導搜尋（對家倒著的牌單張分數只有 0.5 左右）。"""
        hints = {seat: [item if isinstance(item, tuple) else (item, "chow") for item in items]
                 for seat, items in (hints or {}).items()}
        if self.busy():
            return False
        generation = self._generation
        image = frame.copy()

        def work() -> None:
            read = self.reader.read(image)
            found_chows: list[tuple[str, list[MeldTile]]] = []
            guided = getattr(self.reader, "guided_meld", None)
            for seat, items in hints.items():
                for tile, kind in items:
                    kinds = (kind,) if kind else ("chow", "pung")
                    found = guided(image, seat, tile, kinds) if guided is not None else None
                    current = read.get(seat, [])
                    grouped = _grouped_hits(current)
                    if found and not _overlaps(found, grouped):
                        # 沒成組的單張（多半是讀錯）在同一位置的換成引導搜尋找到的
                        kept = [hit for hit in current if hit in grouped or not _overlaps([hit], found)]
                        read[seat] = kept + found
                        found_chows.append((seat, found))
            with self._lock:
                if generation == self._generation:
                    self._pending = (generation, read, found_chows)

        if self.background:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
        else:
            work()
        return True

    def collect(self, known: dict[str, list[int]], counts: dict[str, int] | None = None) -> bool:
        """讀完的結果套用進 self.melds（known：牌河推算的各家副露裡的牌）；有變化回傳 True。"""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return False
        self.reads += 1
        read = pending[1]
        result: dict[str, tuple[Meld, ...]] = {}
        for seat in SEATS:
            hits = sorted(fill_gaps(read.get(seat, [])), key=lambda hit: hit.position, reverse=seat != "left")
            tiles = correct_manzu([hit.tile for hit in hits], known.get(seat, []))
            tiles = fix_odd_middle(fix_confusable_manzu(tiles))
            melds, _ = group_melds(tiles)
            previous = self.melds.get(seat, ())
            if len(melds) < len(previous):
                # 副露在一局中不會變少：這次讀少了（被動畫、按鈕擋住，或碰變成槓、疊起來那張讀不到），
                # 沿用之前的，只補上之前沒有的（10/22 19:34 上家加槓一筒後「碰一筒」就從紀錄消失）
                melds = list(previous) + [meld for meld in melds if meld not in previous]
            elif melds != list(previous) and previous:
                # 組數一樣、內容不同：可能是校正了之前讀錯的，也可能是之前那組這次被擋住、又多讀到一組新的
                # （20:35 對家「一二三萬」被下一次的「一條×3」換掉）。新的優先、再補上之前的，組數不超過牌河推算的
                limit = max(len(melds), (counts or {}).get(seat, len(melds)))
                melds = (melds + [meld for meld in previous if meld not in melds])[:limit]
            if melds:
                result[seat] = tuple(melds[:5])
            self._harvest(seat, hits, tiles, known.get(seat, []))
        self._harvest_guided(pending[2])
        changed = result != self.melds
        self.melds = result
        return changed

    def _harvest_guided(self, guided: list[tuple[str, list[MeldTile]]]) -> None:
        """引導搜尋找到的副露：切圖存成專用模板。哪三張是確定的（包含被吃的那張、三張連號），但吃的三張「誰是誰」
        靠三種排列的分數比，萬子數字倒著或側躺時很小、分數很接近，10/22 對家一二三萬就把一萬存成二萬、二萬存成三萬
        （錯的模板會讓對家越讀越錯）——吃的萬子不收；碰三張一樣、筒條的形狀差很多，照收。"""
        if self.template_dir is None:
            return
        for seat, hits in guided:
            chow = len({hit.tile for hit in hits}) > 1
            for number, hit in enumerate(hits):
                # 整組平均夠高、個別一張卻可能被立著的手牌或動畫蓋住（10/22 上家六條、七條存成半張藍色）
                if (chow and hit.tile < 9) or hit.score < HARVEST_MIN_SCORE:
                    continue
                folder = self.template_dir / seat / tile_code(hit.tile)
                existing = len(list(folder.glob("*.png"))) if folder.exists() else 0
                if hit.crop is None or existing >= MAX_TEMPLATES:
                    continue
                self.save(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_c{number}.png", hit.crop)
                self.saved_templates += 1

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
                if existing >= MAX_TEMPLATES:
                    break
                if hit.crop is None:  # fill_gaps 補上的那張沒有切圖
                    continue
                self.save(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{number}.png", hit.crop)
                existing += 1
                self.saved_templates += 1


def _grouped_hits(hits: list[MeldTile]) -> list[MeldTile]:
    """能成組（碰、吃、槓）的那些牌（依位置排序後分組）。"""
    ordered = sorted(hits, key=lambda hit: hit.position)
    spans = meld_spans([hit.tile for hit in ordered])
    return [hit for _, start, length in spans for hit in ordered[start:start + length]]


def _overlaps(found: list[MeldTile], hits: list[MeldTile], gap: int = 15) -> bool:
    """引導搜尋找到的位置和已經讀到的牌重疊（那裡已經是別的副露）。"""
    return any(abs(a.position - b.position) < gap for a in found for b in hits)


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
