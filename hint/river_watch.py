"""即時模式的牌河追蹤：記錄各家打過的牌，順便自動收集牌河模板。

牌河模板一開始很少，辨識不出的牌記為未知。每次出牌都有可靠的標註來源：
- 自己打的牌：比對打牌前後的暗手牌就知道。
- 別家打的牌：能吃碰時會放大顯示（辨識很準），就是那家牌河接著出現的那張。
辨識器認不出或認錯時，把這張牌存成新模板（train/river_templates/<牌>/auto_*.png），
並以標註修正記錄。

副露數：牌打出後會先落進牌河，有人吃碰才從牌河被拿走（追蹤器的 "claimed" 事件）。
接著第一個出牌的人（牌河多一張，或出現新的放大圓框）就是吃碰的人（吃碰完要馬上打一張），
副露數加 1；被拿走的牌之後又回到原位（其實是被蓋住）就取消。下一個出牌的是自己、而自己
並沒有吃碰時，吃碰的一定是上家（他吃碰完打一張就輪到我們，只是那張被遮住還沒看到）。
吃只能吃上家的牌：拿走牌的人不是打牌者的下家，就一定是碰（或槓），那種牌另外兩張也看得到了。
別家吃碰時吃碰者那邊會出現「吃」／「碰」大字（read_claim_banner）：剛出現的大字之後幾秒內有牌被拿走，
直接照大字算給那家（碰的話另外兩張也看得到），不必等下一個出牌的人；大字比牌河記的「被拿牌的那家」可靠。
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from game.tiles import tile_code
from hint.advisor import RIVER_SEAT, Observation
from perception.capture import write_image
from perception.config import RIVER_TEMPLATE_DIR
from perception.river_reader import SEATS, RiverReader, crop_face
from perception.river_tracker import RiverEvent, RiverTracker
from perception.table_reader import read_claim_banner

NEXT_SEAT = {"me": "right", "right": "top", "top": "left", "left": "me"}
"""出牌順序：自己 → 下家 → 對家 → 上家。"""
BANNER_FRAMES = 10
"""大字出現後這麼多次更新（約 3 秒）內的「被拿走」才算這次吃碰。"""
LABEL_FRAMES = 15
"""標註在這麼多幀內等不到那家牌河多一張就作廢。"""


def _read_banner(frame):
    return read_claim_banner(frame) if frame is not None else None


class RiverWatch:
    def __init__(self, reader: RiverReader, template_dir: Path = RIVER_TEMPLATE_DIR,
                 save=write_image, read_banner=None) -> None:
        self.reader = reader
        self.read_banner = read_banner or _read_banner
        self._banner: tuple[str, str, int] | None = None
        """剛出現、還沒對到被拿走的牌的大字：(吃碰者, chow/pung, 出現時的更新次數)。"""
        self._banner_visible = False
        self.tracker = RiverTracker()
        self.template_dir = template_dir
        self.save = save
        self.saved = 0
        self._frame = 0
        self._labels: dict[str, tuple[int, int]] = {}
        """座位 → (牌, 取得標註時的幀數)。"""
        self._turn_hand: tuple[int, ...] | None = None
        self.melds = {seat: 0 for seat in SEATS}
        """推算的各家副露數（自己的由手牌張數得知，這裡不算）。"""
        self.exposed: dict[str, list[int]] = {seat: [] for seat in SEATS}
        """各家副露裡已知、不在牌河的牌：推算出的碰（另外兩張）、自己吃碰用掉的手牌。"""
        self._taken_from: tuple[str, int, int | None] | None = None
        """最新一張被拿走的牌是誰打的、當時自己的副露數、那張牌，等著看誰接著出牌。"""
        self._my_melds = 0
        self._credited: tuple[str, str, tuple[int, ...]] | None = None
        """最近一次推算：(被拿走牌的那家, 算給誰, 加進 exposed 的牌)；那張牌之後又回來就撤銷。"""
        self._last_circle: tuple[str, int] | None = None

    def new_hand(self) -> None:
        self.tracker.reset()
        self._labels.clear()
        self._turn_hand = None
        self.melds = {seat: 0 for seat in SEATS}
        self.exposed = {seat: [] for seat in SEATS}
        self._taken_from = None
        self._last_circle = None
        self._credited = None
        self._banner = None

    def update(self, frame: np.ndarray, observation: Observation) -> list[RiverEvent]:
        self._frame += 1
        self._my_melds = observation.meld_count
        self._collect_labels(observation)
        banner = self.read_banner(frame)
        if banner is not None and not self._banner_visible:
            self._banner = (*banner, self._frame)  # 只記剛出現的那一次
        self._banner_visible = banner is not None
        events = self.tracker.update(self.reader.read(frame))
        for event in events:
            if event.kind == "discard":
                self._apply_label(frame, event)
                self._someone_discarded(event.seat)
            elif event.kind == "claimed":
                if self._banner and self._frame - self._banner[2] <= BANNER_FRAMES \
                        and self._banner[0] != event.seat:
                    claimer, kind, _ = self._banner
                    self._banner = self._taken_from = None
                    tile = event.tile
                    self._credit(event.seat, claimer,
                                 (tile, tile) if kind == "pung" and tile is not None else ())
                    continue
                self._taken_from = (event.seat, self._my_melds, event.tile)
            elif event.kind == "restored":
                if self._taken_from and self._taken_from[0] == event.seat:
                    self._taken_from = None
                if self._credited and self._credited[0] == event.seat:
                    _, claimer, tiles = self._credited
                    self.melds[claimer] -= 1  # 其實只是被蓋住
                    for tile in tiles:
                        self.exposed[claimer].remove(tile)
                    self._credited = None
        return events

    def _someone_discarded(self, seat: str) -> None:
        taken, self._taken_from = self._taken_from, None
        if taken is None or seat == taken[0]:
            return
        claimer = seat
        if seat == "me":
            if self._my_melds != taken[1] or taken[0] == "left":
                return  # 自己吃碰的
            claimer = "left"  # 自己沒吃碰卻輪到自己：是上家吃碰
        tile = taken[2]
        tiles = (tile, tile) if tile is not None and claimer != NEXT_SEAT[taken[0]] else ()
        self._credit(taken[0], claimer, tiles)

    def _credit(self, source: str, claimer: str, tiles: tuple[int, ...]) -> None:
        self.melds[claimer] += 1
        self.exposed[claimer] += tiles
        self._credited = (source, claimer, tiles)

    def own_claim(self, tiles) -> None:
        """自己吃碰完成：用掉的手牌（被吃碰的那張已在別家牌河）。"""
        self.exposed["me"] += list(tiles)

    def exposed_tiles(self) -> tuple[tuple[int, ...], ...]:
        """依 advisor 的座位編號排列。"""
        result = [()] * 4
        for seat, tiles in self.exposed.items():
            result[RIVER_SEAT[seat]] = tuple(tiles)
        return tuple(result)

    def meld_counts(self) -> tuple[int, int, int, int]:
        """依 advisor 的座位編號排列；自己的欄位為 0。"""
        result = [0] * 4
        for seat, count in self.melds.items():
            if seat != "me":
                result[RIVER_SEAT[seat]] = min(count, 5)
        return tuple(result)

    def discards(self) -> tuple[tuple[int, ...], ...]:
        """依 advisor 的座位編號排列，未知的牌略過。"""
        result = [()] * 4
        for seat, tiles in self.tracker.discards.items():
            result[RIVER_SEAT[seat]] = tuple(tile for tile in tiles if tile is not None)
        return tuple(result)

    def _collect_labels(self, observation: Observation) -> None:
        claim = observation.claim
        if claim is not None and claim.tile is not None:
            self._labels[claim.source] = (claim.tile, self._frame)
            key = (claim.source, claim.tile)
            if key != self._last_circle:  # 新的放大圓框：這家剛打出一張
                self._last_circle = key
                self._someone_discarded(claim.source)
        if observation.my_turn:
            self._turn_hand = observation.hand
        elif self._turn_hand is not None and len(observation.hand) == len(self._turn_hand) - 1:
            gone = Counter(self._turn_hand) - Counter(observation.hand)
            if sum(gone.values()) == 1:
                self._labels["me"] = (next(iter(gone)), self._frame)
            self._turn_hand = None

    def _apply_label(self, frame: np.ndarray, event: RiverEvent) -> None:
        label = self._labels.pop(event.seat, None)
        if label is None or self._frame - label[1] > LABEL_FRAMES:
            return
        tile = label[0]
        entry = self.tracker.rivers[event.seat][-1]
        if entry.tile == tile:
            return  # 已經認得，不必再存
        path = self.template_dir / tile_code(tile) / \
            f"auto_{time.strftime('%Y%m%d_%H%M%S')}_{self._frame}_{event.seat}.png"
        self.save(path, crop_face(frame, event.box))
        self.saved += 1
        entry.tile, entry.score = tile, 1.0


def summary(watch: RiverWatch) -> str:
    from game.tiles import tile_name

    parts = []
    for seat in SEATS:
        tiles = watch.tracker.discards[seat]
        text = f"{seat} " + "".join(tile_name(t) if t is not None else "?" for t in tiles)
        if watch.melds[seat]:
            text += f"（副露 {watch.melds[seat]}）"
        parts.append(text)
    return "；".join(parts)
