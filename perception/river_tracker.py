"""把逐幀的牌河讀數轉成出牌事件：某家打了某張、某家最新的一張被吃碰。

以畫面位置追蹤每張牌，不需要知道牌河的格位怎麼排（上家、下家是一欄一欄排的，
新的一欄從最上面開始）：
- 新位置連續 CONFIRM_FRAMES 幀都有牌 → 那家打出一張；確認的先後就是出牌順序。
  牌滑進牌河的動畫、放大顯示等短暫物件撐不過這幾幀。
- 該家最新的一張連續 CONFIRM_FRAMES 幀不見、其他牌都在 → 被吃碰拿走；之後若很快又在
  原位出現同一張（其實是被蓋住），就放回去，不算新的出牌。
- 其他牌不見都當成被蓋住（吃碰按鈕列會蓋住自己的牌河好幾秒），不移除；新的一局由
  提示迴圈在結算畫面呼叫 reset()。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from perception.hand_reader import Box
from perception.river_reader import SEATS, RiverTile

CONFIRM_FRAMES = 3
RESTORE_FRAMES = 15
"""被判定吃碰的牌在這麼多幀內又在原位出現才算被蓋住；那一格之後會放同一家下一次
打的牌，但要輪完一圈（通常超過這個時間）。兩張都認得出來時則直接比對牌種。"""
SAME_POSITION = 20
"""兩次偵測的左上角相差在這個距離內視為同一格。"""


@dataclass(frozen=True)
class RiverEvent:
    kind: str
    """"discard"（打出）、"claimed"（被吃碰拿走）或 "restored"（原來只是被蓋住）。"""
    seat: str
    tile: int | None
    box: Box


@dataclass
class _Entry:
    box: Box
    tile: int | None
    score: float
    seen: int = 0
    """連續出現的幀數（確認前用）。"""
    missing: int = 0
    """連續不見的幀數。"""

    def near(self, box: Box) -> bool:
        return abs(self.box[0] - box[0]) <= SAME_POSITION and \
            abs(self.box[1] - box[1]) <= SAME_POSITION

    def absorb(self, tile: RiverTile) -> None:
        self.box = tile.box
        if tile.tile is not None and (self.tile is None or tile.score > self.score):
            self.tile, self.score = tile.tile, tile.score


@dataclass
class RiverTracker:
    rivers: dict[str, list[_Entry]] = field(
        default_factory=lambda: {seat: [] for seat in SEATS})
    """目前在牌河上的牌，依出牌先後。"""
    history: dict[str, list[_Entry]] = field(
        default_factory=lambda: {seat: [] for seat in SEATS})
    """各家打過的牌（含之後被吃碰拿走的），與 rivers 共用物件，之後認出的牌會一起更新。"""
    _candidates: dict[str, list[_Entry]] = field(
        default_factory=lambda: {seat: [] for seat in SEATS})
    _claimed: dict[str, list[tuple[_Entry, int]]] = field(
        default_factory=lambda: {seat: [] for seat in SEATS})
    """判定被吃碰拿走、但也可能只是被蓋住的牌，以及判定當時的幀數。"""
    _frame: int = 0

    def reset(self) -> None:
        for seat in SEATS:
            self.rivers[seat].clear()
            self.history[seat].clear()
            self._candidates[seat].clear()
            self._claimed[seat].clear()

    def tiles(self, seat: str) -> list[int | None]:
        return [entry.tile for entry in self.rivers[seat]]

    @property
    def discards(self) -> dict[str, list[int | None]]:
        """各家打過的牌（含被吃碰拿走的），判斷現物用。"""
        return {seat: [entry.tile for entry in self.history[seat]] for seat in SEATS}

    def update(self, reading: dict[str, list[RiverTile]]) -> list[RiverEvent]:
        self._frame += 1
        events = []
        for seat in SEATS:
            events += self._update_seat(seat, reading[seat])
        return events

    def _update_seat(self, seat: str, current: list[RiverTile]) -> list[RiverEvent]:
        events = []
        river, candidates = self.rivers[seat], self._candidates[seat]
        unmatched = list(current)
        for entry in river:
            match = next((tile for tile in unmatched if entry.near(tile.box)), None)
            if match is None:
                entry.missing += 1
            else:
                unmatched.remove(match)
                entry.absorb(match)
                entry.missing = 0
        self._claimed[seat] = [(entry, frame) for entry, frame in self._claimed[seat]
                               if self._frame - frame <= RESTORE_FRAMES]
        for tile in list(unmatched):
            restored = next((item for item in self._claimed[seat] if item[0].near(tile.box)
                             and (item[0].tile is None or tile.tile is None
                                  or item[0].tile == tile.tile)), None)
            if restored is not None:
                unmatched.remove(tile)
                self._claimed[seat].remove(restored)
                restored = restored[0]
                restored.missing = 0
                restored.absorb(tile)
                river.append(restored)
                events.append(RiverEvent("restored", seat, restored.tile, restored.box))
        next_candidates = []
        for tile in unmatched:
            previous = next((c for c in candidates if c.near(tile.box)), None)
            entry = previous or _Entry(tile.box, tile.tile, tile.score)
            if previous is not None:
                entry.absorb(tile)
            entry.seen += 1
            if entry.seen >= CONFIRM_FRAMES:
                entry.seen = 0
                river.append(entry)
                self.history[seat].append(entry)
                self._claimed[seat].clear()  # 有新的出牌後，之前被拿走的牌不會再回來
                events.append(RiverEvent("discard", seat, entry.tile, entry.box))
            else:
                next_candidates.append(entry)
        self._candidates[seat] = next_candidates
        if river and river[-1].missing >= CONFIRM_FRAMES and \
                sum(entry.missing > 0 for entry in river) == 1:
            entry = river.pop()
            self._claimed[seat].append((entry, self._frame))
            events.append(RiverEvent("claimed", seat, entry.tile, entry.box))
        return events
