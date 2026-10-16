"""大老二：從一幀幀的牌桌畫面整理出這局的出牌紀錄。

畫面上每家只顯示自己最近出的牌，有人出牌時別家的常被清掉，一墩結束（其他人都 PASS）整張桌面清空，
所以不能只看當下桌面：某家區域出現「連續兩幀一樣、和上次不同」的牌型就記一次出牌。
飛進來的動畫中途可能只看到幾張（例如五張只看到一張），所以要連續兩幀一致才採用；
同一家剛記的牌被後來看到的更多張包含時，改成大的那手。

自己的出牌以「手牌少了哪幾張」為準（`my_hand`）：別家的牌飛過桌面中間時曾被當成自己出的 4，
「大老二」動畫蓋住時也曾漏掉自己出的黑桃 2（錄影第 3 局）。桌面中間讀到的牌只在是自己手上的牌時採用。
別家的牌如果在自己手上或這局已經出過，就是認錯或動畫，不採用。
"""
from __future__ import annotations

from .combos import Combo, classify
from .game import PLAYERS, Play

ME = 0
HAND_STABLE = 3   # 自己手牌連續幾次一樣才算（大動畫整片蓋住幾張時，少讀的狀態約持續 1 秒）


class Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history: list[Play] = []
        self.shown: dict[int, Combo | None] = {s: None for s in range(PLAYERS)}
        self.pending: dict[int, Combo | None] = {s: None for s in range(PLAYERS)}
        self.trick_open = False
        self.hand: frozenset[int] | None = None       # 最近確認的自己手牌
        self.hand_candidate: frozenset[int] | None = None
        self.hand_streak = 0
        self.my_cards: set[int] = set()                # 這局曾確認在自己手上的牌

    def _record(self, seat: int, combo: Combo) -> Play:
        play = Play(seat, combo)
        self.history.append(play)
        self.shown[seat] = combo
        self.trick_open = True
        return play

    def my_hand(self, cards) -> list[Play]:
        """餵一次讀到的自己手牌（None 表示讀不到）。連續 HAND_STABLE 次一樣才算數；
        比「該有的手牌」少掉、而且少掉的剛好是一手合法的牌，就記成自己出牌。"""
        if cards is None:
            return []
        cards = frozenset(cards)
        if cards != self.hand_candidate:
            self.hand_candidate, self.hand_streak = cards, 1
            return []
        self.hand_streak += 1
        if self.hand_streak != HAND_STABLE:
            return []
        self.hand = cards
        expected = self.my_cards - self.played_cards()
        self.my_cards |= cards
        gone = expected - cards
        if not gone or not cards < expected:
            return []
        combo = classify(gone)
        if combo is not None:
            return [self._record(ME, combo)]
        last = self.history[-1] if self.history else None
        if last is not None and last.seat == ME and last.combo is not None:
            whole = classify(gone | set(last.combo.cards))   # 桌面只記到飛進來的前幾張
            if whole is not None:
                self.history[-1] = Play(ME, whole)
                self.shown[ME] = whole
        return []

    def update(self, plays: dict[int, Combo | None | str]) -> list[Play]:
        """餵一幀 read_plays 的結果，回傳這幀新確認的出牌。"""
        new = []
        for seat, combo in plays.items():
            if combo == "?":
                continue
            if combo is None:
                self.pending[seat] = None
                self.shown[seat] = None
                continue
            if combo != self.pending[seat]:
                self.pending[seat] = combo
                continue
            if combo == self.shown[seat]:
                continue
            cards = set(combo.cards)
            last = self.history[-1] if self.history else None
            grows = (last is not None and last.seat == seat and last.combo is not None
                     and set(last.combo.cards) < cards)
            played = self.played_cards() - (set(last.combo.cards) if grows else set())
            if seat == ME:
                if not cards <= self.my_cards or cards & played:
                    continue
            elif cards & self.my_cards or cards & played:
                continue
            if grows:
                self.history[-1] = Play(seat, combo)      # 動畫中途只看到一部分
                self.shown[seat] = combo
            else:
                new.append(self._record(seat, combo))
        if all(plays.get(s) is None for s in range(PLAYERS)):
            self.trick_open = False                         # 整張桌面清空：一墩結束
        return new

    @property
    def settled(self) -> bool:
        """沒有「出現一幀、還沒確認」的牌：別家剛出牌時先不要下決定。"""
        return all(self.pending[s] is None or self.pending[s] == self.shown[s] for s in range(PLAYERS))

    @property
    def last_play(self) -> Play | None:
        return next((p for p in reversed(self.history) if p.combo is not None), None)

    def to_beat(self) -> tuple[Combo | None, int | None]:
        """輪到自己時要壓的牌與出那手的人；自己帶頭時 (None, None)。"""
        last = self.last_play
        if last is None or not self.trick_open or last.seat == ME:
            return None, None
        return last.combo, last.seat

    def played_by(self, seat: int) -> int:
        return sum(len(p.combo.cards) for p in self.history if p.seat == seat and p.combo)

    def played_cards(self) -> set[int]:
        return {c for p in self.history if p.combo for c in p.combo.cards}
