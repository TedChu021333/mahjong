"""自己的副露數合不合理：擋掉動畫中只讀到一部分的手牌。

自己的副露數由暗手張數反推（17 − 3n），手牌讀取只要求「從第 3n 格連續排到最右格」。
打完牌理牌、摸牌滑入時，左邊幾張常被動畫蓋住，只讀到右邊幾張也符合這條，就被當成有 3～5 組副露：
10/21 有 8 次，例如 12:02:52 只讀到「五條七條白白四條」就打七條並按聽、13:58:12 只讀到兩張。

副露只會在吃、碰、槓之後增加，一次加 1，那之前「吃／碰／槓」按鈕一定亮過，所以：
- 副露數不變或變少（新的一局）照常採用；
- 按鈕亮過 CLAIM_GRACE 秒內可以多 1；
- 其他增加不採用，同一個數字連續 PERSIST_SECONDS 秒才當真（漏看按鈕時不會一直卡住）。
"""
from __future__ import annotations

CLAIM_BUTTONS = frozenset({"chow", "pung", "kong"})
CLAIM_GRACE = 8.0
PERSIST_SECONDS = 3.0


class MeldCheck:
    def __init__(self) -> None:
        self.count = 0
        """目前採用的自己副露數。"""
        self._claim_seen_at = float("-inf")
        self._pending: tuple[int, float] | None = None
        """不合理的副露數與第一次看到的時間。"""

    def accept(self, count: int, buttons, now: float) -> bool:
        """這一幀讀到的副露數能不能採用。"""
        if buttons and buttons & CLAIM_BUTTONS:
            self._claim_seen_at = now
        if count <= self.count or (count == self.count + 1
                                   and now - self._claim_seen_at <= CLAIM_GRACE):
            self.count, self._pending = count, None
            return True
        if self._pending is None or self._pending[0] != count:
            self._pending = (count, now)
            return False
        if now - self._pending[1] >= PERSIST_SECONDS:
            self.count, self._pending = count, None
            return True
        return False
