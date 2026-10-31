"""有人槓牌時存截圖，收集對手槓的樣子（10/28 使用者：對手的槓都被當成碰）。

畫面上槓的第四張疊在中間那張上，副露辨識只讀到三張；要做第四張的辨識得先有實戰截圖。
每有人槓牌，左上角金牌就多一種，所以金牌種類數變多、而且持續 SHOT_DELAY 秒（金牌偶爾少讀一種會跳動）
就存一張（等槓的動畫結束）。自己的槓也會存，不另外排除。"""
from __future__ import annotations

SHOT_DELAY = 2.0
SHOT_LIMIT = 40
"""每次執行最多存幾張。"""


class KongShots:
    def __init__(self, delay: float = SHOT_DELAY, limit: int = SHOT_LIMIT) -> None:
        self.delay = delay
        self.left = limit
        self.base: int | None = None
        """這局目前認定的金牌種類數。"""
        self.pending: tuple[int, float] | None = None
        """(變多後的種類數, 第一次看到的時間)。"""
        self.started = False
        """這次 update 剛看到金牌變多（還沒確認）：當下輪到誰，就是誰槓的。"""
        self.confirmed = False
        """這次 update 確認有人槓（變多持續了 delay 秒；不受截圖張數限制）。"""
        self.started_at: float | None = None
        """最近一次確認的槓，金牌第一次看到變多的時間（和自己決定槓的時間比，確認要多等 delay 秒以上）。"""

    def new_hand(self) -> None:
        self.base = None
        self.pending = None

    def update(self, gold_count: int, now: float) -> bool:
        """每讀到一次手牌畫面呼叫；回傳 True 表示現在該存一張截圖。"""
        self.started = self.confirmed = False
        if self.base is None:
            self.base = gold_count
            return False
        if gold_count <= self.base:
            self.pending = None
            return False
        if self.pending is None or self.pending[0] != gold_count:
            self.started = self.pending is None
            self.pending = (gold_count, now)
            return False
        if now - self.pending[1] < self.delay:
            return False
        self.started_at = self.pending[1]
        self.base, self.pending = gold_count, None
        self.confirmed = True
        if self.left <= 0:
            return False
        self.left -= 1
        return True
