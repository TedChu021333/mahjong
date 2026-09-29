"""對手模型：估計打出每張牌的放槍機率與損失，以及自己進攻的期望收益。

機率表 agent/calibration.json 由 `python -m eval.calibrate` 從模擬器對局產生。

放槍機率：對每個可能聽牌的對手，從看不到的牌隨機組出「聽牌的手牌」（面子 + 對子 +
搭子，或面子 + 單騎），看打出的牌落在他聽的牌裡的比例，再乘上他聽牌的機率：
- 宣告聽牌的對手：一定聽牌；他打過的牌不可能是他要胡的（宣告後手牌鎖住，而且他
  還沒胡），組出的手牌若聽到這些牌就重抽。
- 沒宣告的對手：聽牌機率依他打過幾張、幾組副露查表（表是所有人的聽牌率），再依「聽牌但
  不宣告的比例」q 換算成「沒宣告的人已聽牌」的機率 p·q / (1 − p + p·q)。模擬器裡的 AI 一律
  宣告（q = 0）；實戰 38 張結算畫面中約四成胡牌者沒有「聽牌」台（q ≈ 0.4）。
只用公開資訊：自己的手牌、各家牌河、已知的副露內容、誰宣告了聽牌。
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Sequence

from game.rules import GameState
from game.tiles import NUM_TILE_TYPES
from game.win import winning_tiles

CALIBRATION = Path(__file__).with_name("calibration.json")
BASE = 3
"""底（台）；與 eval.arena 相同。"""
SAMPLE_ATTEMPTS = 4
"""組一副聽牌手牌失敗（牌不夠、違反限制）時最多重試幾次。"""
UNDECLARED_TENPAI_SHARE = 0.4
"""實戰中聽牌卻不宣告的比例（模擬器評估時用 0）。"""
SEQUENCE_SHARE = 0.7
"""組面子時順子的比例（其餘為刻子）。"""
TANKI_SHARE = 0.2
"""聽牌型態中單騎的比例（其餘為對子 + 搭子）。"""


@dataclass(frozen=True, eq=False)
class Calibration:
    """eq=False：以物件本身雜湊，才能當 _monotone_offense 快取的鍵。"""
    tenpai: dict[tuple[int, int], tuple[int, int]]
    offense: dict[tuple[int, int, int], tuple[int, int, int]]
    tai: dict[tuple[bool, bool], tuple[float, int]]
    discard_buckets: int
    wall_bucket: int
    outs_buckets: tuple[int, ...]

    @classmethod
    def load(cls, path: Path = CALIBRATION) -> "Calibration":
        data = json.loads(path.read_text(encoding="utf-8"))

        def keys(text):
            return tuple(int(part) for part in text.split(","))

        return cls(
            tenpai={keys(k): tuple(v) for k, v in data["tenpai"].items()},
            offense={keys(k): tuple(v) for k, v in data["offense"].items()},
            tai={(bool(keys(k)[0]), bool(keys(k)[1])): tuple(v) for k, v in data["tai"].items()},
            discard_buckets=data["discard_buckets"],
            wall_bucket=data["wall_bucket"],
            outs_buckets=tuple(data["outs_buckets"]),
        )

    def tenpai_probability(self, discards: int, melds: int) -> float:
        """打過這麼多張、這麼多副露的人已經聽牌的機率（拉普拉斯平滑；樣本太少時先往副露少、
        再往打過張數少的組別借）。"""
        for d in range(min(discards, self.discard_buckets - 1), -1, -1):
            for m in range(min(melds, 5), -1, -1):
                hit, total = self.tenpai.get((d, m), (0, 0))
                if total >= 30:
                    return (hit + 1) / (total + 2)
        return 0.0

    def undeclared_tenpai_probability(self, discards: int, melds: int,
                                      undeclared_share: float) -> float:
        """沒宣告的人已經聽牌的機率。"""
        p = self.tenpai_probability(discards, melds)
        silent = p * undeclared_share
        return silent / (1 - p + silent) if silent else 0.0

    def offense_value(self, shanten: int, outs: int, drawable: int) -> float:
        """打完後處在這個狀態，這局靠自己胡牌平均贏幾台（沒胡算 0）。

        表來自分組統計，有雜訊（例如聽牌時進張少的組別反而較高），這裡強制單調：
        進張越多不會變低、向聽數越低不會變低。"""
        outs_group = max(i for i, low in enumerate(self.outs_buckets) if outs >= low)
        wall_group = drawable // self.wall_bucket
        return _monotone_offense(self, max(-1, min(shanten, 8)), outs_group, wall_group)

    def raw_offense_value(self, shanten: int, outs_group: int, wall_group: int) -> float:
        wins = total = gain = 0
        # 樣本太少時合併相鄰的牌牆、進張組別
        for spread in range(0, 4):
            for dw in range(-spread, spread + 1):
                for do in range(-spread, spread + 1):
                    if max(abs(dw), abs(do)) != spread:
                        continue
                    w, t, g = self.offense.get((shanten, outs_group + do, wall_group + dw),
                                               (0, 0, 0))
                    wins, total, gain = wins + w, total + t, gain + g
            if total >= 50:
                break
        return gain / total if total else 0.0

    def loss_if_dealing_in(self, declared: bool) -> float:
        """放槍給一家要付多少（底 + 那家放槍胡牌的平均台數）。"""
        average, count = self.tai.get((declared, False), (0.0, 0))
        if count < 30:  # 模擬器裡沒宣告就胡的很少：用宣告者的平均少 1 台（聽牌台）
            declared_average, _ = self.tai.get((True, False), (5.0, 0))
            average = declared_average - 1 if not declared else declared_average
        return BASE + average


@lru_cache(maxsize=None)
def _monotone_offense(calibration: Calibration, shanten: int, outs_group: int,
                      wall_group: int) -> float:
    value = calibration.raw_offense_value(shanten, outs_group, wall_group)
    if outs_group > 0:
        value = max(value, _monotone_offense(calibration, shanten, outs_group - 1, wall_group))
    if shanten < 8:
        value = max(value, _monotone_offense(calibration, shanten + 1, outs_group, wall_group))
    return value


@lru_cache(maxsize=1)
def default_calibration() -> Calibration:
    return Calibration.load()


def unseen_counts(state: GameState, player: int) -> list[int]:
    """player 看不到的每種牌還有幾張（4 減去自己手牌、各家牌河、已知副露）。"""
    seen = list(state.players[player].hand)
    for other in state.players:
        for tile in other.discards:
            seen[tile] += 1
        for meld in other.melds:
            for tile in meld.tiles:
                seen[tile] += 1
    return [max(0, 4 - count) for count in seen]


def _take(pool: list[int], tiles: Sequence[int]) -> bool:
    need = Counter(tiles)
    if any(pool[tile] < count for tile, count in need.items()):
        return False
    for tile, count in need.items():
        pool[tile] -= count
    return True


def _random_meld(rng: Random) -> tuple[int, ...]:
    if rng.random() < SEQUENCE_SHARE:
        suit = rng.randrange(3)
        start = suit * 9 + rng.randrange(7)
        return (start, start + 1, start + 2)
    tile = rng.randrange(NUM_TILE_TYPES)
    return (tile, tile, tile)


def _random_partial(rng: Random) -> tuple[int, ...]:
    """聽牌前的搭子：兩面、坎張、邊張（順子缺一張）或對子（對碰）。"""
    if rng.random() < 0.25:
        tile = rng.randrange(NUM_TILE_TYPES)
        return (tile, tile)
    suit = rng.randrange(3)
    first = suit * 9 + rng.randrange(8)
    if rng.random() < 0.3 and first % 9 < 7:
        return (first, first + 2)
    return (first, first + 1)


def sample_tenpai_hand(pool: Sequence[int], open_melds: int, rng: Random,
                       forbidden: set[int] = frozenset()) -> tuple[list[int], list[int]] | None:
    """從 pool 組出一副聽牌的暗手牌，回傳 (手牌計數, 聽的牌)；聽到 forbidden 的牌、或組不出來時
    回傳 None。不改動 pool。"""
    melds_needed = 4 - open_melds
    for _ in range(SAMPLE_ATTEMPTS):
        left = list(pool)
        counts = [0] * NUM_TILE_TYPES
        # 五組副露時手上只剩一張：只能單騎
        tanki = melds_needed < 0 or rng.random() < TANKI_SHARE
        groups = [_random_meld(rng) for _ in range(melds_needed + (1 if tanki else 0))]
        if tanki:
            groups.append((rng.randrange(NUM_TILE_TYPES),))
        else:
            pair = rng.randrange(NUM_TILE_TYPES)
            groups += [(pair, pair), _random_partial(rng)]
        if not all(_take(left, group) for group in groups):
            continue
        for group in groups:
            for tile in group:
                counts[tile] += 1
        waits = winning_tiles(counts, open_melds)
        if waits and not forbidden.intersection(waits):
            return counts, waits
    return None


def sample_waits(pool: Sequence[int], open_melds: int, rng: Random,
                 forbidden: set[int] = frozenset()) -> list[int] | None:
    """組出一副聽牌手牌，只回傳聽的牌。"""
    sampled = sample_tenpai_hand(pool, open_melds, rng, forbidden)
    return None if sampled is None else sampled[1]


@dataclass(frozen=True)
class Danger:
    probability: dict[int, float]
    """打出這張牌放槍的機率。"""
    expected_loss: dict[int, float]
    """打出這張牌因放槍預期要付的台數。"""


def estimate_danger(state: GameState, player: int, candidates: Sequence[int],
                    rng: Random, calibration: Calibration | None = None,
                    samples: int = 120,
                    undeclared_share: float = UNDECLARED_TENPAI_SHARE) -> Danger:
    calibration = calibration or default_calibration()
    pool = unseen_counts(state, player)
    safe_probability = {tile: 1.0 for tile in candidates}
    loss = {tile: 0.0 for tile in candidates}
    for opponent, other in enumerate(state.players):
        if opponent == player:
            continue
        declared = state.declared[opponent]
        p_tenpai = 1.0 if declared else calibration.undeclared_tenpai_probability(
            len(other.discards), len(other.melds), undeclared_share)
        if p_tenpai < 0.02:
            continue
        forbidden = set(other.discards) if declared else set()
        hits: Counter[int] = Counter()
        valid = 0
        for _ in range(samples):
            waits = sample_waits(pool, len(other.melds), rng, forbidden)
            if waits is None:
                continue
            valid += 1
            hits.update(waits)
        if not valid:
            continue
        per_loss = calibration.loss_if_dealing_in(declared)
        for tile in candidates:
            risk = p_tenpai * hits[tile] / valid
            safe_probability[tile] *= 1 - risk
            loss[tile] += risk * per_loss
    return Danger({tile: 1 - p for tile, p in safe_probability.items()}, loss)
