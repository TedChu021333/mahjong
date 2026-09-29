"""蒙地卡羅打牌（determinization）：補完看不到的牌，把每個候選打法模擬到這局結束，取平均輸贏。

1. 補完：從看不到的牌抽出對手暗手牌與牌牆（agent.opponent_model 的抽樣：宣告聽牌者組成
   聽牌手牌、且不聽自己打過的牌；未宣告者依聽牌機率決定要不要組成聽牌手牌）。
2. 候選：期望值 AI 分數最高的幾張；會聽牌的牌另外把「宣告聽牌」當成一個候選（胡了多 1 台，
   但之後手牌鎖住、摸到什麼都得打）。
3. 每次補完後，所有候選都用同一副牌模擬（agent.rollout），比較平均輸贏；共用同一批隨機
   牌局能大幅降低比較時的運氣雜訊。
4. 預設照期望值 AI 的第一名打（能聽牌就宣告，與規則式 AI 相同），只有模擬顯示別的選項
   顯著較好（配對差距超過 SIGNIFICANCE 倍標準誤差）才改。模擬次數有限時差距多半是雜訊，
   直接取平均最高會在向聽相同、進張較少的牌之間亂換（arena 前 80 對胡牌率掉 11%）。
胡牌、吃碰槓、換三張沿用規則式 AI。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from random import Random

from agent.ev_agent import RISK_WEIGHT, discard_scores
from agent.opponent_model import (
    UNDECLARED_TENPAI_SHARE,
    Calibration,
    default_calibration,
    sample_tenpai_hand,
    unseen_counts,
)
from agent.rollout import Seat, play_out
from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType, GameState, Phase, legal_actions

CANDIDATES = 3
ROLLOUTS = 24
"""每個候選模擬幾次（實戰另有時間上限）。"""
TIME_LIMIT = 2.0
"""實戰每步最多用幾秒（遊戲限時 3 秒）。"""
LIVE_ROLLOUTS = 400
"""實戰的模擬次數上限（通常先碰到時間上限）。"""
SIGNIFICANCE = 2.0


def determinize(state: GameState, player: int, rng: Random, calibration: Calibration,
                undeclared_share: float) -> tuple[list[Seat], list[int]]:
    pool = unseen_counts(state, player)
    seats: list[Seat | None] = [None] * 4
    me = state.players[player]
    seats[player] = Seat(list(me.hand), me.open_melds, state.declared[player])
    order = sorted((opponent for opponent in range(4) if opponent != player),
                   key=lambda opponent: not state.declared[opponent])  # 有限制的先抽
    for opponent in order:
        other = state.players[opponent]
        size = max(1, 16 - 3 * len(other.melds))
        declared = state.declared[opponent]
        tenpai = declared or rng.random() < calibration.undeclared_tenpai_probability(
            len(other.discards), len(other.melds), undeclared_share)
        hand = None
        if tenpai:
            sampled = sample_tenpai_hand(pool, len(other.melds), rng,
                                         set(other.discards) if declared else set())
            if sampled is not None:
                hand = sampled[0]
        if hand is None:
            hand = _random_hand(pool, size, rng)
        for tile, count in enumerate(hand):
            pool[tile] -= count
        seats[opponent] = Seat(hand, len(other.melds), declared)
    wall = [tile for tile, count in enumerate(pool) for _ in range(count)]
    rng.shuffle(wall)
    return seats, wall[:max(0, state.drawable)]


def _random_hand(pool: list[int], size: int, rng: Random) -> list[int]:
    tiles = [tile for tile, count in enumerate(pool) for _ in range(count)]
    hand = [0] * len(pool)
    for tile in rng.sample(tiles, min(size, len(tiles))):
        hand[tile] += 1
    return hand


@dataclass(frozen=True)
class Option:
    tile: int
    declare: bool


def _average_tai(calibration: Calibration, self_draw: bool) -> float:
    """沒宣告聽牌時的平均台數；模擬器裡沒宣告就胡的很少，改用宣告者的平均少 1 台。"""
    average, count = calibration.tai.get((False, self_draw), (0.0, 0))
    if count >= 30:
        return average
    declared, count = calibration.tai.get((True, self_draw), (0.0, 0))
    return declared - 1 if count else 4.0


def evaluate_options(state: GameState, player: int, options: list[Option], rng: Random,
                     calibration: Calibration, rollouts: int, undeclared_share: float,
                     time_limit: float | None = None) -> dict[Option, list[float]]:
    """每個選項在每次補完的牌局中的輸贏（同一個索引是同一副牌）。"""
    tsumo = _average_tai(calibration, self_draw=True)
    ron = _average_tai(calibration, self_draw=False)
    results: dict[Option, list[float]] = {option: [] for option in options}
    started = time.perf_counter()
    done = 0
    for _ in range(rollouts):
        if time_limit is not None and done and time.perf_counter() - started > time_limit:
            break
        seats, wall = determinize(state, player, rng, calibration, undeclared_share)
        for option in options:
            copies = [Seat(list(seat.hand), seat.open_melds, seat.declared) for seat in seats]
            mine = copies[player]
            mine.hand[option.tile] -= 1
            mine.declared = mine.declared or option.declare
            results[option].append(play_out(copies, wall, player, option.tile, tsumo, ron)[player])
        done += 1
    return results


def significantly_better(challenger: list[float], default: list[float],
                         significance: float = SIGNIFICANCE) -> bool:
    diffs = [a - b for a, b in zip(challenger, default)]
    n = len(diffs)
    if n < 2:
        return False
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean > significance * (variance / n) ** 0.5 and mean > 0


def choose_mc_action_for_player(state: GameState, player: int, rng: Random | None = None,
                                calibration: Calibration | None = None,
                                rollouts: int = ROLLOUTS, candidates: int = CANDIDATES,
                                undeclared_share: float = UNDECLARED_TENPAI_SHARE,
                                time_limit: float | None = None) -> Action | None:
    rng = rng or Random()
    if state.phase != Phase.DISCARD or state.declared[player] or player != state.current_player:
        return choose_rule_action_for_player(state, player, rng)
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return win
    calibration = calibration or default_calibration()
    scores = discard_scores(state, player, rng, calibration, risk_weight=RISK_WEIGHT,
                            undeclared_share=undeclared_share)
    ranked = sorted(scores, key=scores.get, reverse=True)[:candidates]
    options = [Option(tile, False) for tile in ranked]
    options += [Option(tile, True) for tile in ranked
                if Action(ActionType.DECLARE, tile=tile) in actions]
    top = ranked[0]
    default = Option(top, Option(top, True) in options)
    best = default
    if len(options) > 1:
        results = evaluate_options(state, player, options, rng, calibration, rollouts,
                                   undeclared_share, time_limit)
        average = {option: sum(values) / max(1, len(values)) for option, values in results.items()}
        challengers = [option for option in options if option != default
                       and significantly_better(results[option], results[default])]
        if challengers:
            best = max(challengers, key=average.get)
    kind = ActionType.DECLARE if best.declare else ActionType.DISCARD
    return Action(kind, tile=best.tile)
