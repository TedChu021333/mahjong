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
from multiprocessing import TimeoutError as MultiprocessingTimeout
from random import Random

from agent.ev_agent import RISK_WEIGHT, discard_scores
from agent.opponent_model import (
    UNDECLARED_TENPAI_SHARE,
    Calibration,
    default_calibration,
    sample_tenpai_hand,
    unseen_counts,
)
from agent.rollout import BASE, Seat, dealer_bonus, heuristic_discard, play_out
from agent.rule_agent import (
    choose_discard,
    choose_rule_action_for_player,
    choose_self_kong,
    declare_is_worth,
)
from agent.shanten import shanten
from game.simulator import after_rake
from game.rules import (
    Action,
    ActionType,
    GameState,
    Phase,
    forbidden_after_claim,
    legal_actions,
)
from game.win import is_win

CANDIDATES = 3
ROLLOUTS = 24
"""每個候選模擬幾次（實戰另有時間上限）。"""
TIME_LIMIT = 1.5
"""實戰每個程序最多模擬幾秒（遊戲限時 3 秒，還要留給辨識、點擊與程序間傳遞）。"""
LIVE_ROLLOUTS = 400
"""實戰的模擬次數上限（通常先碰到時間上限）。"""
SIGNIFICANCE = 2.0
MC_CLAIMS = False
"""吃碰槓是否也用模擬判斷。arena 1724 對（只有吃碰用模擬 vs 規則式）：−0.183±0.321，
胡牌率 25.7%→24.8%、放槍率 19.3%→19.8%，沒有進步，預設關閉、照規則式。"""
PARALLEL_GRACE = 0.5
"""子程序超過 time_limit 這麼多秒還沒回來就放棄模擬，改照期望值第一名打（不能卡住牌局）。"""


def determinize(state: GameState, player: int, rng: Random, calibration: Calibration,
                undeclared_share: float) -> tuple[list[Seat], list[int]]:
    pool = unseen_counts(state, player)
    seats: list[Seat | None] = [None] * 4
    me = state.players[player]
    seats[player] = Seat(list(me.hand), me.open_melds, state.declared[player], set(me.discards))
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
        seats[opponent] = Seat(hand, len(other.melds), declared, set(other.discards))
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
            copies = [seat.copy() for seat in seats]
            mine = copies[player]
            mine.hand[option.tile] -= 1
            mine.declared = mine.declared or option.declare
            value = play_out(copies, wall, player, option.tile, tsumo, ron,
                             state.known_dealer, state.dealer_streak)[player]
            results[option].append(after_rake(value))
        done += 1
    return results


def _evaluate_chunk(state: GameState, player: int, options: list[Option], seed: int,
                    rollouts: int, undeclared_share: float,
                    time_limit: float | None) -> dict[Option, list[float]]:
    """多程序用：在子程序裡跑一部分模擬（校正表由子程序自己載入）。"""
    return evaluate_options(state, player, options, Random(seed), default_calibration(),
                            rollouts, undeclared_share, time_limit)


def evaluate_options_parallel(pool, workers: int, state: GameState, player: int,
                              options: list[Option], rng: Random, rollouts: int,
                              undeclared_share: float,
                              time_limit: float | None) -> dict[Option, list[float]]:
    """把模擬分給 workers 個子程序；每個子程序回傳的清單內部同一索引是同一副牌，
    直接串接仍保持各選項之間的配對。"""
    per_worker = max(1, -(-rollouts // workers))
    tasks = [(state, player, options, rng.randrange(2**31), per_worker, undeclared_share,
              time_limit) for _ in range(workers)]
    merged: dict[Option, list[float]] = {option: [] for option in options}
    wait = None if time_limit is None else time_limit + PARALLEL_GRACE
    for part in pool.starmap_async(_evaluate_chunk, tasks).get(wait):
        for option in options:
            merged[option].extend(part[option])
    return merged


def _discard_after_claim(state: GameState, player: int, action: Action) -> int:
    """吃碰後要打的牌：照規則式 AI 的選法（遵守吃碰後的禁打牌），每個選項只算一次。"""
    me = state.players[player]
    hand = list(me.hand)
    used = list(action.tiles)
    used.remove(action.tile)
    for tile in used:
        hand[tile] -= 1
    return choose_discard(hand, me.open_melds + 1, Random(0),
                          forbidden=forbidden_after_claim(action, action.tile),
                          gold_tiles=state.gold_tiles)


def evaluate_claims(state: GameState, player: int, actions: list[Action], rng: Random,
                    calibration: Calibration, rollouts: int, undeclared_share: float,
                    time_limit: float | None = None) -> dict[Action, list[float]]:
    """吃碰階段：「不要」與各種吃碰槓在同一批補完牌局中的輸贏（同一個索引是同一副牌）。
    別家能胡這張時，不管自己選什麼結果都一樣（胡牌優先）。"""
    tsumo = _average_tai(calibration, self_draw=True)
    ron = _average_tai(calibration, self_draw=False)
    discarder, claimed = state.discard_player, state.last_discard
    # 明槓要先補牌才打，打哪張在模擬中補牌後才決定
    after = {action: _discard_after_claim(state, player, action)
             for action in actions if action.kind in (ActionType.CHOW, ActionType.PUNG)}
    results: dict[Action, list[float]] = {action: [] for action in actions}
    started = time.perf_counter()
    done = 0
    for _ in range(rollouts):
        if time_limit is not None and done and time.perf_counter() - started > time_limit:
            break
        seats, wall = determinize(state, player, rng, calibration, undeclared_share)
        for action in actions:
            copies = [seat.copy() for seat in seats]
            if action.kind == ActionType.PASS or _someone_else_wins(copies, player, discarder,
                                                                     claimed):
                value = play_out(copies, list(wall), discarder, claimed, tsumo, ron,
                                 state.known_dealer, state.dealer_streak, passed=player)
            else:
                value = _play_claim(copies, list(wall), player, action, after.get(action),
                                    tsumo, ron, state.known_dealer, state.dealer_streak)
            results[action].append(after_rake(value[player]))
        done += 1
    return results


def _someone_else_wins(seats: list[Seat], player: int, discarder: int, tile: int) -> bool:
    for step in (1, 2, 3):
        other = (discarder + step) % 4
        if other == player:
            continue
        seat = seats[other]
        seat.hand[tile] += 1
        won = is_win(seat.hand, seat.open_melds)
        seat.hand[tile] -= 1
        if won:
            return True
    return False


def _play_claim(seats: list[Seat], wall: list[int], player: int, action: Action,
                discard: int | None, tsumo: float, ron: float, dealer: int | None,
                streak: int) -> list[float]:
    mine = seats[player]
    used = list(action.tiles)
    used.remove(action.tile)
    for tile in used:
        mine.hand[tile] -= 1
    mine.open_melds += 1
    if action.kind == ActionType.KONG:
        # 明槓補一張：自摸就結束；否則打孤張分數最低的（補牌前選好的那張可能已不是最好）
        if not wall:
            return [0.0] * 4
        drawn = wall.pop()
        mine.hand[drawn] += 1
        if is_win(mine.hand, mine.open_melds):
            amount = BASE + tsumo
            result = [0.0] * 4
            for payer in range(4):
                if payer != player:
                    paid = amount + dealer_bonus(dealer, streak, player, payer)
                    result[payer] -= paid
                    result[player] += paid
            return result
        discard = heuristic_discard(mine.hand, mine.open_melds)
    mine.hand[discard] -= 1
    return play_out(seats, wall, player, discard, tsumo, ron, dealer, streak)


def _claims_chunk(state: GameState, player: int, actions: list[Action], seed: int,
                  rollouts: int, undeclared_share: float,
                  time_limit: float | None) -> dict[Action, list[float]]:
    return evaluate_claims(state, player, actions, Random(seed), default_calibration(),
                           rollouts, undeclared_share, time_limit)


def choose_mc_claim(state: GameState, player: int, rng: Random, calibration: Calibration,
                    rollouts: int, undeclared_share: float, time_limit: float | None,
                    pool=None, workers: int = 1) -> Action | None:
    """吃碰階段：預設照規則式 AI，只有模擬顯示其他選項顯著較好才改。"""
    default = choose_rule_action_for_player(state, player, rng)
    actions = legal_actions(state, player)
    if default is None or default.kind == ActionType.WIN:
        return default
    options = [action for action in actions
               if action.kind in (ActionType.PASS, ActionType.CHOW, ActionType.PUNG,
                                  ActionType.KONG)]
    if len(options) < 2 or default not in options:
        return default
    if pool is not None and workers > 1:
        per_worker = max(1, -(-rollouts // workers))
        tasks = [(state, player, options, rng.randrange(2**31), per_worker, undeclared_share,
                  time_limit) for _ in range(workers)]
        results = {action: [] for action in options}
        try:
            wait = None if time_limit is None else time_limit + PARALLEL_GRACE
            for part in pool.starmap_async(_claims_chunk, tasks).get(wait):
                for action in options:
                    results[action].extend(part[action])
        except MultiprocessingTimeout:
            return default
    else:
        results = evaluate_claims(state, player, options, rng, calibration, rollouts,
                                  undeclared_share, time_limit)
    average = {action: sum(v) / max(1, len(v)) for action, v in results.items()}
    challengers = [action for action in options if action != default
                   and significantly_better(results[action], results[default])]
    return max(challengers, key=average.get) if challengers else default


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
                                time_limit: float | None = None,
                                pool=None, workers: int = 1,
                                mc_claims: bool = MC_CLAIMS) -> Action | None:
    """pool 有給時把模擬分給 workers 個子程序（實戰用；每個子程序各自受 time_limit 限制）。"""
    rng = rng or Random()
    if mc_claims and state.phase == Phase.RESPONSE and not state.robbing_kong \
            and not state.declared[player] and state.last_discard is not None:
        return choose_mc_claim(state, player, rng, calibration or default_calibration(),
                               rollouts, undeclared_share, time_limit, pool, workers)
    if state.phase != Phase.DISCARD or state.declared[player] or player != state.current_player:
        return choose_rule_action_for_player(state, player, rng)
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return win
    kong = choose_self_kong(state, player)
    if kong is not None:
        return kong
    calibration = calibration or default_calibration()
    scores = discard_scores(state, player, rng, calibration, risk_weight=RISK_WEIGHT,
                            undeclared_share=undeclared_share)
    # 只比較不比期望值第一名差（向聽數）的牌：模擬用的簡化打法對留哪些牌有偏好，
    # 曾因此推翻第一名去打讓向聽變差的一萬（實戰 17:53）
    me = state.players[player]

    def level(tile: int) -> int:
        after = list(me.hand)
        after[tile] -= 1
        return shanten(after, me.open_melds)

    ordered = sorted(scores, key=scores.get, reverse=True)
    ranked = [tile for tile in ordered if level(tile) <= level(ordered[0])][:candidates]
    options = [Option(tile, False) for tile in ranked]
    options += [Option(tile, True) for tile in ranked
                if Action(ActionType.DECLARE, tile=tile) in actions]
    top = ranked[0]
    default = Option(top, Option(top, True) in options and declare_is_worth(state, player, top))
    best = default
    if len(options) > 1:
        if pool is not None and workers > 1:
            try:
                results = evaluate_options_parallel(pool, workers, state, player, options, rng,
                                                    rollouts, undeclared_share, time_limit)
            except MultiprocessingTimeout:
                results = {option: [] for option in options}
        else:
            results = evaluate_options(state, player, options, rng, calibration, rollouts,
                                       undeclared_share, time_limit)
        average = {option: sum(values) / max(1, len(values)) for option, values in results.items()}
        challengers = [option for option in options if option != default
                       and significantly_better(results[option], results[default])]
        if challengers:
            best = max(challengers, key=average.get)
    kind = ActionType.DECLARE if best.declare else ActionType.DISCARD
    return Action(kind, tile=best.tile)
