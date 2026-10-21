"""人機對戰的牌局流程（不含視窗）：自己坐 0 號位，三家電腦，打一圈。

視窗在另一條執行緒（Tk 的主執行緒）；牌局在這裡的執行緒跑，輪到自己要決定時呼叫 ask 等視窗回覆。
規則、計分、連莊都沿用模擬器（game.simulator.play_game、payments），所以和訓練、arena 打的是同一套規則。
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Callable

from game.rules import Action, ActionType, GameState, Phase, legal_actions
from game.scoring import TAI
from game.simulator import MAX_HANDS, GameResult, PlayerPolicy, payments, play_game
from game.tiles import tile_name

HUMAN = 0
SEAT_NAMES = ("自己", "下家", "對家", "上家")
WIND_NAMES = {27: "東", 28: "南", 29: "西", 30: "北"}
MONEY_PER_TAI = 50
"""訓練場：底 100（= 2 台）、每台 50。"""

Ask = Callable[[GameState, list[Action]], Action]
"""給視窗一份局面副本與能做的動作，等使用者選一個。換三張時動作是 [Action(SWAP)]，回傳的 tiles 是要換的牌。"""


class HumanSeat:
    """自己那一家的策略：只有一種選擇時自動做（摸牌、只能過、聽牌後代打），其他交給視窗。"""

    def __init__(self, ask: Ask) -> None:
        self.ask = ask

    def __call__(self, state: GameState, player: int, rng: Random) -> Action | None:
        actions = legal_actions(state, player)
        if not actions:
            return None
        if state.phase != Phase.SWAP and len(actions) == 1:
            return actions[0]
        return self.ask(copy.deepcopy(state), actions)


def seat_policy(human: PlayerPolicy, bot: PlayerPolicy) -> PlayerPolicy:
    def policy(state: GameState, player: int, rng: Random) -> Action | None:
        return (human if player == HUMAN else bot)(state, player, rng)

    return policy


def describe_action(player: int, action: Action) -> str | None:
    """牌局訊息（摸牌、過、換三張不顯示）。"""
    who = SEAT_NAMES[player]
    tile = tile_name(action.tile) if action.tile is not None else ""
    kind = action.kind
    if kind == ActionType.DISCARD:
        return f"{who} 打 {tile}"
    if kind == ActionType.DECLARE:
        return f"{who} 打 {tile} 並宣告聽牌"
    if kind == ActionType.CHOW:
        return f"{who} 吃 {' '.join(tile_name(t) for t in action.tiles)}"
    if kind == ActionType.PUNG:
        return f"{who} 碰 {tile}"
    if kind == ActionType.KONG:
        return f"{who} 槓 {tile}"
    if kind == ActionType.WIN:
        return f"{who} 胡！"
    return None


def score_lines(result: GameResult) -> list[str]:
    """台數明細，例如「金牌 ×2　2 台」。"""
    if result.score is None:
        return []
    lines = []
    for name, count in Counter(result.score.patterns).items():
        each = TAI.get(name)
        label = name if count == 1 else f"{name} ×{count}"
        lines.append(f"{label}　{each * count} 台" if each is not None else label)
    return lines


@dataclass
class HandSummary:
    number: int
    """第幾局（從 1 開始）。"""
    result: GameResult
    pay: list[int]
    """這局四家輸贏（台，含底）。"""
    totals: list[int]
    """到這局為止四家合計（台）。"""
    dealer: int
    streak: int


def play_round(policy: PlayerPolicy, rng: Random, on_hand_end: Callable[[HandSummary], None],
               observer=None, first_dealer: int = 0, swap: bool = True) -> list[int]:
    """打一圈（與 game.simulator.play_match 相同的連莊規則），每局結束呼叫 on_hand_end（視窗在這裡等「繼續」）。
    回傳四家整圈輸贏（台）。"""
    dealer, streak, passes, number = first_dealer, 0, 0, 0
    totals = [0, 0, 0, 0]
    while passes < 4 and number < MAX_HANDS:
        number += 1
        result = play_game(rng, player_policy=policy, dealer=dealer, swap=swap,
                           dealer_streak=streak, observer=observer)
        pay = payments(result)
        for seat, amount in enumerate(pay):
            totals[seat] += amount
        on_hand_end(HandSummary(number, result, pay, list(totals), dealer, streak))
        if result.winner is None or result.winner == dealer:
            streak += 1
        else:
            dealer, streak, passes = (dealer + 1) % 4, 0, passes + 1
    return totals
