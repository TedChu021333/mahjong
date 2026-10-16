"""大老二模擬器：四家對打一局、計分。

座位 0-3，出牌順序 0 → 1 → 2 → 3（遊戲是逆時針，所以座位 0 是自己時 1 是下家＝右邊）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable

from .cards import CLUB_THREE, RANK_TWO, deal, rank_of
from .combos import Combo, first_card_required, legal_moves
from .rules import DEFAULT_RULES, Rules

PLAYERS = 4


@dataclass(frozen=True)
class Play:
    seat: int
    combo: Combo | None   # None = PASS


@dataclass
class View:
    """輪到某家時他看得到的資訊。"""
    seat: int
    hand: tuple[int, ...]
    counts: tuple[int, ...]          # 四家剩幾張（依座位）
    last: Combo | None               # 要壓的牌；None 表示自己帶頭
    last_seat: int | None
    history: list[Play]
    moves: list[Combo]               # 能出的牌
    can_pass: bool
    rules: Rules = DEFAULT_RULES

    @property
    def played(self) -> set[int]:
        return {c for p in self.history if p.combo for c in p.combo.cards}

    def relative(self, seat: int) -> int:
        """1 = 下家、2 = 對家、3 = 上家。"""
        return (seat - self.seat) % PLAYERS


Policy = Callable[[View, Random], Combo | None]


@dataclass
class GameResult:
    winner: int
    hands: list[tuple[int, ...]]     # 結束時四家剩的牌
    history: list[Play]
    starter: int
    conditions: list[list[str]] = field(default_factory=list)   # 每家輸家符合的加倍項目
    scores: list[int] = field(default_factory=list)             # 以「張」為單位，贏家為正


def penalty_conditions(hand, winner_last: Combo, starter_grand_slam: bool) -> list[str]:
    n = len(hand)
    out = []
    if n > 9:
        out.append("超過9張牌")
    if any(rank_of(c) == RANK_TWO for c in hand):
        out.append("老2未打")
    if n == 13:
        out.append("13張未出")
    if CLUB_THREE in winner_last.cards:
        out.append("最後脫手3")
    if starter_grand_slam:
        out.append("滿貫全壘打")
    return out


def penalty(hand, conditions, rules: Rules = DEFAULT_RULES) -> int:
    return len(hand) * rules.rate ** len(conditions)


SwapPolicy = Callable[[tuple[int, ...], Random], list[int]]


def swap_cards(hands: list[list[int]], swappers: list[SwapPolicy], rng: Random) -> list[list[int]]:
    """換三張：每家選 0～3 張放進同一堆，洗過後每家拿回同樣張數（可能拿回自己的牌）。"""
    given = []
    for s in range(PLAYERS):
        out = list(dict.fromkeys(swappers[s](tuple(hands[s]), rng)))
        if len(out) > 3 or any(c not in hands[s] for c in out):
            raise RuntimeError(f"座位 {s} 換牌不合法：{out}")
        given.append(out)
    pool = [c for out in given for c in out]
    rng.shuffle(pool)
    new_hands = []
    for s in range(PLAYERS):
        kept = [c for c in hands[s] if c not in given[s]]
        new_hands.append(sorted(kept + pool[:len(given[s])]))
        pool = pool[len(given[s]):]
    return new_hands


def no_swap(hand, rng) -> list[int]:
    return []


def play_game(policies: list[Policy], rng: Random, rules: Rules = DEFAULT_RULES,
              hands: list[list[int]] | None = None,
              swappers: list[SwapPolicy] | None = None) -> GameResult:
    hands = [sorted(h) for h in (hands or deal(rng))]
    if rules.swap and swappers is not None:
        hands = swap_cards(hands, swappers, rng)
    starter = next(s for s in range(PLAYERS) if CLUB_THREE in hands[s])
    return _run(hands, [], starter, None, None, set(), starter, policies, rng, rules)


def continue_game(view: View, hands: list[list[int]], move: Combo | None, policies: list[Policy],
                  rng: Random) -> GameResult:
    """從 view 的局面（四家手牌由 hands 補完）讓 view.seat 出 move，之後照 policies 打完。"""
    history = list(view.history)
    starter = history[0].seat if history else view.seat
    passed = set()
    for play in reversed(history):
        if play.combo is not None:
            break
        passed.add(play.seat)
    if view.last is None:
        passed = set()
    hands = [sorted(h) for h in hands]
    # 只有第一步用指定的出法，之後換回原本的策略
    wrapped = list(policies)
    wrapped[view.seat] = _once(lambda v, r: move, policies[view.seat])
    return _run(hands, history, view.seat, view.last, view.last_seat, passed, starter, wrapped, rng,
                view.rules)


def _once(first: Policy, then: Policy) -> Policy:
    used = [False]

    def policy(v: View, r: Random):
        if used[0]:
            return then(v, r)
        used[0] = True
        return first(v, r)

    return policy


def _run(hands, history, seat, last, last_seat, passed, starter, policies, rng, rules) -> GameResult:
    first = not any(p.combo for p in history)
    while True:
        must = first_card_required(rules) if first else None
        moves = legal_moves(hands[seat], last, rules, must)
        view = View(seat, tuple(hands[seat]), tuple(len(h) for h in hands), last, last_seat,
                    list(history), moves, last is not None, rules)
        choice = policies[seat](view, rng) if moves else None
        if choice is None and last is None:
            raise RuntimeError(f"座位 {seat} 帶頭不能 PASS")
        if choice is not None and choice not in moves:
            raise RuntimeError(f"座位 {seat} 出了不合法的牌：{choice}")
        history.append(Play(seat, choice))
        if choice is None:
            passed.add(seat)
        else:
            first = False
            for c in choice.cards:
                hands[seat].remove(c)
            last, last_seat = choice, seat
            if not hands[seat]:
                break
        seat = (seat + 1) % PLAYERS
        while rules.pass_locks_out and seat in passed and seat != last_seat:
            seat = (seat + 1) % PLAYERS
        if seat == last_seat:      # 其他人都 PASS，拿回出牌權
            last, passed = None, set()
    winner = seat
    grand_slam = winner == starter and all(p.combo is None for p in history if p.seat != starter)
    conditions, scores = [], []
    for s in range(PLAYERS):
        if s == winner:
            conditions.append([])
            scores.append(0)
        else:
            cond = penalty_conditions(hands[s], last, grand_slam)
            conditions.append(cond)
            scores.append(-penalty(hands[s], cond, rules))
    scores[winner] = -sum(scores)
    return GameResult(winner, [tuple(h) for h in hands], history, starter, conditions, scores)
