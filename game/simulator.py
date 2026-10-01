"""用隨機策略執行對局，驗證規則引擎可供 self-play 使用。"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from typing import Callable

from .rules import Action, ActionType, GameState, Phase, apply_action, initial_state, legal_actions
from .scoring import BASE_TAI, TAI, Score, score_flower_win, score_hand
from .tiles import tile_name

ActionChooser = Callable[[GameState, Random], tuple[int, Action] | None]
PlayerPolicy = Callable[[GameState, int, Random], Action | None]


def make_uniform_policy(policy: PlayerPolicy) -> PlayerPolicy:
    """建立四家共用同一玩家策略的 self-play policy。"""
    def uniform_policy(state: GameState, player: int, rng: Random) -> Action | None:
        return policy(state, player, rng)

    return uniform_policy


@dataclass(frozen=True)
class GameResult:
    state: GameState
    steps: int
    winner: int | None = None
    win_tile: int | None = None
    win_by_discard: bool = False
    history: tuple["GameEvent", ...] = ()
    score: Score | None = None
    discarder: int | None = None
    discard_count: int = 0
    discards_by_player: tuple[int, ...] = (0, 0, 0, 0)


@dataclass(frozen=True)
class GameEvent:
    step: int
    player: int
    phase: Phase
    action: Action


@dataclass(frozen=True)
class SimulationStats:
    games: int
    wins: int
    draws: int
    total_steps: int
    discard_wins: int = 0
    self_draw_wins: int = 0
    total_tai: int = 0
    total_discards: int = 0
    deal_in_count: int = 0
    discards_by_player: tuple[int, ...] = (0, 0, 0, 0)
    deal_ins_by_player: tuple[int, ...] = (0, 0, 0, 0)
    wins_by_player: tuple[int, ...] = (0, 0, 0, 0)
    self_draw_wins_by_player: tuple[int, ...] = (0, 0, 0, 0)
    discard_wins_by_player: tuple[int, ...] = (0, 0, 0, 0)

    @property
    def average_steps(self) -> float:
        return self.total_steps / self.games if self.games else 0.0

    @property
    def average_tai(self) -> float:
        return self.total_tai / self.wins if self.wins else 0.0

    @property
    def deal_in_rate(self) -> float:
        return self.deal_in_count / self.total_discards if self.total_discards else 0.0

    @property
    def deal_in_game_rate(self) -> float:
        """發生放槍的對局比例；不同於每次棄牌的 deal_in_rate。"""
        return self.discard_wins / self.games if self.games else 0.0

    @property
    def discard_win_rate(self) -> float:
        """放銃胡占所有胡牌的比例。"""
        return self.discard_wins / self.wins if self.wins else 0.0

    @property
    def deal_in_rate_by_player(self) -> tuple[float, ...]:
        return tuple(
            deal_ins / discards if discards else 0.0
            for deal_ins, discards in zip(self.deal_ins_by_player, self.discards_by_player)
        )

    @property
    def win_rate_by_player(self) -> tuple[float, ...]:
        return tuple(wins / self.games for wins in self.wins_by_player)

    @staticmethod
    def _wilson(successes: int, trials: int) -> tuple[float, float]:
        if trials == 0:
            return (0.0, 0.0)
        z = 1.96
        rate = successes / trials
        denominator = 1 + z * z / trials
        center = (rate + z * z / (2 * trials)) / denominator
        margin = z * sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))

    @property
    def deal_in_confidence_by_player(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            self._wilson(deal_ins, discards)
            for deal_ins, discards in zip(self.deal_ins_by_player, self.discards_by_player)
        )

    @property
    def win_confidence_by_player(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._wilson(wins, self.games) for wins in self.wins_by_player)


def choose_random_action(state: GameState, rng: Random) -> tuple[int, Action] | None:
    """選一個合法動作；胡牌優先，其餘動作隨機。"""
    if state.phase == Phase.RESPONSE:
        players = state.response_players or (
            [state.response_player] if state.response_player is not None else []
        )
    else:
        players = [state.current_player]
    if not players:
        return None
    available = [(player, legal_actions(state, player)) for player in players]
    available = [(player, actions) for player, actions in available if actions]
    if not available:
        return None
    winning = [
        (player, action)
        for player, actions in available
        for action in actions
        if action.kind == ActionType.WIN
    ]
    if winning:
        return winning[0]
    player, actions = rng.choice(available)
    return player, rng.choice(actions)


def _choose_random_for_player(state: GameState, player: int, rng: Random) -> Action | None:
    actions = legal_actions(state, player)
    winning = [action for action in actions if action.kind == ActionType.WIN]
    if winning:
        return winning[0]
    return rng.choice(actions) if actions else None


def _choose_with_player_policy(
    state: GameState, rng: Random, policy: PlayerPolicy
) -> tuple[int, Action] | None:
    players = state.response_players if state.phase == Phase.RESPONSE else [state.current_player]
    choices = [(player, policy(state, player, rng)) for player in players]
    choices = [(player, action) for player, action in choices if action is not None]
    if not choices:
        return None
    winning = [(player, action) for player, action in choices
               if action.kind == ActionType.WIN]
    if winning:
        return winning[0]
    # 同時回應時：胡 > 槓/碰 > 吃；同優先權依座位順序
    for kinds in ({ActionType.KONG, ActionType.PUNG}, {ActionType.CHOW}):
        claims = [(player, action) for player, action in choices if action.kind in kinds]
        if claims:
            return claims[0]
    active = [(player, action) for player, action in choices
              if action.kind != ActionType.PASS]
    return active[0] if active else choices[0]


def _win_events(state: GameState, self_draw: bool) -> list[str]:
    """胡牌當下（套用動作前）的對局事件台。"""
    if self_draw:
        events = ["槓上開花"] if state.after_kong else []
        if not state.drawable:
            events.append("海底撈月")
        return events
    if state.robbing_kong:
        return ["搶槓"]
    return [] if state.drawable else ["河底撈魚"]


def _flower_win_result(state: GameState, steps: int, history: list[GameEvent],
                       discard_count: int, discards_by_player: list[int]) -> GameResult:
    flower_win = state.flower_win
    assert flower_win is not None
    dealer = state.dealer in {flower_win.winner, flower_win.from_player}
    return GameResult(
        state, steps, winner=flower_win.winner,
        win_by_discard=flower_win.from_player is not None,
        history=tuple(history),
        score=score_flower_win(flower_win.kind, dealer, state.dealer_streak if dealer else 0),
        discarder=flower_win.from_player, discard_count=discard_count,
        discards_by_player=tuple(discards_by_player),
    )


def play_game(
    rng: Random | None = None,
    max_steps: int = 10_000,
    choose_action: ActionChooser | None = None,
    record_history: bool = False,
    player_policy: PlayerPolicy | None = None,
    dealer: int = 0,
    swap: bool = False,
    dealer_streak: int = 0,
) -> GameResult:
    """執行一局對局；摸到留牌或達到步數上限都視為流局。"""
    if max_steps <= 0:
        raise ValueError("max_steps 必須為正數")
    rng = rng or Random()
    choose_action = choose_action or choose_random_action
    state = initial_state(rng, dealer=dealer, swap=swap, dealer_streak=dealer_streak)
    history: list[GameEvent] = []
    discard_count = 0
    discards_by_player = [0, 0, 0, 0]
    for steps in range(1, max_steps + 1):
        if state.flower_win is not None:
            return _flower_win_result(state, steps - 1, history,
                                      discard_count, discards_by_player)
        if state.phase == Phase.ENDED:
            return GameResult(state, steps - 1, history=tuple(history),
                              discard_count=discard_count,
                              discards_by_player=tuple(discards_by_player))
        selected = (
            _choose_with_player_policy(state, rng, player_policy)
            if player_policy is not None else choose_action(state, rng)
        )
        if selected is None:
            state.phase = Phase.ENDED
            return GameResult(state, steps - 1, history=tuple(history),
                              discard_count=discard_count,
                              discards_by_player=tuple(discards_by_player))
        player, action = selected
        if record_history:
            history.append(GameEvent(steps, player, state.phase, action))
        win_by_discard = state.phase == Phase.RESPONSE and action.kind == ActionType.WIN
        discarder = state.discard_player if win_by_discard else None
        if action.kind == ActionType.WIN:
            events = _win_events(state, self_draw=not win_by_discard)
            win_tile = action.tile if win_by_discard else state.last_drawn
        if action.kind in {ActionType.DISCARD, ActionType.DECLARE}:
            discard_count += 1
            discards_by_player[player] += 1
        apply_action(state, action, player)
        if action.kind == ActionType.WIN:
            winner_state = state.players[player]
            score = score_hand(
                winner_state.hand,
                winner_state.melds,
                flowers=winner_state.flowers,
                self_draw=not win_by_discard,
                seat_wind=state.seat_winds[player],
                round_wind=state.round_wind,
                win_tile=win_tile,
                # 莊家有參與（莊家胡或莊家放槍）才加莊家台；非莊自摸時莊家
                # 多付的部分屬於付款計算，不算在胡牌者的台數裡。
                dealer=state.dealer in {player, discarder},
                dealer_streak=state.dealer_streak if state.dealer in {player, discarder} else 0,
                events=events,
                gold_tiles=state.gold_tiles,
                declared=state.declared[player],
                early_declared=state.early_declared[player],
            )
            return GameResult(
                state,
                steps,
                winner=player,
                win_tile=win_tile,
                win_by_discard=win_by_discard,
                history=tuple(history),
                score=score,
                discarder=discarder,
                discard_count=discard_count,
                discards_by_player=tuple(discards_by_player),
            )
    state.phase = Phase.ENDED
    return GameResult(state, max_steps, history=tuple(history),
                      discard_count=discard_count,
                      discards_by_player=tuple(discards_by_player))


RAKE = 0.15
"""實戰贏的錢遊戲抽 15%（eval.stats 由金幣餘額對出），輸的照付；比較策略、蒙地卡羅模擬都照實戰算。"""


def after_rake(net: float) -> float:
    return net * (1 - RAKE) if net > 0 else net


def payments(result: GameResult, base: int = BASE_TAI) -> list[int]:
    """每家這局的輸贏（台）：放槍由放槍者付，自摸三家各付。非莊家自摸時莊家多付莊家台
    與連莊台（1 + 2n；莊家胡或莊家放槍時已算在胡牌者的台數裡）。"""
    pay = [0, 0, 0, 0]
    if result.winner is None or result.score is None:
        return pay
    amount = base + result.score.tai
    losers = [result.discarder] if result.win_by_discard else         [player for player in range(4) if player != result.winner]
    dealer = result.state.dealer
    for loser in losers:
        extra = 0
        if not result.win_by_discard and loser == dealer and result.winner != dealer:
            extra = TAI["莊家"] + 2 * getattr(result.state, "dealer_streak", 0)
        pay[loser] -= amount + extra
        pay[result.winner] += amount + extra
    return pay


MAX_HANDS = 40
"""一圈最多打幾局（防止連莊無限延長）。"""


@dataclass(frozen=True)
class MatchResult:
    hands: tuple[GameResult, ...]
    totals: tuple[int, int, int, int]
    """四家整圈的輸贏（台）。"""


def play_match(
    rng: Random | None = None,
    player_policy: PlayerPolicy | None = None,
    first_dealer: int = 0,
    swap: bool = True,
    hand_seed=None,
) -> MatchResult:
    """打一圈（東風圈）：莊家從 first_dealer 開始，莊家胡牌或流局就連莊（連莊數 +1），
    否則下家接莊、連莊數歸零；四家都當過莊、莊家輪回 first_dealer 時結束。
    hand_seed(局序) 有給時，每局用它產生的種子洗牌（配對比較用）。"""
    rng = rng or Random()
    dealer, streak, passes = first_dealer, 0, 0
    hands: list[GameResult] = []
    totals = [0, 0, 0, 0]
    while passes < 4 and len(hands) < MAX_HANDS:
        hand_rng = Random(hand_seed(len(hands))) if hand_seed is not None else rng
        result = play_game(hand_rng, player_policy=player_policy, dealer=dealer, swap=swap,
                           dealer_streak=streak)
        hands.append(result)
        for seat, amount in enumerate(payments(result)):
            totals[seat] += amount
        if result.winner is None or result.winner == dealer:
            streak += 1
        else:
            dealer, streak, passes = (dealer + 1) % 4, 0, passes + 1
    return MatchResult(tuple(hands), tuple(totals))


def format_game_trace(result: GameResult, limit: int | None = None) -> str:
    """把單局 history 格式化成適合終端閱讀的逐步紀錄。"""
    events = result.history if limit is None else result.history[:limit]
    lines = []
    for event in events:
        detail = event.action.kind.value
        if event.action.tile is not None:
            detail += f" {tile_name(event.action.tile)}"
        if event.action.tiles:
            detail += " [" + " ".join(tile_name(tile) for tile in event.action.tiles) + "]"
        lines.append(f"{event.step:04d} P{event.player + 1} {event.phase.value:8s} {detail}")
    if limit is not None and len(result.history) > limit:
        lines.append(f"...（省略 {len(result.history) - limit} 步）")
    outcome = "流局" if result.winner is None else f"P{result.winner + 1} 胡牌"
    lines.append(f"結果：{outcome}，總步數 {result.steps}")
    return "\n".join(lines)


def simulate_games(
    games: int,
    seed: int | None = None,
    max_steps: int = 10_000,
    choose_action: ActionChooser | None = None,
    player_policy: PlayerPolicy | None = None,
) -> SimulationStats:
    """執行多局隨機對局，回傳胡牌與流局統計。"""
    if games < 0:
        raise ValueError("games 不可為負數")
    rng = Random(seed)
    wins = 0
    discard_wins = 0
    self_draw_wins = 0
    total_tai = 0
    total_discards = 0
    deal_in_count = 0
    discards_by_player = [0, 0, 0, 0]
    deal_ins_by_player = [0, 0, 0, 0]
    wins_by_player = [0, 0, 0, 0]
    self_draw_wins_by_player = [0, 0, 0, 0]
    discard_wins_by_player = [0, 0, 0, 0]
    total_steps = 0
    for _ in range(games):
        result = play_game(rng, max_steps, choose_action, player_policy=player_policy)
        wins += result.winner is not None
        if result.winner is not None:
            wins_by_player[result.winner] += 1
            if result.win_by_discard:
                discard_wins_by_player[result.winner] += 1
            else:
                self_draw_wins_by_player[result.winner] += 1
        if result.winner is not None:
            if result.win_by_discard:
                discard_wins += 1
            else:
                self_draw_wins += 1
            if result.score is not None:
                total_tai += result.score.tai
        total_steps += result.steps
        total_discards += result.discard_count
        deal_in_count += result.discarder is not None
        if result.discarder is not None:
            deal_ins_by_player[result.discarder] += 1
        for player, count in enumerate(result.discards_by_player):
            discards_by_player[player] += count
    return SimulationStats(
        games, wins, games - wins, total_steps,
        discard_wins, self_draw_wins, total_tai, total_discards, deal_in_count,
        tuple(discards_by_player), tuple(deal_ins_by_player),
        tuple(wins_by_player), tuple(self_draw_wins_by_player),
        tuple(discard_wins_by_player),
    )