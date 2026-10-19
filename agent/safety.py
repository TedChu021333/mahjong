"""有對手宣告聽牌時，用期望值檢查網路的打牌與吃碰（agent.nn_agent 使用）。

每張可打的牌：淨值 = 進攻期望收益 × (1 − 抽成) − 放槍機率 × 放槍要付的台數 × DANGER_SCALE。
- 進攻期望收益查 agent/calibration.json（打完後的向聽、進張、牌牆剩幾張 → 平均贏幾台）；打完聽牌時
  再乘「這手聽的牌實際台數 / 模擬器胡牌平均台數」，大牌值得多冒險、小牌早點棄胡。
- 放槍機率與損失由 agent.opponent_model.estimate_danger 估計（宣告者一定聽牌、不聽自己打過的牌）。
網路要打的牌照打，只有改打「放槍機率較低」的牌、少付的放槍損失扣掉少掉的進攻收益後還多
OVERRIDE_MARGIN 台以上才換；吃碰時比較
「吃碰完打最好的那張」與「不要」的淨值，不要明顯較好才改成不要。

10/17 實戰 45 局放槍 10 次，6 次放給已宣告的人且當下都有認到宣告：例如兩家宣告、自己還三向聽，
網路打生張五條（估計 8.5%），手上一筒只有 0.3%、改打它進攻收益不會少（eval.deal_in）。
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

from agent.opponent_model import BASE, Calibration, default_calibration, estimate_danger
from agent.shanten import effective_tiles, shanten
from agent.rule_agent import visible_tile_counts
from game.rules import Action, ActionType, GameState, Meld, allowed_discards, forbidden_after_claim
from game.scoring import score_hand
from game.simulator import RAKE
from game.tiles import tile_name
from game.win import winning_tiles

DANGER_SCALE = 1.3
"""對手模型的放槍機率偏低：10/5 校準估計 27 次、實際 35 次。"""
OVERRIDE_MARGIN = 0.2
"""至少好這麼多台才推翻網路。進攻收益表依向聽與進張分組，同一組內的差別看不出來（很多改法顯示
「進攻少 0.00」），太小的差距留給網路。10/17 記錄重放：門檻 0.05 時有人宣告的出牌 66 次改 37 次，
0.2 改 29 次；放槍那幾手（少付 0.35～1.41 台）都還會改。"""
SAMPLES = 40
MIN_VALID = 150
"""每位宣告者湊滿這麼多副有效的聽牌手牌（見 opponent_model.estimate_danger）；原本只嘗試 160 次、
有效的約 24 副，估計值亂跳。"""


@dataclass(frozen=True)
class Choice:
    tile: int
    net: float
    offense: float
    danger: float
    """放槍機率（未乘 DANGER_SCALE）。"""


def anyone_declared(state: GameState, player: int) -> bool:
    return any(state.declared[seat] for seat in range(4) if seat != player)


def _hand_value_ratio(state: GameState, player: int, hand: list[int], melds: list,
                      visible: list[int], calibration: Calibration) -> float:
    """聽牌的手：聽的牌實際台數（依剩餘張數加權，放槍胡）/ 模擬器胡牌平均台數；算不出來回傳 1。
    提示模式自己的副露內容讀不到（佔位副露）、花牌也不知道，這時就用平均。"""
    if any(not meld.tiles for meld in melds):
        return 1.0
    total = weight = 0.0
    dealer = state.known_dealer == player
    for wait in winning_tiles(hand, len(melds)):
        live = max(0, 4 - visible[wait])
        if not live:
            continue
        counts = list(hand)
        counts[wait] += 1
        try:
            tai = score_hand(counts, melds, state.players[player].flowers,
                             seat_wind=state.seat_winds[player], round_wind=state.round_wind,
                             win_tile=wait, dealer=dealer,
                             dealer_streak=state.dealer_streak if dealer else 0,
                             gold_tiles=state.gold_tiles).tai
        except ValueError:
            return 1.0
        total += live * (BASE + tai)
        weight += live
    if not weight:
        return 1.0
    average, _ = calibration.tai.get((True, False), (5.0, 0))
    return (total / weight) / (BASE + average)


def evaluate(state: GameState, player: int, hand: list[int], melds: list, candidates,
             loss: dict[int, float], probability: dict[int, float],
             calibration: Calibration) -> dict[int, Choice]:
    visible = visible_tile_counts(state, player)
    result = {}
    for tile in candidates:
        after = list(hand)
        after[tile] -= 1
        level = shanten(after, len(melds))
        outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(after, len(melds)))
        offense = calibration.offense_value(level, outs, state.drawable)
        if level == 0:
            offense *= _hand_value_ratio(state, player, after, melds, visible, calibration)
        offense *= 1 - RAKE
        result[tile] = Choice(tile, offense - DANGER_SCALE * loss[tile], offense,
                              probability[tile])
    return result


def safer_discard(state: GameState, player: int, tile: int, rng: Random,
                  calibration: Calibration | None = None) -> tuple[int, str | None]:
    """網路要打 tile；有人宣告時若有更划算的較安全牌就換。回傳 (要打的牌, 說明或 None)。"""
    if not anyone_declared(state, player):
        return tile, None
    calibration = calibration or default_calibration()
    me = state.players[player]
    candidates = allowed_discards(me.hand, state.forbidden_discards if state.claimed else ())
    if tile not in candidates or len(candidates) < 2:
        return tile, None
    danger = estimate_danger(state, player, candidates, rng, calibration, SAMPLES,
                             min_valid=MIN_VALID)
    choices = evaluate(state, player, me.hand, me.melds, candidates, danger.expected_loss,
                       danger.probability, calibration)
    current = choices[tile]
    safer = [c for c in choices.values() if c.danger < current.danger]
    if not safer:
        return tile, None
    best = max(safer, key=lambda c: _gain(current, c))
    if _gain(current, best) <= OVERRIDE_MARGIN:
        return tile, None
    return best.tile, _describe(current, best)


def _gain(current: Choice, other: Choice) -> float:
    """改打 other 少付的放槍損失，扣掉進攻少掉的部分。查表說 other 進攻較好時不算進來：網路的進攻判斷
    比這張粗分組的表準（不算的話，今天有人宣告時的出牌會被改掉一半以上，很多只是查表偏好）。"""
    risk_saved = (current.offense - current.net) - (other.offense - other.net)
    return risk_saved - max(0.0, current.offense - other.offense)


def claim_is_worse(state: GameState, player: int, action: Action, rng: Random,
                   calibration: Calibration | None = None) -> str | None:
    """有人宣告時，網路要吃碰：吃碰完打最好的牌，淨值比「不要」差 OVERRIDE_MARGIN 以上就回傳說明
    （改成不要），否則 None。「不要」的淨值用現在的手牌（下一手要打的牌以現在最安全的那張近似）。"""
    if action.kind not in (ActionType.CHOW, ActionType.PUNG) or not anyone_declared(state, player):
        return None
    calibration = calibration or default_calibration()
    me = state.players[player]
    hand = list(me.hand)
    used = list(action.tiles)
    used.remove(action.tile)
    for tile in used:
        hand[tile] -= 1
    melds = list(me.melds) + [_claimed_meld(action)]
    after_claim = allowed_discards(hand, forbidden_after_claim(action, action.tile))
    tiles = [tile for tile, count in enumerate(me.hand) if count]
    danger = estimate_danger(state, player, tiles, rng, calibration, SAMPLES,
                             min_valid=MIN_VALID)
    claimed = evaluate(state, player, hand, melds, after_claim, danger.expected_loss,
                       danger.probability, calibration)
    best = max(claimed.values(), key=lambda c: c.net)
    # 不要：手牌維持現狀，進攻收益就是現在這手的狀態；之後還要打一張，取現在最小的放槍損失
    visible = visible_tile_counts(state, player)
    level = shanten(me.hand, me.open_melds)
    outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(me.hand, me.open_melds))
    offense = calibration.offense_value(level, outs, state.drawable)
    if level == 0:
        offense *= _hand_value_ratio(state, player, me.hand, me.melds, visible, calibration)
    offense *= 1 - RAKE
    pass_net = offense - DANGER_SCALE * min(danger.expected_loss[t] for t in tiles)
    if pass_net - best.net <= OVERRIDE_MARGIN:
        return None
    return (f"有人宣告，吃碰後打{tile_name(best.tile)}放槍 {best.danger:.1%}、淨值 {best.net:+.2f} 台，"
            f"不要 {pass_net:+.2f} 台")


def _claimed_meld(action: Action) -> Meld:
    kind = "chow" if action.kind == ActionType.CHOW else "pung"
    return Meld(kind, tuple(sorted(action.tiles)), action.from_player)


def _describe(current: Choice, best: Choice) -> str:
    risk_saved = (current.offense - current.net) - (best.offense - best.net)
    return (f"有人宣告，{tile_name(current.tile)}放槍 {current.danger:.1%} → 改打{tile_name(best.tile)} "
            f"{best.danger:.1%}：少付 {risk_saved:.2f} 台、進攻少 "
            f"{max(0.0, current.offense - best.offense):.2f} 台")
