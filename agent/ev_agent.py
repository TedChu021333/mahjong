"""期望值打牌：每張能打的牌 = 進攻的期望收益 − 放槍的期望損失，打分數最高的。

進攻收益查模擬器蒐集的表（打完後的向聽數、有效進張、牌牆剩餘 → 這局靠自己胡牌平均贏
幾台），放槍損失由 agent.opponent_model 抽樣對手手牌估計。棄胡不再是寫死的門檻：進攻
收益低、危險高時自然會打安全的牌。胡牌、吃碰槓、換三張與宣告聽牌沿用規則式 AI。
"""
from __future__ import annotations

from random import Random

from agent.opponent_model import (
    UNDECLARED_TENPAI_SHARE,
    Calibration,
    default_calibration,
    estimate_danger,
)
from agent.rule_agent import (
    choose_rule_action_for_player,
    gold_in_use,
    keep_value,
    choose_self_kong,
    declare_is_worth,
    visible_tile_counts,
)
from agent.shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, allowed_discards, legal_actions

RISK_WEIGHT = 0.5
"""放槍損失的權重（1 = 純期望值）。arena 各 1500 對、對規則式 AI：0.5 → −0.029±0.430，
1 → −0.138±0.428，2 → −0.289±0.435，皆打平；0.5 最接近，放槍率 19.5% → 17.6%。"""
GOLD_VALUE_SHARE = 1 / 8
"""胡牌平均約 底 2 + 6 台，用得上的金牌多 1 台 ≈ 收益多 1/8。"""
GOLD_WAIT_SHARE = 3 / 8
HONOR_TAI_SHARE = GOLD_VALUE_SHARE
"""碰出來會加台的字牌（三元牌、圈風、已知的門風）每 1 台同樣算進攻收益的 1/8；手上已成刻子
全算，對子依還沒出現的張數打折（見 honor_pung_value）。實戰 10/1 01:06 曾為了稍微安全一點拆掉一對東
（東風圈，碰出來有台），因為進攻收益只看向聽與進張。"""
PAIR_PUNG_CHANCE = 0.25
"""對子每剩一張沒出現的，約有這麼多機會湊成刻子（粗估，剩 2 張 → 0.5）。"""
OUTS_TIE_BREAK = 1e-3
KEEP_TIE_BREAK = 1e-6
"""進張也相同時先打最孤立的牌（孤張字牌優先），見 rule_agent.keep_value。"""
"""表是分組的，同一組內再以精確的有效進張數區分（不改變不同組之間的排序）。"""


def honor_tai(state: GameState, player: int, tile: int) -> int:
    """這種字牌碰成刻子能加幾台（game.scoring 的三元牌、圈風牌、門風牌）；莊家不明時門風不算。"""
    if tile >= 31:
        return 1
    tai = int(tile == state.round_wind)
    if state.known_dealer is not None and tile == state.seat_winds[player]:
        tai += 1
    return tai


def honor_pung_value(state: GameState, player: int, hand, visible) -> float:
    """手上會加台的字牌刻子與對子，換算成期望多幾台。"""
    value = 0.0
    for tile in range(27, 34):
        if hand[tile] < 2:
            continue
        tai = honor_tai(state, player, tile)
        if hand[tile] >= 3:
            value += tai
        else:
            value += tai * PAIR_PUNG_CHANCE * max(0, 4 - visible[tile])
    return value


def discard_scores(state: GameState, player: int, rng: Random,
                   calibration: Calibration | None = None, samples: int = 120,
                   risk_weight: float = RISK_WEIGHT,
                   undeclared_share: float = UNDECLARED_TENPAI_SHARE,
                   honor_share: float = HONOR_TAI_SHARE) -> dict[int, float]:
    calibration = calibration or default_calibration()
    me = state.players[player]
    candidates = allowed_discards(me.hand, state.forbidden_discards if state.claimed else ())
    visible = visible_tile_counts(state, player)
    gold = set(state.gold_tiles)
    danger = estimate_danger(state, player, candidates, rng, calibration, samples,
                             undeclared_share)
    scores = {}
    for tile in candidates:
        after = list(me.hand)
        after[tile] -= 1
        outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(after, me.open_melds))
        level = shanten(after, me.open_melds)
        value = calibration.offense_value(level, outs, state.drawable)
        if tile in gold and gold_in_use(me.hand, tile):
            # 用得上的金牌留著，胡牌時多 1 台：打掉它約少掉進攻收益的 1 /（底 + 平均台數）
            value *= 1 - GOLD_VALUE_SHARE
        if level == 0 and gold.intersection(effective_tiles(after, me.open_melds)):
            value *= 1 + GOLD_WAIT_SHARE  # 聽金牌：胡那張再多 3 台
        value *= 1 + honor_share * honor_pung_value(state, player, after, visible)
        # 打出金牌被胡對方多 3 台，已算在 danger.expected_loss 裡
        scores[tile] = (value - risk_weight * danger.expected_loss[tile]
                        + OUTS_TIE_BREAK * (outs - 100 * level)
                        - KEEP_TIE_BREAK * keep_value(me.hand, tile, gold))
    return scores


def choose_ev_action_for_player(state: GameState, player: int, rng: Random | None = None,
                                calibration: Calibration | None = None, samples: int = 120,
                                risk_weight: float = RISK_WEIGHT,
                                undeclared_share: float = UNDECLARED_TENPAI_SHARE,
                                honor_share: float = HONOR_TAI_SHARE,
                                declare_min_live: int | None = None) -> Action | None:
    rng = rng or Random()
    if state.phase != Phase.DISCARD or state.declared[player] or player != state.current_player:
        return choose_rule_action_for_player(state, player, rng)
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return win
    kong = choose_self_kong(state, player)
    if kong is not None:
        return kong
    scores = discard_scores(state, player, rng, calibration, samples, risk_weight,
                            undeclared_share, honor_share)
    best = max(scores.values())
    tile = rng.choice(sorted(t for t, score in scores.items() if score >= best - 1e-9))
    declare = Action(ActionType.DECLARE, tile=tile)
    if declare in actions and declare_is_worth(state, player, tile, declare_min_live):
        return declare  # 與規則式 AI 相同：聽的牌夠多才宣告
    return Action(ActionType.DISCARD, tile=tile)
