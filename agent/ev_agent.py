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
    GOLD_DISCARD_PENALTY,
    choose_rule_action_for_player,
    choose_self_kong,
    visible_tile_counts,
)
from agent.shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, allowed_discards, legal_actions

RISK_WEIGHT = 0.5
"""放槍損失的權重（1 = 純期望值）。arena 各 1500 對、對規則式 AI：0.5 → −0.029±0.430，
1 → −0.138±0.428，2 → −0.289±0.435，皆打平；0.5 最接近，放槍率 19.5% → 17.6%。"""
OUTS_TIE_BREAK = 1e-3
"""表是分組的，同一組內再以精確的有效進張數區分（不改變不同組之間的排序）。"""


def discard_scores(state: GameState, player: int, rng: Random,
                   calibration: Calibration | None = None, samples: int = 120,
                   risk_weight: float = RISK_WEIGHT,
                   undeclared_share: float = UNDECLARED_TENPAI_SHARE) -> dict[int, float]:
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
        if tile in gold:  # 金牌留著胡牌時多台、打出去被胡對方多 3 台
            outs = max(0, outs - GOLD_DISCARD_PENALTY)
        level = shanten(after, me.open_melds)
        value = calibration.offense_value(level, outs, state.drawable)
        scores[tile] = value - risk_weight * danger.expected_loss[tile]             + OUTS_TIE_BREAK * (outs - 100 * level)
    return scores


def choose_ev_action_for_player(state: GameState, player: int, rng: Random | None = None,
                                calibration: Calibration | None = None, samples: int = 120,
                                risk_weight: float = RISK_WEIGHT,
                                undeclared_share: float = UNDECLARED_TENPAI_SHARE) -> Action | None:
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
                            undeclared_share)
    best = max(scores.values())
    tile = rng.choice(sorted(t for t, score in scores.items() if score >= best - 1e-9))
    declare = Action(ActionType.DECLARE, tile=tile)
    if declare in actions:
        return declare  # 與規則式 AI 相同：聽牌就宣告
    return Action(ActionType.DISCARD, tile=tile)
