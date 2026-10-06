"""學習型 AI 的動作空間：每個決定對應到固定編號，網路輸出 ACTION_COUNT 個分數。

輪到自己（打牌階段）：打出某張、打出並宣告聽牌、槓某張、自摸。
別人打牌後（回應階段）：過、胡、碰、明槓、三種吃法（被吃的牌在順子的低／中／高位置）。
摸牌與換三張不交給網路（摸牌只有一種選擇；換三張先沿用規則式 AI）。
"""
from __future__ import annotations

import numpy as np

from game.rules import Action, ActionType
from game.tiles import NUM_TILE_TYPES

DISCARD = 0
DECLARE = DISCARD + NUM_TILE_TYPES
KONG = DECLARE + NUM_TILE_TYPES
PUNG = KONG + NUM_TILE_TYPES
CHOW_LOW = PUNG + 1
"""被吃的牌是順子最小的那張（例如吃三萬組成三四五）。"""
CHOW_MID = CHOW_LOW + 1
CHOW_HIGH = CHOW_LOW + 2
PASS = CHOW_HIGH + 1
WIN = PASS + 1
ACTION_COUNT = WIN + 1


def action_index(action: Action) -> int:
    kind = action.kind
    if kind == ActionType.DISCARD:
        return DISCARD + action.tile
    if kind == ActionType.DECLARE:
        return DECLARE + action.tile
    if kind == ActionType.KONG:
        return KONG + action.tile
    if kind == ActionType.PUNG:
        return PUNG
    if kind == ActionType.CHOW:
        return CHOW_LOW + sorted(action.tiles).index(action.tile)
    if kind == ActionType.PASS:
        return PASS
    if kind == ActionType.WIN:
        return WIN
    raise ValueError(f"網路不處理這種動作：{kind}")


def handled_by_network(action: Action) -> bool:
    return action.kind not in (ActionType.DRAW, ActionType.SWAP)


def legal_mask(actions: list[Action]) -> tuple[np.ndarray, dict[int, Action]]:
    """合法動作的遮罩，以及編號 → 動作。同一編號只會對應一個動作
    （同一張牌的暗槓與加槓不會同時成立；吃法以被吃的牌位置區分）。"""
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    lookup: dict[int, Action] = {}
    for action in actions:
        if not handled_by_network(action):
            continue
        index = action_index(action)
        mask[index] = True
        lookup[index] = action
    return mask, lookup
