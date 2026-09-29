"""把畫面辨識結果轉成牌局狀態，交給規則式 AI，產生給人看的建議。

座位以自己為 0，逆時針（下家）1、對家 2、上家 3。只在自己要做決定時給建議：
- 輪到自己打牌（暗手 17 - 3n 張）：建議打哪張、要不要聽、能不能自摸。
- 別家打牌後按鈕列出現：建議胡、碰、槓、吃或過。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import cv2
import numpy as np

from agent.rule_agent import choose_rule_action_for_player, choose_swap_tiles
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from game.tiles import NUM_TILE_TYPES, tile_name
from perception.config import CLAIM_TEMPLATE_BOX, GOLD_MIN_SCORE, GOLD_TEMPLATE_BOX, TEMPLATE_DIR
from perception.config import MAX_HAND_TILES
from perception.hand_reader import TileClassifier, _cell_box, drawn_tile_box, read_hand
from perception.state_parser import ObservedPlayer, ObservedTable, parse_observed_table
from perception.capture import read_image
from perception.config import DECLARE_TEMPLATE, HAND_TILE_TOP
from perception.table_reader import (
    ClaimTile,
    read_buttons,
    read_chow_panels,
    read_claim_tile,
    read_declared,
    read_gold_tiles,
    read_swap_prompt,
)

ME = 0
SOURCE_SEAT = {"left": 3, "top": 2, "right": 1}
RIVER_SEAT = {"me": 0, "right": 1, "top": 2, "left": 3}


@dataclass(frozen=True)
class Observation:
    hand: tuple[int, ...]
    buttons: frozenset[str] | None
    claim: ClaimTile | None
    gold_tiles: tuple[int, ...]
    boxes: tuple[tuple[int, int, int, int], ...] = field(default=(), compare=False)
    """每張暗手牌在畫面上的位置（自動操作點擊用）；被選起時會上移，不參與比較。"""
    swap_prompt: bool | None = None
    """換三張選牌中時為「換牌」按鈕是否可按；不在換三張畫面為 None。"""
    raised: tuple[bool, ...] = ()
    """各張暗手牌是否被選起（換三張）。"""
    chow_panels: tuple[tuple[int, int], ...] = ()
    """多種吃法時的選項框中心。"""
    claimed: bool = False
    """剛吃碰完、要打一張（這一手不能胡）。畫面上看不出來（遊戲會把最右邊的牌移到
    第 17 張的位置），由提示迴圈比對吃碰前後的手牌補上。"""
    forbidden: tuple[int, ...] = ()
    """剛吃碰後不能打的牌。"""
    melds: tuple[int, ...] = field(default=(), compare=False)
    """推算的對手副露數（座位同 discards；自己的欄位不用，由手牌張數得知）。"""
    declared: frozenset[str] = field(default=frozenset(), compare=False)
    """旁邊有「聽」標記的對手（left/top/right）。"""
    discards: tuple[tuple[int, ...], ...] = field(default=(), compare=False)
    """各家打過的牌（座位同 SOURCE_SEAT：0 自己、1 下家、2 對家、3 上家），由牌河追蹤補上；
    不參與比較，牌河變動不會讓同一個畫面重新給建議。"""

    @property
    def my_turn(self) -> bool:
        return len(self.hand) % 3 == 2

    @property
    def meld_count(self) -> int:
        return (17 - len(self.hand)) // 3 if self.my_turn else (16 - len(self.hand)) // 3


class Readers:
    """載入一次模板，之後每幀重複使用。"""

    def __init__(self, template_dir=TEMPLATE_DIR) -> None:
        self.tiles = TileClassifier.from_directory(template_dir)
        self.claim = TileClassifier.from_directory(template_dir, template_box=CLAIM_TEMPLATE_BOX)
        self.gold = TileClassifier.from_directory(template_dir, template_box=GOLD_TEMPLATE_BOX,
                                                  min_score=GOLD_MIN_SCORE)
        self.declare = None
        if DECLARE_TEMPLATE.exists():
            self.declare = cv2.cvtColor(read_image(DECLARE_TEMPLATE), cv2.COLOR_BGR2GRAY)


def observe(image: np.ndarray, readers: Readers) -> Observation | None:
    """辨識一幀；手牌不完整或張數不合理（動畫中）時回傳 None。"""
    reading = read_hand(image, readers.tiles)
    if not reading.tiles or not reading.complete or len(reading.tiles) % 3 == 0:
        return None
    if not _consistent_layout(reading.boxes):
        return None
    gold = tuple(t for t in read_gold_tiles(image, readers.gold) if t is not None)
    return Observation(reading.known_tiles(), read_buttons(image),
                       read_claim_tile(image, readers.claim), gold, reading.boxes,
                       read_swap_prompt(image),
                       tuple(box[1] != HAND_TILE_TOP for box in reading.boxes),
                       read_chow_panels(image),
                       declared=read_declared(image, readers.declare))


def _consistent_layout(boxes) -> bool:
    """暗手牌一定從第 3n 格（n 為副露組數）連續排到最右格，輪到自己時第 17 張
    放在獨立位置。動畫（代打的手、放槍字樣）擋住部分手牌時，讀到的張數可能
    剛好合理，只有檢查位置才能排除，否則會給出錯誤建議（例如假的自摸）。"""
    grid_lefts = [_cell_box(index)[0] for index in reversed(range(MAX_HAND_TILES))]
    cells = [grid_lefts.index(box[0]) for box in boxes if box[0] in grid_lefts]
    extra = [box for box in boxes if box[0] == drawn_tile_box()[0]]
    if not cells or len(cells) + len(extra) != len(boxes):
        return False
    return cells[-1] == MAX_HAND_TILES - 1 and cells[0] % 3 == 0


def _rivers(observation: Observation, claim_seat: int | None = None,
            claim_tile: int | None = None) -> list[list[int]]:
    """各家牌河；待回應的那張一定排在打牌者最後。牌河可能誤讀，任一種牌加上自己手牌
    超過 4 張時丟掉較早的，避免規則引擎拒絕整個狀態。"""
    rivers = [list(tiles) for tiles in observation.discards] or [[] for _ in range(4)]
    rivers += [[] for _ in range(4 - len(rivers))]
    if claim_seat is not None and (not rivers[claim_seat] or rivers[claim_seat][-1] != claim_tile):
        rivers[claim_seat].append(claim_tile)
    seen = [0] * NUM_TILE_TYPES
    for tile in observation.hand:
        seen[tile] += 1
    if claim_seat is not None:
        seen[claim_tile] += 1  # 待回應的那張優先保留
    capped = [[] for _ in range(4)]
    for seat, tiles in enumerate(rivers):
        last = len(tiles) - 1
        for index, tile in enumerate(tiles):
            if seat == claim_seat and index == last:
                capped[seat].append(tile)
            elif seen[tile] < 4:
                seen[tile] += 1
                capped[seat].append(tile)
    return capped


def build_state(observation: Observation) -> GameState | None:
    claim = observation.claim
    responding = not observation.my_turn and observation.buttons is not None         and claim is not None and claim.tile is not None
    source = SOURCE_SEAT[claim.source] if responding else None
    rivers = _rivers(observation, source, claim.tile if responding else None)
    melds = list(observation.melds) + [0] * (4 - len(observation.melds))
    players = [ObservedPlayer(discards=tuple(rivers[seat]), unknown_melds=melds[seat])
               for seat in range(4)]
    players[ME] = ObservedPlayer(hand=observation.hand, unknown_melds=observation.meld_count,
                                 discards=tuple(rivers[ME]))
    declared = [False] * 4
    for seat in observation.declared:
        declared[SOURCE_SEAT[seat]] = True
    common = dict(gold_tiles=observation.gold_tiles, declared=tuple(declared))
    if observation.my_turn:
        if observation.claimed:
            table = ObservedTable(players=tuple(players), current_player=ME, phase=Phase.DISCARD,
                                  claimed=True, forbidden_discards=observation.forbidden,
                                  **common)
        else:
            table = ObservedTable(players=tuple(players), current_player=ME, phase=Phase.DISCARD,
                                  last_drawn=observation.hand[-1], **common)
        return parse_observed_table(table)
    if not responding:
        return None
    table = ObservedTable(players=tuple(players), current_player=source, phase=Phase.RESPONSE,
                          last_discard=claim.tile, discard_player=source,
                          response_players=(ME,), **common)
    return parse_observed_table(table)


def describe(action: Action) -> str:
    if action.kind == ActionType.WIN:
        return "胡！" if action.tile is not None else "自摸胡！"
    if action.kind == ActionType.DISCARD:
        return f"打 {tile_name(action.tile)}"
    if action.kind == ActionType.DECLARE:
        return f"打 {tile_name(action.tile)}，並按「聽」"
    if action.kind == ActionType.PUNG:
        return f"碰 {tile_name(action.tile)}"
    if action.kind == ActionType.KONG:
        return f"槓 {tile_name(action.tile)}"
    if action.kind == ActionType.CHOW:
        return "吃，組成 " + "".join(tile_name(t) for t in action.tiles)
    if action.kind == ActionType.PASS:
        return "不要，按「取消」"
    if action.kind == ActionType.SWAP:
        if not action.tiles:
            return "換三張：不換"
        return "換三張：換 " + "、".join(tile_name(t) for t in action.tiles)
    return action.kind.value


def decide(observation: Observation, rng: Random | None = None) -> Action | None:
    if observation.swap_prompt is not None:
        counts = [0] * NUM_TILE_TYPES
        for tile in observation.hand:
            counts[tile] += 1
        return Action(ActionType.SWAP, tiles=choose_swap_tiles(counts))
    state = build_state(observation)
    if state is None:
        return None
    return choose_rule_action_for_player(state, ME, rng or Random(0))


def hand_after_claim(observation: Observation, action: Action) -> tuple[int, ...]:
    """吃碰後暗手牌應有的樣子（排序過），用來認出吃碰完成後的畫面。"""
    hand = list(observation.hand)
    used = list(action.tiles)
    used.remove(action.tile)
    for tile in used:
        hand.remove(tile)
    return tuple(sorted(hand))


def chow_options(observation: Observation) -> list[tuple[int, ...]]:
    """所有吃法，順序與遊戲選項框相同（順子由小到大）。"""
    state = build_state(observation)
    if state is None or state.phase != Phase.RESPONSE:
        return []
    return [action.tiles for action in legal_actions(state, ME) if action.kind == ActionType.CHOW]


def advise(observation: Observation, rng: Random | None = None) -> str | None:
    action = decide(observation, rng)
    return describe(action) if action is not None else None
