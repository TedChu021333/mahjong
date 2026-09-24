"""把畫面辨識結果轉成牌局狀態，交給規則式 AI，產生給人看的建議。

座位以自己為 0，逆時針（下家）1、對家 2、上家 3。只在自己要做決定時給建議：
- 輪到自己打牌（暗手 17 - 3n 張）：建議打哪張、要不要聽、能不能自摸。
- 別家打牌後按鈕列出現：建議胡、碰、槓、吃或過。
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType, GameState, Phase
from game.tiles import tile_name
from perception.config import GOLD_TEMPLATE_BOX, TEMPLATE_DIR
from perception.config import MAX_HAND_TILES
from perception.hand_reader import TileClassifier, _cell_box, drawn_tile_box, read_hand
from perception.state_parser import ObservedPlayer, ObservedTable, parse_observed_table
from perception.table_reader import ClaimTile, read_buttons, read_claim_tile, read_gold_tiles

ME = 0
SOURCE_SEAT = {"left": 3, "right": 1}


@dataclass(frozen=True)
class Observation:
    hand: tuple[int, ...]
    buttons: frozenset[str] | None
    claim: ClaimTile | None
    gold_tiles: tuple[int, ...]

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
        self.gold = TileClassifier.from_directory(template_dir, template_box=GOLD_TEMPLATE_BOX)


def observe(image: np.ndarray, readers: Readers) -> Observation | None:
    """辨識一幀；手牌不完整或張數不合理（動畫中）時回傳 None。"""
    reading = read_hand(image, readers.tiles)
    if not reading.tiles or not reading.complete or len(reading.tiles) % 3 == 0:
        return None
    if not _consistent_layout(reading.boxes):
        return None
    gold = tuple(t for t in read_gold_tiles(image, readers.gold) if t is not None)
    return Observation(reading.known_tiles(), read_buttons(image),
                       read_claim_tile(image, readers.tiles), gold)


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


def build_state(observation: Observation) -> GameState | None:
    players = [ObservedPlayer() for _ in range(4)]
    players[ME] = ObservedPlayer(hand=observation.hand, unknown_melds=observation.meld_count)
    common = dict(gold_tiles=observation.gold_tiles)
    if observation.my_turn:
        table = ObservedTable(players=tuple(players), current_player=ME, phase=Phase.DISCARD,
                              last_drawn=observation.hand[-1], **common)
        return parse_observed_table(table)
    claim = observation.claim
    if observation.buttons is None or claim is None or claim.tile is None:
        return None
    source = SOURCE_SEAT[claim.source]
    players[source] = ObservedPlayer(discards=(claim.tile,))
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
    return action.kind.value


def advise(observation: Observation, rng: Random | None = None) -> str | None:
    state = build_state(observation)
    if state is None:
        return None
    action = choose_rule_action_for_player(state, ME, rng or Random(0))
    return describe(action) if action is not None else None
