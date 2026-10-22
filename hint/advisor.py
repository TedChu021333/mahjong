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

from agent.rule_agent import (LIVE_SWAP_SAMPLES, choose_rule_action_for_player, choose_swap_tiles,
                              kong_is_worth)
from game.rules import Action, ActionType, GameState, Meld, Phase, legal_actions, seat_winds_for_dealer
from game.tiles import NUM_TILE_TYPES, tile_name
from perception.config import CLAIM_TEMPLATE_BOX, GOLD_MIN_SCORE, GOLD_TEMPLATE_BOX, TEMPLATE_DIR
from perception.config import MAX_HAND_TILES
from perception.hand_reader import (
    TileClassifier,
    _cell_box,
    drawn_tile_box,
    fill_covered_tops,
    lower_part_classifier,
    read_hand,
)
from perception.state_parser import ObservedPlayer, ObservedTable, parse_observed_table
from perception.capture import read_image
from perception.config import DECLARE_TEMPLATE, HAND_TILE_TOP
from perception.table_reader import (
    ClaimTile,
    read_buttons,
    read_chow_panels,
    read_claim_tile,
    load_streak_templates,
    read_dealer,
    read_dealer_streak,
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
    dealer: str | None = field(default=None, compare=False)
    """莊家座位（me/left/top/right），認不出來為 None。"""
    dealer_streak: int | None = field(default=0, compare=False)
    """連莊數（「莊」右下角的小數字）；有數字但還沒有它的模板時為 None。"""
    declared: frozenset[str] = field(default=frozenset(), compare=False)
    """旁邊有「聽」標記的對手（left/top/right）。"""
    discards: tuple[tuple[int, ...], ...] = field(default=(), compare=False)
    """各家打過的牌（座位同 SOURCE_SEAT：0 自己、1 下家、2 對家、3 上家），由牌河追蹤補上；
    不參與比較，牌河變動不會讓同一個畫面重新給建議。"""
    exposed: tuple[tuple[int, ...], ...] = field(default=(), compare=False)
    """各家副露裡已知、又不在牌河的牌（座位同 discards）：自己吃碰用掉的手牌、推算出別家碰的
    那種牌另外兩張。用來扣掉看得到的牌（例如聽的牌其實已經被碰走）。"""
    my_pungs: tuple[int, ...] | None = field(default=None, compare=False)
    """這局自己碰過的牌（RiverWatch.my_pungs）；None＝沒有追蹤（離線測試）。"""
    declared_safe: tuple[tuple[int, ...], ...] = field(default=(), compare=False)
    """各家宣告後別家打出、他沒胡的牌（座位同 discards；RiverWatch.declared_safe）：一定不是他要胡的。"""
    opponent_melds: tuple[tuple[Meld, ...], ...] = field(default=(), compare=False)
    """畫面上讀到的對手副露（座位同 discards；RiverWatch.read_melds，10/22 起）：內容已知，對手模型會算進他的台數。"""
    my_discards: int = field(default=0, compare=False)
    """這局自己已打出幾張（提示迴圈由手牌變化數出）。牌河追蹤只記到約八成的出牌（10/5 分析 1436 局），
    每位對手打過的張數至少算這麼多，聽牌率與牌牆剩餘才不會低估。"""

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
        self.lower = lower_part_classifier(template_dir)
        self.claim = TileClassifier.from_directory(template_dir, template_box=CLAIM_TEMPLATE_BOX)
        self.gold = TileClassifier.from_directory(template_dir, template_box=GOLD_TEMPLATE_BOX,
                                                  min_score=GOLD_MIN_SCORE)
        self.declare = None
        if DECLARE_TEMPLATE.exists():
            self.declare = cv2.cvtColor(read_image(DECLARE_TEMPLATE), cv2.COLOR_BGR2GRAY)
        self.streak = load_streak_templates()


def observe(image: np.ndarray, readers: Readers) -> Observation | None:
    """辨識一幀；手牌不完整或張數不合理（動畫中）時回傳 None。"""
    reading = read_hand(image, readers.tiles)
    buttons = read_buttons(image)
    if reading.tiles and not reading.complete and buttons is not None:
        # 「上家出牌，不可明槓。」蓋住手牌上緣：認不出的牌改用下半部再認
        reading = fill_covered_tops(image, reading, readers.lower)
    if not reading.tiles or not reading.complete or len(reading.tiles) % 3 == 0:
        return None
    if not _consistent_layout(reading.boxes):
        return None
    gold = tuple(t for t in read_gold_tiles(image, readers.gold) if t is not None)
    return Observation(reading.known_tiles(), buttons,
                       read_claim_tile(image, readers.claim), gold, reading.boxes,
                       read_swap_prompt(image),
                       tuple(box[1] != HAND_TILE_TOP for box in reading.boxes),
                       read_chow_panels(image),
                       declared=read_declared(image, readers.declare),
                       dealer=read_dealer(image),
                       dealer_streak=read_dealer_streak(image, readers.streak))


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


def _known_melds(observation: Observation, rivers: list[list[int]]) -> list[tuple[Meld, ...]]:
    """讀到的對手副露；加上手牌、牌河、前面的副露會超過 4 張的那組丟掉（讀錯）。來源座位不明，記成他的上家
    （Meld.from_player 不能是 None，那是暗槓）。"""
    seen = [0] * NUM_TILE_TYPES
    for tile in observation.hand:
        seen[tile] += 1
    for tiles in rivers:
        for tile in tiles:
            seen[tile] += 1
    result: list[tuple[Meld, ...]] = [()] * 4
    for seat in range(1, 4):
        read = observation.opponent_melds[seat] if seat < len(observation.opponent_melds) else ()
        kept = []
        for meld in read:
            need = [0] * NUM_TILE_TYPES
            for tile in meld.tiles:
                need[tile] += 1
            if all(seen[tile] + need[tile] <= 4 for tile in set(meld.tiles)):
                for tile in meld.tiles:
                    seen[tile] += 1
                kept.append(Meld(meld.kind, meld.tiles, (seat + 3) % 4))
        result[seat] = tuple(kept[:5])
    return result


def _exposed(observation: Observation, rivers: list[list[int]],
             melds: list[int]) -> list[tuple[int, ...]]:
    """副露裡已知的牌；沒有副露組數可放、或加上手牌與牌河會超過 4 張的丟掉（推算可能有誤）。"""
    seen = [0] * NUM_TILE_TYPES
    for tile in observation.hand:
        seen[tile] += 1
    for tiles in rivers:
        for tile in tiles:
            seen[tile] += 1
    result = []
    for seat in range(4):
        kept = []
        known = observation.exposed[seat] if seat < len(observation.exposed) else ()
        if melds[seat]:
            for tile in known:
                if seen[tile] < 4:
                    seen[tile] += 1
                    kept.append(tile)
        result.append(tuple(kept))
    return result


STREAK_UNKNOWN_GUESS = 3
"""有連莊數字但沒有它的模板：目前收集到 1、2，認不出的多半是 3 以上。"""
START_DRAWABLE = 60
"""開局可摸張數的估計：模擬器補完花後平均 59，莊家第一張不必摸再加 1。"""


def estimate_wall(observation: Observation) -> int:
    """剩餘可摸張數 ≈ 60 − 總出牌數 + 吃碰次數（吃碰的人不摸牌）；補花讀不到，未扣。
    牌河漏記的出牌以「自己打了幾張」補：每家至少算這麼多。"""
    rivers = list(observation.discards) + [()] * (4 - len(observation.discards))
    discards = sum(max(len(tiles), observation.my_discards) for tiles in rivers)
    claims = sum(observation.melds) + observation.meld_count
    return max(0, START_DRAWABLE - discards + claims)


def build_state(observation: Observation, waiting: bool = False) -> GameState | None:
    """waiting：不是自己的回合、也沒有可吃碰的牌時，仍建一個「等別人出牌」的局面（放槍表用）。"""
    claim = observation.claim
    responding = not observation.my_turn and observation.buttons is not None         and claim is not None and claim.tile is not None
    source = SOURCE_SEAT[claim.source] if responding else None
    rivers = _rivers(observation, source, claim.tile if responding else None)
    melds = list(observation.melds) + [0] * (4 - len(observation.melds))
    melds[ME] = observation.meld_count
    exposed = _exposed(observation, rivers, melds)
    known = _known_melds(observation, rivers)
    players = []
    for seat in range(4):
        read = known[seat]
        unknown = max(0, melds[seat] - len(read))
        # 讀到的副露裡已經有的牌，不再當成內容未知那組的已知牌
        left_over = list(exposed[seat])
        for meld in read:
            for tile in meld.tiles:
                if tile in left_over:
                    left_over.remove(tile)
        players.append(ObservedPlayer(discards=tuple(rivers[seat]), melds=read, unknown_melds=unknown,
                                      exposed=tuple(left_over) if unknown else ()))
    players[ME] = ObservedPlayer(hand=observation.hand, unknown_melds=observation.meld_count,
                                 discards=tuple(rivers[ME]), exposed=exposed[ME])
    declared = [False] * 4
    for seat in observation.declared:
        declared[SOURCE_SEAT[seat]] = True
    dealer = None if observation.dealer is None else         ME if observation.dealer == "me" else SOURCE_SEAT[observation.dealer]
    streak = STREAK_UNKNOWN_GUESS if observation.dealer_streak is None \
        else observation.dealer_streak
    common = dict(gold_tiles=observation.gold_tiles, declared=tuple(declared), dealer=dealer,
                  dealer_streak=streak,
                  wall_remaining=estimate_wall(observation),
                  min_discards=observation.my_discards,
                  declared_safe=observation.declared_safe)
    if dealer is not None:
        common["seat_winds"] = seat_winds_for_dealer(dealer)  # 門風跟著莊家轉，莊家是東
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
        if not waiting:
            return None
        table = ObservedTable(players=tuple(players), current_player=(ME + 1) % 4,
                              phase=Phase.DRAW, **common)
        return parse_observed_table(table)
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


POLICIES = ("rule", "ev", "mc", "nn")
_policy = "rule"
_pool = None
_workers = 1
_time_limit = None
_rollouts = None
MOVE_OVERHEAD = 2.0
"""每手限時中要留給辨識、期望值計算、點擊與程序間傳遞的秒數（3 秒場模擬 1 秒、6 秒場 4 秒）。
原本 1.5（3 秒場模擬 1.5 秒），10/3 電腦較忙時出牌平均 2.4 秒、最多 4 秒，點擊來不及，卡住 34 次。"""


def use_move_seconds(seconds: float) -> None:
    """依遊戲每手限時設定蒙地卡羅的模擬時間與次數上限（次數上限隨模擬時間等比例調整）。"""
    from agent.mc_agent import LIVE_ROLLOUTS, TIME_LIMIT

    global _time_limit, _rollouts
    _time_limit = max(0.5, seconds - MOVE_OVERHEAD)
    _rollouts = round(LIVE_ROLLOUTS * _time_limit / TIME_LIMIT)


def use_worker_pool(pool, workers: int) -> None:
    """蒙地卡羅 AI 用多個子程序平行模擬（由提示模式的主程式建立，必須在 __main__ 之下）。"""
    global _pool, _workers
    _pool, _workers = pool, workers


_network = None


def use_network(path: str, efficiency_turns: int = 0, guard: bool = True, attack: bool = False) -> None:
    """學習型 AI 的權重（--ai nn；需要 PyTorch）；efficiency_turns 見 agent.nn_agent.efficient_discard，
    guard 見 NetworkPolicy.guard（按聽照規則式、有人宣告時檢查放槍）。"""
    from agent.nn_agent import NetworkPolicy

    global _network
    _network = NetworkPolicy.from_file(path, efficiency_turns=efficiency_turns, guard=guard or attack,
                                       attack=attack)


def last_note() -> str | None:
    """上一個決定被網路的 guard 改掉的原因（沒有就是 None）。"""
    return _network.last_note if _policy == "nn" and _network is not None else None


def use_policy(name: str) -> None:
    """選擇打牌 AI：rule（規則式，預設）、ev（期望值打牌，估計放槍風險）、
    mc（蒙地卡羅，把候選打法模擬到結束比較，每步最多約 2 秒）。"""
    global _policy
    if name not in POLICIES:
        raise ValueError(f"未知的 AI：{name}")
    _policy = name


def _added_kong(observation: Observation, state: GameState) -> Action | None:
    """輪到自己時「槓」亮著、手上卻沒有 4 張一樣的：是加槓（碰過的刻子內容讀不到，規則引擎列不出
    這個選項），划算就槓。能加槓的只有自己碰過的牌（observation.my_pungs）：10/19 18:23 碰了七條、
    手上有六七八條，摸進中時「槓」亮著，原本一律當成摸進的中要加槓，按下去遊戲卻是把六七八裡的七條
    加槓上去，拆了順子。沒有追蹤碰過哪些牌時才退回「摸進的牌」。"""
    buttons = observation.buttons or frozenset()
    if not observation.my_turn or observation.claimed or "kong" not in buttons:
        return None
    me = state.players[ME]
    if any(count >= 4 for count in me.hand):
        return None  # 暗槓由 AI 自己判斷
    if observation.my_pungs is None:
        candidates = [observation.hand[-1]]
    else:
        candidates = [tile for tile in dict.fromkeys(observation.my_pungs) if me.hand[tile]]
    for tile in candidates:
        if kong_is_worth(me.hand, me.open_melds, tile):
            return Action(ActionType.KONG, tile=tile, tiles=(tile,) * 4)
    return None


def decide(observation: Observation, rng: Random | None = None) -> Action | None:
    if _network is not None:
        _network.last_note = None  # 換三張、加槓不經過網路，不能沿用上一手的說明
    if observation.swap_prompt is not None:
        counts = [0] * NUM_TILE_TYPES
        for tile in observation.hand:
            counts[tile] += 1
        # 會加台的字牌：中發白、圈風（東風圈的東）、認得出莊家時自己的門風
        valued = [27, 31, 32, 33]
        if observation.dealer is not None:
            dealer = ME if observation.dealer == "me" else SOURCE_SEAT[observation.dealer]
            valued.append(seat_winds_for_dealer(dealer)[ME])
        return Action(ActionType.SWAP, tiles=choose_swap_tiles(counts,
                                                               gold_tiles=observation.gold_tiles,
                                                               valued_honors=valued,
                                                               samples=LIVE_SWAP_SAMPLES))
    state = build_state(observation)
    if state is None:
        return None
    added_kong = _added_kong(observation, state)
    if added_kong is not None:
        return added_kong
    if _policy == "mc":
        from agent.mc_agent import LIVE_ROLLOUTS, TIME_LIMIT, choose_mc_action_for_player

        return choose_mc_action_for_player(state, ME, rng or Random(0),
                                           rollouts=_rollouts or LIVE_ROLLOUTS,
                                           time_limit=_time_limit or TIME_LIMIT,
                                           pool=_pool, workers=_workers)
    if _policy == "nn":
        if _network is None:
            raise RuntimeError("--ai nn 需要先 use_network(權重檔)")
        return _network(state, ME, rng or Random(0))
    if _policy == "ev":
        from agent.ev_agent import choose_ev_action_for_player

        return choose_ev_action_for_player(state, ME, rng or Random(0))
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


DANGER_ROWS = 5
DANGER_MIN_VALID = 100
"""放槍表：每位宣告者湊這麼多副有效的聽牌手牌（見 opponent_model.estimate_danger），約 0.1～0.3 秒。"""
SEAT_NAMES = {1: "下家", 2: "對家", 3: "上家"}


def danger_table(observation: Observation, rows: int = DANGER_ROWS) -> str | None:
    """手上每種牌打出去的放槍機率，最高的幾張（浮動小視窗的放槍表，10/20 使用者要一直亮著）。
    不是自己的回合時算的是現在這 16 張，給下一手參考。每行：牌、放槍機率、最可能放給誰（宣告者加 ★）。
    讀不出局面時回傳 None。"""
    from agent.opponent_model import estimate_danger

    state = build_state(observation, waiting=True)
    if state is None:
        return None
    tiles = sorted(set(observation.hand))
    danger = estimate_danger(state, ME, tiles, Random(0), samples=40,
                             min_valid=DANGER_MIN_VALID)
    title = "放槍機率（手上最高 5 張）" if observation.my_turn else "放槍機率（下一手參考）"
    lines = [title]
    for tile in sorted(tiles, key=lambda t: -danger.probability[t])[:rows]:
        per = {seat: risks[tile] for seat, risks in (danger.by_opponent or {}).items()}
        seat = max(per, key=per.get) if per and max(per.values()) > 0 else None
        who = "" if seat is None else \
            f"  {SEAT_NAMES[seat]}{'★' if state.declared[seat] else ''}"
        lines.append(f"{tile_name(tile):3}{danger.probability[tile]:6.1%}{who}")
    return "\n".join(lines)
