"""分析實戰的每一次放槍：AI 當時估的危險度、有沒有更安全的牌、胡牌者聽什麼牌型。

資料：提示模式的記錄檔 result/log_<日期>.txt（每一步的手牌、金牌、各家打出的牌）與小結算截圖
（四家手牌、胡的那張、莊家、胡牌者有沒有宣告，見 eval.settlement）。每局從「換三張」開始，
到結算截圖為止；流局沒有結算截圖，那局的牌都算沒放槍。

還原局面時：各家牌河取那一步之前記錄到的出牌（含被吃碰的），對手副露數取該局結束時的推算值，
莊家取結算畫面，宣告聽牌只知道胡牌者（結算台數有「聽」）。所以這裡的危險度不完全等於實戰當下
AI 看到的，只用來找系統性的偏差。

    python -m eval.deal_in                          # 全部日期
    python -m eval.deal_in --dates 20261004 20261005 --csv result/放槍分析.csv
    python -m eval.deal_in --calibrate 3000          # 另外抽 3000 手沒放槍的出牌，檢查危險度校準
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random

import cv2

from agent.opponent_model import default_calibration, estimate_danger
from agent.rule_agent import visible_tile_counts
from agent.shanten import effective_tiles, shanten
from eval.settlement import Settlement, SettlementReader, is_dealer
from eval.stats import RESULT_DIR, ROW_TOPS, classify, load_digits, load_names, read_row, wind_row
from game.rules import allowed_discards
from game.tiles import NUM_TILE_TYPES, is_suited, tile_name
from game.win import decompose, winning_tiles
from hint.advisor import ME, RIVER_SEAT, Observation, build_state
from perception.capture import read_image

LINE = re.compile(r"^\d\d:\d\d:\d\d \[(\d\d:\d\d:\d\d)\] (.*?)(?:　（(.*)）)?$")
DISCARD = re.compile(r"^打 (\S+?)(，並按「聽」)?(?:（.*）)?$")
"""10/17 起網路的護欄改掉決定時，後面附「（有人宣告，…）」「（不按聽…）」等說明。"""
RIVER = re.compile(r"^牌河 (me|right|top|left) (discard|claimed|restored) (\S+)$")
SETTLEMENT = re.compile(r"結算截圖存到 result[\\/](\d{8}_\d{6})\.png")
MELD_SUMMARY = re.compile(r"(me|right|top|left) [^；（]*（副露 (\d)）")
SEAT_NAMES = {1: "下家", 2: "對家", 3: "上家"}
_NAMES = {tile_name(tile): tile for tile in range(42)}


def parse_tiles(text: str) -> tuple[int, ...]:
    """「一萬五萬東」→ 牌編碼；數牌兩個字、字牌與花牌一個字。"""
    tiles, index = [], 0
    while index < len(text):
        for size in (2, 1):
            name = text[index:index + size]
            if name in _NAMES:
                tiles.append(_NAMES[name])
                index += size
                break
        else:
            raise ValueError(f"認不出牌名：{text[index:]}")
    return tuple(tiles)


DECLARED_SEATS = {"下家": 1, "對家": 2, "上家": 3}


def parse_detail(detail: str | None) -> tuple[tuple[int, ...], tuple[int, ...], frozenset[int] | None]:
    """記錄行括號內的（手牌, 金牌, AI 當下認到的宣告者）；沒寫宣告時為 None。"""
    hand, gold, declared = (), (), None
    for part in (detail or "").split("，"):
        if part.startswith("手牌 "):
            hand = parse_tiles(part[3:])
        elif part.startswith("金牌 "):
            gold = parse_tiles(part[3:])
        elif part.startswith("宣告 "):
            declared = frozenset(DECLARED_SEATS[name] for name in part[3:].split("、"))
    return hand, gold, declared


@dataclass(frozen=True)
class Decision:
    clock: str
    tile: int
    declare: bool
    hand: tuple[int, ...]
    gold: tuple[int, ...]
    rivers: tuple[tuple[int, ...], ...]
    """當時各家牌河（advisor 座位：0 自己、1 下家、2 對家、3 上家）。"""
    declared: frozenset[int] | None = None
    """AI 當下認到的宣告者（記錄有寫時）。"""


@dataclass
class Game:
    date: str
    decisions: list[Decision] = field(default_factory=list)
    stamp: str | None = None
    melds: tuple[int, int, int, int] = (0, 0, 0, 0)
    """該局結束時推算的各家副露數（自己的欄位不用）。"""


def parse_log(path: Path) -> list[Game]:
    date = path.stem.split("_")[-1]
    games: list[Game] = []
    game: Game | None = None
    rivers: list[list[int]] = [[] for _ in range(4)]
    ended = False
    """上一局已結束（本局牌河摘要或按了結算按鈕），下一個出牌事件開始新的一局。"""
    logs_declared = False
    """10/5 晚上起啟動時會寫「AI …」，之後每一步都記認到的宣告者；以前的記錄分不出「沒人宣告」與「沒記錄」。"""
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        clock, text, detail = match.groups()
        if "啟動，AI " in text:
            logs_declared = True
            continue
        if text.startswith("換三張："):
            game = Game(date)
            games.append(game)
            rivers = [[] for _ in range(4)]
            continue
        if text.startswith("按「"):
            ended = True  # 按結算、流局的「再贏一局」：之後是下一局
            continue
        river = RIVER.match(text)
        discard = DISCARD.match(text)
        if (river or discard) and (game is None or ended):
            # 沒有換三張的場（10/15 起）：上一局結束後第一個出牌事件就是新的一局
            game, ended = Game(date), False
            games.append(game)
            rivers = [[] for _ in range(4)]
        if game is None:
            continue
        if river:
            seat, kind, name = river.groups()
            # 自己的牌河用自己的決定（牌河辨識會讀錯：10/18 03:24 打的八萬記成二筒）
            if kind == "discard" and name in _NAMES and seat != "me":
                rivers[RIVER_SEAT[seat]].append(_NAMES[name])
            continue
        if discard and discard.group(1) in _NAMES:
            hand, gold, declared = parse_detail(detail)
            if declared is None and logs_declared:
                declared = frozenset()
            if hand and len(hand) % 3 == 2:
                game.decisions.append(Decision(clock, _NAMES[discard.group(1)],
                                               bool(discard.group(2)), hand, gold,
                                               tuple(tuple(r) for r in rivers), declared))
                rivers[ME].append(_NAMES[discard.group(1)])
            continue
        settled = SETTLEMENT.search(text)
        if settled and game.stamp is None:
            game.stamp = settled.group(1)
            continue
        if text.startswith("本局牌河："):
            melds = [0] * 4
            for seat, count in MELD_SUMMARY.findall(text):
                melds[RIVER_SEAT[seat]] = int(count)
            game.melds = tuple(melds)
            ended = True
    return games


@dataclass(frozen=True)
class Outcome:
    result: str
    """eval.stats 的分類：放槍、胡牌、自摸、被自摸、沒輸贏、未知。"""
    winner: int | None
    """胡牌者（advisor 座位）；自己胡或認不出時為 None。"""
    dealer: int | None
    settlement: Settlement | None
    """只有放槍時才讀。"""
    winner_row: int | None


class OutcomeReader:
    def __init__(self) -> None:
        self.digits = load_digits()
        self.my_name = load_names()
        self.settlement = SettlementReader.load()

    def read(self, path: Path) -> Outcome | None:
        image = read_image(path)
        rows = [read_row(image, top, self.digits, self.my_name) for top in ROW_TOPS]
        mine = [index for index, row in enumerate(rows) if row.is_me]
        if len(mine) != 1:
            row = wind_row(image)
            mine = [] if row is None else [row]
        if not mine:
            return None
        _, result = classify([replace(row, is_me=index == mine[0])
                              for index, row in enumerate(rows)])
        dealers = [index for index, top in enumerate(ROW_TOPS) if is_dealer(image, top)]
        settlement, winner_row = None, None
        if result == "放槍":  # 四家手牌只有放槍時要用（讀一張約 0.7 秒）
            settlement = self.settlement.read(image)
            winners = [index for index, row in enumerate(settlement.rows)
                       if row.win_tile is not None]
            winner_row = winners[0] if len(winners) == 1 else None

        def seat(row: int | None) -> int | None:
            return None if row is None else (row - mine[0]) % 4

        return Outcome(result, seat(winner_row), seat(dealers[0]) if len(dealers) == 1 else None,
                       settlement, winner_row)


def wait_shapes(hand: list[int], win: int) -> set[str]:
    """胡牌者 16 張加上胡的那張，所有拆法中胡的那張可能扮演的聽牌型態。"""
    counts = [0] * NUM_TILE_TYPES
    for tile in hand:
        counts[tile] += 1
    counts[win] += 1
    shapes = set()
    for pair, melds in decompose(counts):
        if pair == win:
            shapes.add("單吊")
        for kind, start in melds:
            if kind == "pung" and start == win:
                shapes.add("雙碰")
            elif kind == "chow" and start <= win <= start + 2:
                position, rank = win - start, start % 9
                if position == 1:
                    shapes.add("嵌張")
                elif position == 0:
                    shapes.add("邊張" if rank == 6 else "兩面")
                else:
                    shapes.add("邊張" if rank == 0 else "兩面")
    return shapes


MAX_HIDDEN = 4
"""結算畫面的獎勵卡片常蓋住中間兩列左邊 3～4 張；最多補這麼多張。"""


def _completions(counts: list[int], jokers: int, need_pair: bool):
    """把 counts 拆成（一對眼 +）面子，缺的牌用 jokers 張萬用牌補，產生每種補法補上的牌；
    最後只剩萬用牌時，它們自成幾組、不影響胡的那張，以 None 代表。"""
    first = next((tile for tile, count in enumerate(counts) if count), None)
    if first is None:
        if jokers % 3 == (2 if need_pair else 0):
            yield [None] * jokers
        return
    groups = [(first,) * 3]
    if need_pair:
        groups.append((first,) * 2)
    if is_suited(first):
        suit = first // 9 * 9
        groups += [tuple(range(start, start + 3)) for start in (first - 2, first - 1, first)
                   if start >= suit and start - suit <= 6]
    for group in groups:
        used, added = [], []
        for tile, number in Counter(group).items():
            real = min(number, counts[tile])  # 比 first 小的牌 counts 都是 0，只能用萬用牌
            used.append((tile, real))
            added += [tile] * (number - real)
        if len(added) > jokers:
            continue
        for tile, real in used:
            counts[tile] -= real
        pair_left = need_pair and len(group) != 2
        for rest in _completions(counts, jokers - len(added), pair_left):
            yield added + rest
        for tile, real in used:
            counts[tile] += real


def fill_hidden(tiles: tuple[int | None, ...], win: int) -> list[list[int]]:
    """認不出的牌位補上可能的牌，回傳能和胡的那張組成胡牌的 16 張手牌。"""
    known = [tile for tile in tiles if tile is not None]
    hidden = len(tiles) - len(known)
    if hidden > MAX_HIDDEN:
        return []
    counts = [0] * NUM_TILE_TYPES
    for tile in known + [win]:
        counts[tile] += 1
    hands, seen = [], set()
    for added in _completions(counts, hidden, True):
        real = [tile for tile in added if tile is not None]
        total = Counter(known + [win] + real)
        if max(total.values()) > 4 or tuple(sorted(real)) in seen:
            continue
        seen.add(tuple(sorted(real)))
        # 自成一組的補牌換成沒出現過的字牌（刻子，剩兩張時為對子），不影響胡的那張的型態
        spare = [tile for tile in range(27, NUM_TILE_TYPES) if not total[tile]]
        free = len(added) - len(real)
        filler = []
        if free % 3 == 2:
            filler += [spare.pop()] * 2
        while len(filler) < free:
            filler += [spare.pop()] * 3
        hands.append(known + real + filler)
    return hands


SHAPE_ORDER = ("兩面", "雙碰", "嵌張", "邊張", "單吊")


def main_shape(shapes: set[str]) -> str:
    return next((shape for shape in SHAPE_ORDER if shape in shapes), "?")


def is_suji(tile: int, river: tuple[int, ...]) -> bool:
    if not is_suited(tile):
        return False
    rank = tile % 9
    partners = [tile - 3] if rank >= 3 else []
    if rank <= 5:
        partners.append(tile + 3)
    return any(partner in river for partner in partners)


def declared_safe(decisions: list[Decision], index: int) -> tuple[tuple[int, ...], ...]:
    """第 index 步時，各宣告者從第一次被認到宣告起、別家打出的牌（與 RiverWatch.declared_safe 相同）。"""
    result: list[tuple[int, ...]] = [()] * 4
    current = decisions[index]
    for seat in current.declared or ():
        first = next(d for d in decisions[:index + 1] if d.declared and seat in d.declared)
        result[seat] = tuple(tile for other in range(4) if other != seat
                             for tile in current.rivers[other][len(first.rivers[other]):])
    return tuple(result)


def to_state(decision: Decision, melds: tuple[int, ...], dealer: int | None,
             declared: frozenset[int] = frozenset(), my_discards: int = 0,
             safe: tuple[tuple[int, ...], ...] = ()):
    """my_discards：這局之前記錄到幾次自己出牌（與提示迴圈的 RiverWatch.my_discards 相同用途）；
    safe：declared_safe 的結果。"""
    names = {value: key for key, value in RIVER_SEAT.items()}
    observation = Observation(
        decision.hand, None, None, decision.gold, discards=decision.rivers,
        melds=melds, dealer=None if dealer is None else names[dealer],
        declared=frozenset(names[seat] for seat in declared), my_discards=my_discards,
        declared_safe=safe)
    try:
        return build_state(observation)
    except ValueError:
        return None  # 牌河誤讀或副露推算超過上限，局面不合法


@dataclass
class Analysis:
    date: str
    stamp: str
    clock: str
    turn: int
    tile: int
    win_tile: int | None
    winner: int | None
    declared: bool
    ai_knew: bool | None
    """出牌當下 AI 有沒有認到胡牌者宣告（記錄沒寫時 None）。"""
    winner_melds: int
    winner_river: int
    genbutsu: bool
    suji: bool
    shape: str
    waits: str
    shanten_after: int
    p_winner: float
    p_total: float
    rank: int
    candidates: int
    safest: int
    safest_p: float
    offense_loss: float
    """改打最安全的牌，進攻期望收益少多少台。"""


def offense_values(state) -> dict[int, float]:
    calibration = default_calibration()
    me = state.players[ME]
    visible = visible_tile_counts(state, ME)
    values = {}
    for tile in allowed_discards(me.hand, ()):
        after = list(me.hand)
        after[tile] -= 1
        outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(after, me.open_melds))
        values[tile] = calibration.offense_value(shanten(after, me.open_melds), outs,
                                                 state.drawable)
    return values


def analyse(game: Game, outcome: Outcome, rng: Random, samples: int) -> Analysis | None:
    decision = game.decisions[-1]
    winner = outcome.winner
    declared = outcome.settlement.winner_declared
    safe = declared_safe(game.decisions, len(game.decisions) - 1)
    state = to_state(decision, game.melds, outcome.dealer,
                     frozenset({winner}) if declared and winner else frozenset(),
                     len(game.decisions) - 1,
                     tuple(safe[seat] if seat == winner else () for seat in range(4)))
    if state is None:
        return None
    me = state.players[ME]
    candidates = allowed_discards(me.hand, ())
    danger = estimate_danger(state, ME, candidates, rng, samples=samples, min_valid=150)
    offense = offense_values(state)
    winner_row = outcome.settlement.rows[outcome.winner_row]
    shape, waits = "?", "?"
    if len(winner_row.tiles) == 16 and winner_row.win_tile is not None:
        # 被蓋住的牌有多種補法時，只採用所有補法都一致的結果
        shapes, wait_sets = set(), set()
        for hand in fill_hidden(winner_row.tiles, winner_row.win_tile):
            counts = [0] * NUM_TILE_TYPES
            for tile in hand:
                counts[tile] += 1
            wait_sets.add(tuple(winning_tiles(counts, 0)))
            shapes.add(main_shape(wait_shapes(hand, winner_row.win_tile)))
        if len(shapes) == 1:
            shape = shapes.pop()
        if len(wait_sets) == 1:
            waits = "".join(tile_name(t) for t in wait_sets.pop())
    by_winner = danger.by_opponent.get(winner, {}) if winner else {}
    order = sorted(candidates, key=lambda t: danger.probability[t])
    safest = order[0]
    after = list(me.hand)
    after[decision.tile] -= 1
    river = decision.rivers[winner] if winner else ()
    return Analysis(
        game.date, game.stamp, decision.clock, len(game.decisions), decision.tile,
        winner_row.win_tile, winner, declared,
        None if decision.declared is None else winner in decision.declared, game.melds[winner] if winner else 0, len(river),
        decision.tile in river, is_suji(decision.tile, river), shape, waits,
        shanten(after, me.open_melds), by_winner.get(decision.tile, 0.0),
        danger.probability.get(decision.tile, 0.0),
        sum(1 for t in candidates if danger.probability[t] < danger.probability[decision.tile] - 1e-9),
        len(candidates), safest, danger.probability[safest],
        offense.get(decision.tile, 0.0) - offense.get(safest, 0.0))


def calibration_rows(games: list[tuple[Game, Outcome | None]], limit: int, rng: Random,
                     samples: int) -> list[tuple[float, int, int, int, int]]:
    """抽樣每一手出牌：(估計放槍給這家的機率, 實際有沒有放槍給他, 他打過幾張, 副露數, 對手座位)。"""
    pool = []
    for game, outcome in games:
        for index, decision in enumerate(game.decisions):
            last = index == len(game.decisions) - 1
            hit = None
            if outcome is not None and outcome.result == "放槍" and last:
                hit = outcome.winner
            pool.append((game, outcome, decision, hit))
    rng.shuffle(pool)
    rows = []
    for game, outcome, decision, hit in pool[:limit]:
        state = to_state(decision, game.melds, outcome.dealer if outcome else None,
                         my_discards=game.decisions.index(decision))
        if state is None:
            continue
        danger = estimate_danger(state, ME, [decision.tile], rng, samples=samples)
        for seat in (1, 2, 3):
            p = danger.by_opponent.get(seat, {}).get(decision.tile, 0.0)
            rows.append((p, int(hit == seat), len(decision.rivers[seat]), game.melds[seat], seat))
    return rows


def _pct(values: list[bool]) -> str:
    return f"{sum(values) / len(values):.0%}" if values else "-"


def report(analyses: list[Analysis], mismatched: int) -> str:
    lines = [f"放槍 {len(analyses)} 次（另有 {mismatched} 次記錄的最後一張與結算胡的那張不同，未列入）", ""]
    by_seat = Counter(SEAT_NAMES.get(a.winner, "?") for a in analyses)
    lines.append("放槍給：" + "、".join(f"{k} {v}" for k, v in by_seat.most_common()))
    lines.append(f"胡牌者有宣告聽牌：{_pct([a.declared for a in analyses])}")
    known = [a for a in analyses if a.declared and a.ai_knew is not None]
    if known:
        lines.append(f"  其中出牌當下 AI 認到他宣告：{_pct([a.ai_knew for a in known])}（{len(known)} 次有記錄）")
    shapes = Counter(a.shape for a in analyses)
    lines.append("胡牌者的聽牌型態：" + "、".join(f"{k} {v}" for k, v in shapes.most_common()))
    lines.append(f"打的是胡牌者打過的牌（現物）：{_pct([a.genbutsu for a in analyses])}；"
                 f"筋牌：{_pct([a.suji for a in analyses])}")
    kinds = Counter("字牌" if a.tile >= 27 else "么九" if a.tile % 9 in (0, 8) else
                    "二八" if a.tile % 9 in (1, 7) else "中張" for a in analyses)
    lines.append("放槍的牌：" + "、".join(f"{k} {v}" for k, v in kinds.most_common()))
    after = Counter(a.shanten_after for a in analyses)
    lines.append("打完後自己的向聽數：" + "、".join(f"{k} 向聽 {v}" for k, v in sorted(after.items())))
    lines.append(f"胡牌者副露數：" + "、".join(
        f"{k} 組 {v}" for k, v in sorted(Counter(a.winner_melds for a in analyses).items())))
    turns = [a.turn for a in analyses]
    lines.append(f"放槍時是自己第幾手：平均 {sum(turns) / len(turns):.1f}" if turns else "")
    lines += ["", "AI 事後重估（同一個對手模型）："]
    p = [a.p_winner for a in analyses]
    lines.append(f"  估計放槍給胡牌者的機率：平均 {sum(p) / len(p):.1%}；" +
                 "、".join(f"<{hi:.0%} {sum(lo <= x < hi for x in p)}"
                           for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01))))
    safer = [a for a in analyses if a.safest_p < a.p_total - 0.03]
    cheap = [a for a in safer if a.offense_loss <= 0.3]
    lines.append(f"  有比這張低 3% 以上的牌：{len(safer)} 次；其中改打它進攻收益少不到 0.3 台：{len(cheap)} 次")
    ranks = [a.rank / max(1, a.candidates - 1) for a in analyses]
    lines.append(f"  打的牌在可打的牌中危險度排名（0 最安全、1 最危險）：平均 {sum(ranks) / len(ranks):.2f}")
    return "\n".join(lines)


def calibration_report(rows) -> str:
    lines = ["", "危險度校準（每手出牌 × 每位對手）：估計的放槍機率 vs 實際"]

    def table(title, key):
        lines.append(title)
        groups: dict = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row)
        for name in sorted(groups):
            group = groups[name]
            predicted = sum(r[0] for r in group)
            actual = sum(r[1] for r in group)
            lines.append(f"  {name}：{len(group)} 筆，估計 {predicted:.1f} 次、實際 {actual} 次"
                         f"（比值 {actual / predicted:.2f}）" if predicted else f"  {name}：{len(group)} 筆")

    buckets = ((0, .01), (.01, .03), (.03, .06), (.06, .1), (.1, 1.01))
    table("依估計機率", lambda r: next(f"{lo:.0%}~{hi:.0%}" for lo, hi in buckets if lo <= r[0] < hi))
    table("依對手打過幾張", lambda r: f"{min(r[2] // 4 * 4, 16):02d}+ 張")
    table("依對手副露數（局末）", lambda r: f"{min(r[3], 3)} 組")
    table("依對手座位", lambda r: SEAT_NAMES[r[4]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析實戰的每一次放槍")
    parser.add_argument("--dates", nargs="*", help="YYYYMMDD，預設全部記錄檔")
    parser.add_argument("--dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--samples", type=int, default=400, help="每位對手抽幾副聽牌手牌")
    parser.add_argument("--calibrate", type=int, default=0, help="抽幾手出牌檢查危險度校準")
    parser.add_argument("--csv", type=Path, help="逐次放槍另存 CSV")
    args = parser.parse_args()
    logs = sorted(args.dir.glob("log_*.txt"))
    if args.dates:
        logs = [path for path in logs if path.stem.split("_")[-1] in args.dates]
    reader = OutcomeReader()
    rng = Random(0)
    games, analyses, mismatched, after_declare = [], [], 0, 0
    started = time.time()
    for path in logs:
        for game in parse_log(path):
            if not game.decisions:
                continue
            outcome = None
            if game.stamp:
                image = args.dir / f"{game.stamp}.png"
                outcome = reader.read(image) if image.exists() else None
                if outcome is None:
                    continue  # 有結算但讀不出自己那列：不知道結果
            games.append((game, outcome))
            if outcome is None or outcome.result != "放槍" or outcome.winner is None:
                continue
            if outcome.settlement.rows[outcome.winner_row].win_tile != game.decisions[-1].tile:
                mismatched += 1
                after_declare += any(d.declare for d in game.decisions)
                continue
            analysis = analyse(game, outcome, rng, args.samples)
            if analysis is not None:
                analyses.append(analysis)
    print(f"{len(games)} 局（{time.time() - started:.0f} 秒）")
    print(report(analyses, mismatched))
    print(f"（不同的 {mismatched} 次中，{after_declare} 次自己已按聽、之後由遊戲代打）")
    if args.calibrate:
        print(calibration_report(calibration_rows(games, args.calibrate, rng, args.samples)))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["截圖", "時間", "第幾手", "放槍的牌", "胡牌者", "宣告", "胡牌者副露",
                             "胡牌者打過", "現物", "筋", "型態", "聽的牌", "打完向聽",
                             "估計放槍給他", "估計放槍合計", "比它安全的牌數", "可打張數",
                             "最安全的牌", "最安全的估計", "改打最安全少的進攻收益"])
            for a in analyses:
                writer.writerow([a.stamp, a.clock, a.turn, tile_name(a.tile),
                                 SEAT_NAMES.get(a.winner, "?"), int(a.declared), a.winner_melds,
                                 a.winner_river, int(a.genbutsu), int(a.suji), a.shape, a.waits,
                                 a.shanten_after, f"{a.p_winner:.3f}", f"{a.p_total:.3f}", a.rank,
                                 a.candidates, tile_name(a.safest), f"{a.safest_p:.3f}",
                                 f"{a.offense_loss:.2f}"])


if __name__ == "__main__":
    main()
