"""實戰對手（訓練場的遊戲電腦）和模擬器裡的對手（規則式）打法比較：差很多時，模擬器練出來的 AI 不一定適用實戰。

實戰：自動模式的記錄檔 result/log_<日期>.txt 逐局解析（宣告聽牌、各家牌河、自己打了幾張），配上小結算截圖
（四家輸贏金額 → 誰胡、誰放槍、自摸、每家付多少；胡牌者手牌 → 聽牌型態、有沒有「聽」台）。
結算畫面那一列連副露一起列（幾乎都是 16 張），副露數改用記錄檔每局推算的（會少算）；金額有加成倍數、
場館也不只一種，換算不出台數，所以比「對手胡牌時每家付的 ÷ 自己胡牌時每家付的」，同一桌的倍數會消掉。
模擬器：自己 nnguard:rl3_0960、三家規則式，開換三張，算同樣的數字。只比三家對手（自己那家只列參考）。

    python -m eval.opponents --since 20261015 --games 3000 --workers 14
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field, replace
from math import sqrt
from multiprocessing import Pool
from pathlib import Path
from random import Random

from eval.deal_in import LINE, RIVER, SETTLEMENT, fill_hidden, main_shape, wait_shapes
from game.tiles import is_honor, tile_name

SEAT_OF = {"下家": 1, "對家": 2, "上家": 3, "right": 1, "top": 2, "left": 3, "me": 0}
DECLARE = re.compile(r"^(上家|對家|下家)宣告聽牌（第一次認到；牌河 (.*)）$")
COUNT = re.compile(r"(自己|下家|對家|上家) (\d+)")
RIVER_SUMMARY = re.compile(r"(me|right|top|left) ([^；（]*)")
MELD_COUNT = re.compile(r"(me|right|top|left) [^；（]*（副露 (\d)）")


@dataclass
class HandRecord:
    """一局（座位 0 自己、1 下家、2 對家、3 上家）。"""
    winner: int | None = None
    """None：流局或讀不出。"""
    self_draw: bool = False
    discarder: int | None = None
    tai: int | None = None
    """模擬器的台數（實戰換算不出）。"""
    patterns: tuple[str, ...] = ()
    """模擬器胡牌的台數項目（game.scoring.Score.patterns）。"""
    paid: float | None = None
    """每家付多少（自摸取付最少的那家）：實戰是金額，模擬器是台數＋底。"""
    winner_declared: bool | None = None
    melds: tuple[int | None, ...] = (None, None, None, None)
    shape: str | None = None
    declares: dict[int, int] = field(default_factory=dict)
    """宣告聽牌的對手 → 第幾巡宣告。"""
    turns: int | None = None
    first_discards: dict[int, tuple[int, ...]] = field(default_factory=dict)
    drawn: bool = False
    """流局（實戰：沒有結算畫面就換局，也可能是結算沒認到）。"""
    stamp: str | None = None


# ---------- 實戰 ----------

def parse_log(path: Path) -> list[HandRecord]:
    """自動模式的局；結算截圖之後一行是「本局牌河」，之前的宣告與出牌屬於這局。"""
    hands: list[HandRecord] = []
    current = HandRecord()
    my_discards = 0
    stamp = None
    auto = True
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        _, text, _ = match.groups()
        if "模式啟動" in text:
            auto = text.startswith("自動模式")
            current, my_discards, stamp = HandRecord(), 0, None
            continue
        if text.startswith("打 "):
            my_discards += 1
            continue
        river = RIVER.match(text)
        if river and river.group(2) == "discard" and river.group(1) != "me":
            seat = SEAT_OF[river.group(1)]
            from eval.deal_in import _NAMES
            tile = _NAMES.get(river.group(3))
            first = current.first_discards.get(seat, ())
            if tile is not None and len(first) < 2:
                current.first_discards[seat] = first + (tile,)
            continue
        declared = DECLARE.match(text)
        if declared:
            counts = [int(n) for _, n in COUNT.findall(declared.group(2))]
            current.declares.setdefault(SEAT_OF[declared.group(1)], max(counts + [my_discards]))
            continue
        settled = SETTLEMENT.search(text)
        if settled:
            stamp = settled.group(1)
            continue
        if text.startswith("本局牌河：") or text.startswith("新的一局（上一局沒有結算畫面）"):
            if text.startswith("本局牌河："):
                lengths = [len(_tiles(tiles)) for _, tiles in RIVER_SUMMARY.findall(text[5:])]
                current.turns = max(lengths + [my_discards])
                melds = [0, 0, 0, 0]
                for seat, count in MELD_COUNT.findall(text):
                    melds[SEAT_OF[seat]] = int(count)
                current.melds = tuple(melds)
                current.stamp = stamp
                current.drawn = stamp is None
            else:
                current.turns = my_discards
                current.drawn = True
            if auto and (current.stamp or current.drawn) and current.turns:
                hands.append(current)
            current, my_discards, stamp = HandRecord(), 0, None
    return hands


def _tiles(text: str) -> list[int]:
    from eval.deal_in import _NAMES
    found, rest = [], text.strip()
    while rest:
        for size in (2, 1):
            if rest[:size] in _NAMES:
                found.append(_NAMES[rest[:size]])
                rest = rest[size:]
                break
        else:
            rest = rest[1:]
    return found



_READER = None


def read_settlement(path: Path) -> dict | None:
    """一張小結算：四家金額（自己那列）、手牌張數、胡牌者的聽牌型態與「聽」台。"""
    global _READER
    from eval.deal_in import OutcomeReader
    from eval.stats import ROW_TOPS, read_row, wind_row
    from perception.capture import read_image

    if _READER is None:
        _READER = OutcomeReader()
    image = read_image(path)
    rows = [read_row(image, top, _READER.digits, _READER.my_name) for top in ROW_TOPS]
    mine = [index for index, row in enumerate(rows) if row.is_me]
    if len(mine) != 1:
        row = wind_row(image)
        mine = [] if row is None else [row]
    if not mine:
        return None
    me = mine[0]
    amounts = [row.amount or 0 for row in rows]
    settlement = _READER.settlement.read(image)
    out = {"amounts": [amounts[(me + seat) % 4] for seat in range(4)],
           "sizes": [len(settlement.rows[(me + seat) % 4].tiles) for seat in range(4)],
           "declared": settlement.winner_declared, "shape": None}
    winners = [seat for seat in range(4) if settlement.rows[(me + seat) % 4].win_tile is not None]
    if len(winners) == 1:
        row = settlement.rows[(me + winners[0]) % 4]
        shapes = set()
        for hand in fill_hidden(row.tiles, row.win_tile):
            shapes |= wait_shapes(hand, row.win_tile)
        out["shape"] = main_shape(shapes) if shapes else None
    return out


def apply_settlement(hand: HandRecord, read: dict | None) -> HandRecord:
    if read is None:
        return replace(hand, winner=None, drawn=False)
    amounts = read["amounts"]
    winners = [seat for seat in range(4) if amounts[seat] > 0]
    losers = [seat for seat in range(4) if amounts[seat] < 0]
    if len(winners) != 1 or not losers:
        return replace(hand, winner=None)
    self_draw = len(losers) == 3
    paid = min(-amounts[seat] for seat in losers)
    return replace(hand, winner=winners[0], self_draw=self_draw,
                   discarder=None if self_draw else losers[0], paid=paid,
                   winner_declared=read["declared"], shape=read["shape"])


def real_hands(directory: Path, since: str, workers: int, cache: Path | None = None) -> list[HandRecord]:
    """cache 有給時讀過的結算存起來（每張約 0.7 秒），下次只讀新的。"""
    import pickle

    known = pickle.loads(cache.read_bytes()) if cache is not None and cache.exists() else {}
    hands = []
    for path in sorted(directory.glob("log_*.txt")):
        if path.stem.split("_")[-1] >= since:
            hands += parse_log(path)
    settled = [hand for hand in hands if hand.stamp]
    missing = [hand.stamp for hand in settled if hand.stamp not in known]
    with Pool(workers) as pool:
        reads = pool.map(_read_if_exists, [directory / f"{stamp}.png" for stamp in missing], chunksize=8)
    known.update(zip(missing, reads))
    if cache is not None:
        cache.write_bytes(pickle.dumps(known))
    by_stamp = {hand.stamp: apply_settlement(hand, known[hand.stamp]) for hand in settled}
    return [by_stamp.get(hand.stamp, hand) if hand.stamp else hand for hand in hands]


def _read_if_exists(path: Path) -> dict | None:
    return read_settlement(path) if path.exists() else None


# ---------- 模擬器 ----------

def sim_hand(seed: int, me: str = "nnguard:models/rl3_0960.pt", opponents: str = "rule") -> HandRecord:
    from eval.arena import make_policy
    from game.rules import ActionType
    from game.simulator import play_game

    mine, others = make_policy(me), make_policy(opponents)
    discards = [0] * 4
    hand = HandRecord()

    def policy(state, player, rng):
        return (mine if player == 0 else others)(state, player, rng)

    def observer(state, player, action):
        if action.kind in (ActionType.DISCARD, ActionType.DECLARE):
            discards[player] += 1
            first = hand.first_discards.get(player, ())
            if player and len(first) < 2:
                hand.first_discards[player] = first + (action.tile,)
            if action.kind == ActionType.DECLARE and player:
                hand.declares.setdefault(player, discards[player])

    result = play_game(Random(seed), player_policy=policy, dealer=seed % 4, swap=True, observer=observer)
    hand.turns = max(discards)
    hand.melds = tuple(len(p.melds) for p in result.state.players)
    if result.winner is None or result.score is None:
        hand.drawn = True
        return hand
    winner = result.state.players[result.winner]
    hand.winner, hand.self_draw, hand.discarder = result.winner, not result.win_by_discard, result.discarder
    hand.tai = result.score.tai
    hand.patterns = result.score.patterns
    from game.simulator import payments
    hand.paid = min(-pay for pay in payments(result) if pay < 0)
    hand.winner_declared = result.state.declared[result.winner]
    tiles = [t for t, n in enumerate(winner.hand) for _ in range(n)]
    tiles.remove(result.win_tile)
    shapes = wait_shapes(tiles, result.win_tile)
    hand.shape = main_shape(shapes) if shapes else None
    return hand


# ---------- 統計 ----------

def _p(hits: int, total: int) -> str:
    if not total:
        return "—"
    p = hits / total
    return f"{p:6.1%} ±{1.96 * sqrt(p * (1 - p) / total):.1%}"


def _mean(values: list[float]) -> str:
    if len(values) < 2:
        return "—"
    m = sum(values) / len(values)
    sd = sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return f"{m:6.2f} ±{1.96 * sd / sqrt(len(values)):.2f}"


def _ratio(a: list[float], b: list[float]) -> str:
    """兩組平均的比值，誤差用重抽（bootstrap）。"""
    if len(a) < 2 or len(b) < 2:
        return "—"
    rng = Random(0)
    ratio = (sum(a) / len(a)) / (sum(b) / len(b))
    samples = sorted((sum(rng.choices(a, k=len(a))) / len(a)) / (sum(rng.choices(b, k=len(b))) / len(b))
                     for _ in range(300))
    return f"{ratio:6.2f} ±{(samples[292] - samples[7]) / 2:.2f}"


def metrics(hands: list[HandRecord]) -> list[tuple[str, str]]:
    known = [h for h in hands if h.winner is not None or h.drawn]
    won = [h for h in known if h.winner is not None]
    opp_wins = [h for h in won if h.winner != 0]
    n = len(known)
    rows = [("局數（有結果）", f"{n}"),
            ("流局率", _p(sum(h.drawn for h in known), n)),
            ("自己胡牌率（參考）", _p(sum(h.winner == 0 for h in known), n)),
            ("自己放槍率（參考）", _p(sum(h.discarder == 0 for h in known), n)),
            ("對手胡牌率（每家每局）", _p(len(opp_wins), 3 * n)),
            ("　其中自摸", _p(sum(h.self_draw for h in opp_wins), len(opp_wins))),
            ("對手放槍率（每家每局）", _p(sum(h.discarder not in (None, 0) for h in known), 3 * n)),
            ("　放給自己", _p(sum(h.discarder not in (None, 0) and h.winner == 0 for h in known), 3 * n)),
            ("　放給別的對手", _p(sum(h.discarder not in (None, 0) and h.winner != 0 for h in known), 3 * n))]
    rows.append(("對手胡牌每家付的 ÷ 自己胡牌", _ratio([h.paid for h in opp_wins if h.paid],
                                                   [h.paid for h in won if h.winner == 0 and h.paid])))
    rows.append(("對手胡牌有「聽」台", _p(sum(bool(h.winner_declared) for h in opp_wins), len(opp_wins))))
    win_melds = [h.melds[h.winner] for h in opp_wins if h.melds[h.winner] is not None]
    rows += [("對手胡牌時門清", _p(sum(m == 0 for m in win_melds), len(win_melds))),
             ("對手胡牌時副露組數", _mean(win_melds))]
    all_melds = [h.melds[s] for h in known for s in (1, 2, 3) if h.melds[s] is not None]
    rows.append(("對手每局副露組數（局末）", _mean(all_melds)))
    rows.append(("自己每局副露組數（參考）", _mean([h.melds[0] for h in known if h.melds[0] is not None])))
    shapes = [h.shape for h in opp_wins if h.shape]
    for shape in ("兩面", "雙碰", "嵌張", "邊張", "單吊"):
        rows.append((f"對手聽牌型態：{shape}", _p(shapes.count(shape), len(shapes))))
    declares = [(h, s, t) for h in known for s, t in h.declares.items()]
    rows += [("對手按聽（每家每局）", _p(len(declares), 3 * n)),
             ("對手按聽巡目", _mean([t for _, _, t in declares]))]
    with_declare = [h for h in known if h.declares]
    rows += [("有對手按聽的局：按聽者胡", _p(sum(h.winner in h.declares for h in with_declare), len(with_declare))),
             ("　其他對手放槍給按聽者", _p(sum(h.winner in h.declares and h.discarder not in (None, 0)
                                            for h in with_declare), len(with_declare))),
             ("　自己放槍給按聽者", _p(sum(h.winner in h.declares and h.discarder == 0
                                          for h in with_declare), len(with_declare)))]
    rows.append(("一局巡數", _mean([h.turns for h in known if h.turns])))
    firsts = [t for h in known for s in (1, 2, 3) for t in h.first_discards.get(s, ())]
    honors = sum(is_honor(t) for t in firsts)
    terminals = sum(not is_honor(t) and t % 9 in (0, 8) for t in firsts)
    rows += [("對手前兩張打字牌", _p(honors, len(firsts))),
             ("對手前兩張打么九", _p(terminals, len(firsts))),
             ("對手前兩張打中張", _p(len(firsts) - honors - terminals, len(firsts)))]
    return rows


def report(real: list[HandRecord], sim: list[HandRecord]) -> str:
    left, right = metrics(real), metrics(sim)
    lines = [f"{'項目':<22}{'實戰':>18}{'模擬器':>18}"]
    for (name, a), (_, b) in zip(left, right):
        lines.append(f"{name:<22}{a:>18}{b:>18}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("result"))
    parser.add_argument("--since", default="20261015")
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--me", default="nnguard:models/rl3_0960.pt")
    parser.add_argument("--opponents", default="rule")
    args = parser.parse_args()
    real = real_hands(args.dir, args.since, args.workers, args.dir / "opponents_settlements.pkl")
    print(f"實戰 {len(real)} 局", flush=True)
    from functools import partial
    with Pool(args.workers) as pool:
        sim = pool.map(partial(sim_hand, me=args.me, opponents=args.opponents), range(args.games), chunksize=8)
    print(report(real, sim))
    unknown = sum(h.winner is None and not h.drawn for h in real)
    print(f"（實戰另有 {unknown} 局結算讀不出；實戰的副露數、按聽是記錄檔認到的，都會少算）")


if __name__ == "__main__":
    main()
