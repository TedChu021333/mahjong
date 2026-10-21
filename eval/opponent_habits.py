"""遊戲電腦的打牌習慣：從結算截圖（四家完整手牌、誰胡、誰放槍、有沒有宣告）與記錄檔（每一步認到的宣告者）統計，
給「讓模擬器的對手更像遊戲電腦」用。

    python -m eval.opponent_habits [--dates 20261015 20261016 ...] [--csv 檔案]

每局一筆紀錄（HandRecord）：四家（東南西北列）各自的結果、局末手牌的向聽數、這局有沒有被認到宣告；
自己那列另外標出。對手＝自己以外的三家。結算截圖不分副露與暗手，向聽數把 16 張當成全部暗手算
（副露本來就是完整的一組，算法一樣）；有認不出的牌就不算這家的向聽數。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent.shanten import shanten
from eval.deal_in import parse_log
from eval.settlement import SettlementReader
from eval.stats import RESULT_DIR, ROW_TOPS, collect, load_digits, load_names, read_row, wind_row
from perception.capture import read_image

WINDS = "東南西北"


@dataclass
class HandRecord:
    stamp: str
    me: int
    """自己是第幾列（東南西北）。"""
    results: list[str]
    """每列：胡牌、自摸、放槍、被自摸、沒輸贏。"""
    shantens: list[int | None]
    """每列局末手牌的向聽數（胡牌者為 -1；有認不出的牌為 None）。"""
    winner_declared: bool
    declared: set[int] = field(default_factory=set)
    """記錄檔裡這局被認到宣告的列（只有對手；10/5 以前的記錄沒寫，空集合）。"""
    first_declared: dict[int, int] = field(default_factory=dict)
    """列 → 第一次認到宣告時自己已經打了幾手。"""
    has_log: bool = False


def row_results(amounts: list[int | None]) -> list[str]:
    """四列金額 → 每列的結果。"""
    losers = [i for i, amount in enumerate(amounts) if amount is not None and amount < 0]
    tsumo = len(losers) == 3
    results = []
    for amount in amounts:
        if amount is not None and amount > 0:
            results.append("自摸" if tsumo else "胡牌")
        elif amount is not None and amount < 0:
            results.append("被自摸" if tsumo else "放槍")
        else:
            results.append("沒輸贏")
    return results


def hand_shanten(tiles) -> int | None:
    if any(tile is None for tile in tiles) or len(tiles) not in (16, 17):
        return None
    counts = [0] * 34
    for tile in tiles:
        if tile >= 34:
            return None
        counts[tile] += 1
    if max(counts) > 4:
        return None
    return shanten(counts, 0)


def collect_records(dates: list[str], directory: Path = RESULT_DIR) -> list[HandRecord]:
    digits, names, reader = load_digits(), load_names(), SettlementReader.load()
    games = {}
    for date in dates:
        log = directory / f"log_{date}.txt"
        if log.exists():
            for game in parse_log(log):
                if game.stamp:
                    games[game.stamp] = game
    records = []
    for date in dates:
        for rnd in collect(directory, date, None, None):
            for hand in rnd.hands:
                path = directory / f"{hand.stamp}.png"
                image = read_image(path)
                if image is None:
                    continue
                rows = [read_row(image, top, digits, names) for top in ROW_TOPS]
                mine = [i for i, row in enumerate(rows) if row.is_me]
                me = mine[0] if len(mine) == 1 else wind_row(image)
                if me is None:
                    continue
                settlement = reader.read(image)
                shantens = []
                for row in settlement.rows:
                    shantens.append(-1 if row.win_tile is not None else hand_shanten(row.tiles))
                record = HandRecord(hand.stamp, me, row_results([row.amount for row in rows]),
                                    shantens, settlement.winner_declared)
                game = games.get(hand.stamp)
                if game is not None and any(d.declared is not None for d in game.decisions):
                    record.has_log = True
                    for index, decision in enumerate(game.decisions):
                        for seat in decision.declared or ():
                            row = (me + seat) % 4  # 下家 1、對家 2、上家 3 依出牌順序在自己後面
                            record.declared.add(row)
                            record.first_declared.setdefault(row, index + 1)
                records.append(record)
    return records


def summarize(records: list[HandRecord]) -> str:
    lines = [f"{len(records)} 局"]
    me_res, bot_res = Counter(), Counter()
    for record in records:
        for row, result in enumerate(record.results):
            (me_res if row == record.me else bot_res)[result] += 1
    n_me, n_bot = sum(me_res.values()), sum(bot_res.values())

    def rates(c: Counter, n: int) -> str:
        win = c["胡牌"] + c["自摸"]
        return (f"胡 {win / n:.1%}（自摸 {c['自摸'] / n:.1%}）、放槍 {c['放槍'] / n:.1%}、"
                f"被自摸 {c['被自摸'] / n:.1%}")

    lines.append(f"自己：{rates(me_res, n_me)}")
    lines.append(f"電腦（平均每家）：{rates(bot_res, n_bot)}")
    wins = [(r, row) for r in records for row, res in enumerate(r.results) if res in ("胡牌", "自摸")]
    bot_wins = [(r, row) for r, row in wins if row != r.me]
    if bot_wins:
        declared = sum(r.winner_declared for r, _ in bot_wins) / len(bot_wins)
        tsumo = sum(r.results[row] == "自摸" for r, row in bot_wins) / len(bot_wins)
        lines.append(f"電腦胡牌時：有宣告 {declared:.0%}、自摸 {tsumo:.0%}（{len(bot_wins)} 次）")
    # 局末沒胡的電腦：聽牌率；有記錄的局再看聽牌的有沒有被認到宣告
    tenpai = total = tenpai_logged = tenpai_declared = 0
    for record in records:
        for row, value in enumerate(record.shantens):
            if row == record.me or value is None or value < 0:
                continue
            total += 1
            if value == 0:
                tenpai += 1
                if record.has_log:
                    tenpai_logged += 1
                    tenpai_declared += row in record.declared
    if total:
        lines.append(f"局末沒胡的電腦：聽牌 {tenpai / total:.0%}（{total} 家次）")
    if tenpai_logged:
        lines.append(f"  其中有記錄的局、聽牌的電腦被認到宣告：{tenpai_declared / tenpai_logged:.0%}"
                     f"（{tenpai_logged} 家次；地聽可能沒有標記，偏低）")
    # 電腦放槍：放給有宣告／沒宣告的胡牌者
    deal_ins = Counter()
    for record in records:
        for row, result in enumerate(record.results):
            if row != record.me and result == "放槍":
                deal_ins["放給宣告者" if record.winner_declared else "放給沒宣告的"] += 1
    if deal_ins:
        lines.append("電腦放槍：" + "、".join(f"{k} {v}" for k, v in deal_ins.items()))
    timing = [turn for r in records for row, turn in r.first_declared.items()]
    if timing:
        timing.sort()
        lines.append(f"電腦宣告的時機（自己打到第幾手時第一次認到）：中位數 {timing[len(timing) // 2]}、"
                     f"四分位 {timing[len(timing) // 4]}～{timing[len(timing) * 3 // 4]}（{len(timing)} 次）")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="遊戲電腦的打牌習慣")
    parser.add_argument("--dates", nargs="*", help="YYYYMMDD，預設 result/ 裡所有有結算截圖的日期")
    parser.add_argument("--dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--csv", type=Path, help="每局一行另存 CSV")
    args = parser.parse_args()
    dates = args.dates or sorted({path.name[:8] for path in args.dir.glob("2026????_??????.png")})
    records = collect_records(dates, args.dir)
    print(summarize(records))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["截圖", "自己", *[f"{w}結果" for w in WINDS], *[f"{w}向聽" for w in WINDS],
                             "胡牌者宣告", "認到宣告的列", "第一次認到宣告"])
            for r in records:
                writer.writerow([r.stamp, WINDS[r.me], *r.results, *r.shantens, int(r.winner_declared),
                                 "".join(WINDS[row] for row in sorted(r.declared)),
                                 " ".join(f"{WINDS[row]}{turn}" for row, turn in r.first_declared.items())])


if __name__ == "__main__":
    main()
