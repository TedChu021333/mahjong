"""分析人機對戰的記錄（play/record.py）：自己打得哪裡有問題。

    python -m play.analyze [--date 20261030] [--skip 1] [--all]

--skip N 不算那天的前 N 圈；--all 列出每一個和 AI 不同的決定（預設只列有問題的）。
比較對象是同一局面 rl3_0960＋護欄（三家電腦用的那個）會怎麼打。AI 不是標準答案，只是參考：
一個決定差多少台很難從單手看出來（打法好壞每局約 0.1～0.3 台，被一局 7 台的輸贏波動蓋住），
所以這裡只挑看得出的錯：漏胡、退向聽、同向聽進張少很多、有人聽牌時打危險牌、放槍那張有沒有更安全的選擇。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from game.tiles import tile_name

from play.record import RESULT_DIR

ME = 0
SEATS = ("自己", "下家", "對家", "上家")
OUTS_GAP = 4
"""同向聽進張比最好的少這麼多張以上算「進張少」（和護欄的浪費檢查 WASTE_OUTS 一樣）。"""
DANGER_GAP = 0.05
DANGER_MIN = 0.08
"""有人聽牌時，打的牌放槍機率 ≥8%、又比 AI 那張高 5 個百分點以上，算「打危險牌」。"""
SUIT_CHARS = "萬筒條"


def hand_text(counts: list[int]) -> str:
    parts = []
    for suit in range(3):
        ranks = "".join(tile_name(suit * 9 + r)[0] * counts[suit * 9 + r] for r in range(9))
        if ranks:
            parts.append(ranks + SUIT_CHARS[suit])
    honors = "".join(tile_name(t) * counts[t] for t in range(27, 34))
    if honors:
        parts.append(honors)
    return " ".join(parts)


def action_text(action: dict | None) -> str:
    if action is None:
        return "（沒有）"
    kind, tile = action["kind"], action.get("tile")
    name = tile_name(tile) if tile is not None else ""
    if kind == "DISCARD":
        return f"打{name}"
    if kind == "DECLARE":
        return f"打{name}聽"
    if kind == "CHOW":
        return "吃" + "".join(tile_name(t)[0] for t in action["tiles"]) + tile_name(action["tiles"][0])[1]
    if kind == "PUNG":
        return f"碰{name}"
    if kind == "KONG":
        return f"槓{name}"
    if kind == "WIN":
        return "胡"
    if kind == "PASS":
        return "不要"
    if kind == "SWAP":
        return "換" + ("".join(tile_name(t) for t in action["tiles"]) or "（不換）")
    return kind


@dataclass
class Hand:
    round_index: int
    decisions: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    end: dict | None = None


def load_rounds(paths: list[Path]) -> list[list[Hand]]:
    rounds = []
    for index, path in enumerate(paths):
        hands = [Hand(index)]
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            kind = record["type"]
            if kind == "decision":
                hands[-1].decisions.append(record)
            elif kind == "event":
                hands[-1].events.append(record)
            elif kind == "hand_end":
                hands[-1].end = record
                hands.append(Hand(index))
        rounds.append([hand for hand in hands if hand.end is not None])
    return rounds


def meld_text(meld: dict) -> str:
    tiles = meld["tiles"]
    if meld["kind"] == "chow":
        return "吃" + "".join(tile_name(t)[0] for t in tiles) + tile_name(tiles[0])[1]
    if meld["kind"] == "kong":
        return ("暗槓" if meld.get("concealed") else "槓") + tile_name(tiles[0])
    return "碰" + tile_name(tiles[0])


def other_declared(view: dict) -> list[int]:
    return [seat for seat in range(4) if seat != ME and view["declared"][seat]]


def judge(decision: dict) -> list[str]:
    """這個決定有哪些看得出的問題（空的＝沒問題或看不出）。"""
    problems = []
    kinds = {a["kind"] for a in decision["actions"]}
    choice, ai = decision["choice"], decision.get("ai")
    if "WIN" in kinds and choice["kind"] != "WIN":
        problems.append("漏胡：能胡沒胡")
    if same(choice, ai):
        return problems  # 和 AI 一樣（例如有人聽牌時為了安全犧牲進張）不算問題
    options = {int(t): v for t, v in (decision.get("options") or {}).items()}
    if choice["kind"] in ("DISCARD", "DECLARE") and options and choice["tile"] in options:
        mine = options[choice["tile"]]
        best = min(v["shanten"] for v in options.values())
        if mine["shanten"] > best:
            problems.append(f"退向聽：打完 {mine['shanten']} 向聽，最好的打法 {best} 向聽")
        else:
            top = max(v["outs"] for v in options.values() if v["shanten"] == mine["shanten"])
            if mine["outs"] <= top - OUTS_GAP:
                better = [tile_name(t) for t, v in options.items()
                          if v["shanten"] == mine["shanten"] and v["outs"] == top]
                problems.append(f"進張少：打完進張 {mine['outs']} 張，打{'／'.join(better)} {top} 張")
        declared = other_declared(decision["view"])
        if declared and ai and ai.get("tile") in options:
            theirs = options[ai["tile"]]
            if mine["danger"] >= DANGER_MIN and mine["danger"] >= theirs["danger"] + DANGER_GAP:
                who = "、".join(SEATS[s] for s in declared)
                problems.append(f"危險牌：{who}聽牌，打{tile_name(choice['tile'])}放槍 {mine['danger']:.0%}，"
                                f"AI 打{tile_name(ai['tile'])} {theirs['danger']:.0%}")
    if ai and choice["kind"] in ("CHOW", "PUNG", "KONG") and ai["kind"] == "PASS" \
            and decision["view"]["phase"] == "RESPONSE":
        note = decision.get("ai_note") or ""
        problems.append("吃碰：AI 不要" + (f"（{note}）" if note else ""))
    return problems


def describe(decision: dict) -> str:
    view = decision["view"]
    hand = hand_text(view["hand"])
    where = ""
    if view["phase"] == "RESPONSE" and view["last_discard"] is not None:
        where = f"　{SEATS[view['discard_player']]}打{tile_name(view['last_discard'])}"
    elif view.get("last_drawn") is not None:
        where = f"　摸{tile_name(view['last_drawn'])}"
    melds = view["melds"][ME]
    shown = ("　副露 " + " ".join(meld_text(m) for m in melds)) if melds else ""
    declared = other_declared(view)
    wall = f"　牌牆 {view['drawable']}" + (f"　{'、'.join(SEATS[s] for s in declared)}已聽" if declared else "")
    return f"{decision['time']}　{hand}{shown}{where}{wall}"


def discard_line(decision: dict) -> str | None:
    options = {int(t): v for t, v in (decision.get("options") or {}).items()}
    picks = [decision["choice"].get("tile"), (decision.get("ai") or {}).get("tile")]
    parts = []
    for tile in dict.fromkeys(t for t in picks if t in options):
        v = options[tile]
        parts.append(f"{tile_name(tile)}：{v['shanten']} 向聽 進張 {v['outs']} 放槍 {v['danger']:.0%}")
    return "　".join(parts) or None


def category(decision: dict) -> str:
    phase = decision["view"]["phase"]
    if phase == "SWAP":
        return "換三張"
    if phase == "RESPONSE":
        return "吃碰胡"
    return "打牌"


def same(choice: dict, ai: dict | None) -> bool:
    if ai is None:
        return True
    if choice["kind"] == "SWAP" and ai["kind"] == "SWAP":
        return sorted(choice["tiles"]) == sorted(ai["tiles"])
    return choice["kind"] == ai["kind"] and choice.get("tile") == ai.get("tile") \
        and sorted(choice.get("tiles") or []) == sorted(ai.get("tiles") or [])


def deal_in_report(hand: Hand) -> list[str]:
    end = hand.end
    win_tile = end["win_tile"]
    mine = [d for d in hand.decisions if d["choice"].get("tile") == win_tile
            and d["choice"]["kind"] in ("DISCARD", "DECLARE")]
    winner = SEATS[end["winner"]]
    head = f"放槍給{winner}（{tile_name(win_tile)}，{end['tai']} 台{'、' + winner + '有聽' if end['declared'][end['winner']] else ''}）"
    if not mine:
        return [head + "：那張不是自己選的（只剩一種打法或聽牌後代打）"]
    decision = mine[-1]
    lines = [head, "  " + describe(decision)]
    options = {int(t): v for t, v in (decision.get("options") or {}).items()}
    if options and win_tile in options:
        mine_v = options[win_tile]
        safer = sorted((v["danger"], t) for t, v in options.items()
                       if v["shanten"] <= mine_v["shanten"] and v["danger"] < mine_v["danger"] - 0.03)
        lines.append(f"  當時估計放槍 {mine_v['danger']:.0%}" + (
            f"；不退向聽、較安全的有 " + "、".join(f"{tile_name(t)} {d:.0%}" for d, t in safer[:3])
            if safer else "；沒有不退向聽又明顯安全的牌"))
    lines.append(f"  AI：{action_text(decision.get('ai'))}" + (f"（{decision['ai_note']}）" if decision.get("ai_note") else ""))
    return lines


def report(rounds: list[list[Hand]], show_all: bool) -> str:
    hands = [hand for hands in rounds for hand in hands]
    lines = []
    if not hands:
        return "沒有記錄（play_log_*.jsonl）。"
    n = len(hands)
    wins = sum(h.end["winner"] == ME for h in hands)
    tsumo = sum(h.end["winner"] == ME and h.end["self_draw"] for h in hands)
    deal = sum(h.end["discarder"] == ME for h in hands)
    pay = sum(h.end["pay"][ME] for h in hands)
    lines.append(f"{len(rounds)} 圈 {n} 局：自己胡 {wins} 局（{wins / n:.0%}，自摸 {tsumo}）、放槍 {deal} 局（{deal / n:.0%}）、"
                 f"合計 {pay:+d} 台（每局 {pay / n:+.2f}）")
    for seat in (1, 2, 3):
        w = sum(h.end["winner"] == seat for h in hands)
        d = sum(h.end["discarder"] == seat for h in hands)
        p = sum(h.end["pay"][seat] for h in hands)
        lines.append(f"  {SEATS[seat]}（電腦）：胡 {w / n:.0%}、放槍 {d / n:.0%}、合計 {p:+d} 台")
    lines.append("  （每局輸贏標準差約 7 台，局數少時排名和合計大多是運氣）")

    decisions = [(hand, d) for hand in hands for d in hand.decisions]
    agree = Counter()
    total = Counter()
    for _, d in decisions:
        total[category(d)] += 1
        agree[category(d)] += same(d["choice"], d.get("ai"))
    lines.append("")
    lines.append("和 AI 一樣的比例：" + "　".join(
        f"{c} {agree[c]}/{total[c]}（{agree[c] / total[c]:.0%}）" for c in ("打牌", "吃碰胡", "換三張") if total[c]))

    flagged = Counter()
    detail = []
    for hand, d in decisions:
        problems = judge(d)
        for problem in problems:
            flagged[problem.split("：")[0]] += 1
        if problems or (show_all and not same(d["choice"], d.get("ai"))):
            detail.append((d, problems))
    lines.append("")
    lines.append("看得出的問題：" + ("、".join(f"{k} {v} 次" for k, v in flagged.most_common()) or "沒有"))

    deal_hands = [h for h in hands if h.end["discarder"] == ME]
    if deal_hands:
        lines.append("")
        lines.append(f"放槍 {len(deal_hands)} 次：")
        for hand in deal_hands:
            lines.extend("  " + line for line in deal_in_report(hand))

    if detail:
        lines.append("")
        lines.append("逐手（有問題的" + ("，與和 AI 不同的" if show_all else "") + "）：")
        for d, problems in detail:
            lines.append("  " + describe(d))
            lines.append(f"    你：{action_text(d['choice'])}　AI：{action_text(d.get('ai'))}"
                         + (f"（{d['ai_note']}）" if d.get("ai_note") else ""))
            numbers = discard_line(d)
            if numbers:
                lines.append("    " + numbers)
            for problem in problems:
                lines.append("    ⚠ " + problem)
    return "\n".join(lines)


def log_paths(date: str, directory: Path = RESULT_DIR) -> list[Path]:
    return sorted(directory.glob(f"play_log_{date}_*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="分析人機對戰的記錄")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--skip", type=int, default=0, help="不算前幾圈")
    parser.add_argument("--all", action="store_true", help="列出每一個和 AI 不同的決定")
    args = parser.parse_args()
    paths = log_paths(args.date)[args.skip:]
    print(report(load_rounds(paths), args.all))


if __name__ == "__main__":
    main()
