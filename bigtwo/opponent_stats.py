"""從實戰記錄（result/bigtwo_log_*.txt）統計對手的出牌習慣，和模擬器裡的 AI 比較。

記錄只寫出牌，PASS 由出牌順序推回（自己 → 下家 → 對家 → 上家）：下一手是別人出的就是在壓上一手，
中間輪到的人 PASS；下一手還是同一家出的，就是其他三家都 PASS、他重新帶頭。記錄有漏的局（跟牌牌型對不上）整局不用。

python -m bigtwo.opponent_stats            實戰對手 vs 模擬器裡的規則式 AI
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

from .cards import RANK_TWO, SUIT_NAMES, RANK_NAMES, card, rank_of
from .combos import Combo, Kind, beats, classify

SEATS = {"自己": 0, "下家": 1, "對家": 2, "上家": 3}
_CARD_NAME = {SUIT_NAMES[s] + RANK_NAMES[r]: card(r, s) for r in range(13) for s in range(4)}


def parse_combo(text: str) -> Combo | None:
    names = text.split()[1:]
    try:
        return classify([_CARD_NAME[n] for n in names])
    except KeyError:
        return None


@dataclass
class Event:
    seat: int
    kind: str                 # "lead" / "follow" / "pass"
    combo: Combo | None       # 出的牌（PASS 時 None）
    target: Combo | None      # 要壓的牌（帶頭時 None）
    count: int                # 出牌前手上幾張
    others_min: int           # 其他三家最少幾張


def games_from_logs(paths) -> list[list[tuple[int, Combo]]]:
    """每局的出牌序列 [(座位, 牌)]。以「換三張」分局。"""
    games, cur, pressed = [], None, None
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "換三張：" in line or "模式啟動" in line:
                if cur:
                    games.append(cur)
                cur = [] if "換三張：" in line else None
                pressed = None
                continue
            if "按「出牌」" in line and cur is not None:
                pressed = len(cur)      # 自己的出牌在按鈕當下就發生了，「自己出」常晚幾秒才記
                continue
            m = re.search(r"\] (自己|下家|對家|上家)出 (.+)$", line)
            if m and cur is not None:
                combo = parse_combo(m.group(2))
                if combo is None:
                    cur = None          # 認不出的牌：這局不用
                    continue
                seat = SEATS[m.group(1)]
                if seat == 0 and pressed is not None:
                    cur.insert(pressed, (seat, combo))
                    pressed = None
                else:
                    cur.append((seat, combo))
    if cur:
        games.append(cur)
    return [g for g in games if g]


def events_of(plays: list[tuple[int, Combo]]) -> list[Event] | None:
    """推回每個人每次輪到時做了什麼；牌型對不上（記錄有漏）回傳 None。"""
    counts = [13, 13, 13, 13]
    seen = set()
    out = []
    last_seat, last = None, None
    for seat, combo in plays:
        if set(combo.cards) & seen:
            return None
        seen |= set(combo.cards)
        if last_seat is None or seat == last_seat:
            if last_seat is not None:          # 其他三家都 PASS
                for k in range(1, 4):
                    s = (last_seat + k) % 4
                    out.append(Event(s, "pass", None, last, counts[s], _others(counts, s)))
            out.append(Event(seat, "lead", combo, None, counts[seat], _others(counts, seat)))
        else:
            if not beats(combo, last):
                return None
            s = (last_seat + 1) % 4
            while s != seat:
                out.append(Event(s, "pass", None, last, counts[s], _others(counts, s)))
                s = (s + 1) % 4
            out.append(Event(seat, "follow", combo, last, counts[seat], _others(counts, seat)))
        counts[seat] -= combo.size
        if counts[seat] < 0:
            return None
        last_seat, last = seat, combo
    return out


def _others(counts, seat):
    return min(n for s, n in enumerate(counts) if s != seat)


def summarize(events: list[Event], seats=(1, 2, 3)) -> dict[str, float]:
    """幾個習慣的比例（只算 seats 的決定）。"""
    ev = [e for e in events if e.seat in seats]
    lead = [e for e in ev if e.kind == "lead"]
    face_single = [e for e in ev if e.target is not None and e.target.kind == Kind.SINGLE]
    face_pair = [e for e in ev if e.target is not None and e.target.kind == Kind.PAIR]
    early_single_follow = [e for e in face_single if e.kind == "follow" and e.count >= 8]
    stats = {
        "帶頭出單張": _ratio(lead, lambda e: e.combo.kind == Kind.SINGLE),
        "帶頭出對子": _ratio(lead, lambda e: e.combo.kind == Kind.PAIR),
        "帶頭出五張": _ratio(lead, lambda e: e.combo.size == 5),
        "面對單張時壓": _ratio(face_single, lambda e: e.kind == "follow"),
        "面對對子時壓": _ratio(face_pair, lambda e: e.kind == "follow"),
        "手上8張以上壓單張時用2": _ratio(early_single_follow, lambda e: rank_of(e.combo.cards[0]) == RANK_TWO),
        "壓單張時點數跳超過3": _ratio([e for e in face_single if e.kind == "follow"],
                                  lambda e: rank_of(e.combo.cards[0]) - rank_of(e.target.cards[0]) > 3),
        "每局出牌手數": 0.0,
    }
    stats["_n"] = {"帶頭": len(lead), "面對單張": len(face_single), "面對對子": len(face_pair)}
    return stats


def _ratio(items, pred) -> float:
    return sum(1 for e in items if pred(e)) / len(items) if items else float("nan")


def simulated_events(policy, games: int, seed: int = 0) -> list[list[Event]]:
    from .agents import rule_swap
    from .game import play_game
    rng = Random(seed)
    out = []
    for _ in range(games):
        result = play_game([policy] * 4, rng, swappers=[rule_swap] * 4)
        plays = [(p.seat, p.combo) for p in result.history if p.combo is not None]
        ev = events_of(plays)
        if ev:
            out.append(ev)
    return out


def report(real: list[list[Event]], sim: list[list[Event]], label: str = "模擬器規則式") -> str:
    r = summarize([e for g in real for e in g])
    s = summarize([e for g in sim for e in g], seats=(0, 1, 2, 3))
    r["每局出牌手數"] = sum(sum(1 for e in g if e.kind != "pass") for g in real) / len(real)
    s["每局出牌手數"] = sum(sum(1 for e in g if e.kind != "pass") for g in sim) / len(sim)
    lines = [f"實戰 {len(real)} 局（對手的決定 {r['_n']}） vs {label} {len(sim)} 局"]
    for key in r:
        if key.startswith("_"):
            continue
        fmt = "{:6.2f}" if key == "每局出牌手數" else "{:6.1%}"
        lines.append(f"  {key:16} 實戰 {fmt.format(r[key])}   {label} {fmt.format(s[key])}")
    return "\n".join(lines)


def main(argv=None):
    from .agents import rule_policy
    paths = sorted(Path("result").glob("bigtwo_log_*.txt"))
    games = games_from_logs(paths)
    real = [ev for ev in (events_of(g) for g in games) if ev]
    print(f"記錄 {len(games)} 局，推得回出牌順序的 {len(real)} 局")
    print(report(real, simulated_events(rule_policy, 2000)))


if __name__ == "__main__":
    sys.exit(main())
