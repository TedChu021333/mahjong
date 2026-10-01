"""配對比較兩種策略：同一副牌各打一次，只差在其中一個座位用的策略。

    python -m eval.arena --variant gold:8 --baseline gold:0 --games 2000

每個種子打兩局：四家都是 baseline、以及把一個座位換成 variant（座位與莊家依種子輪替）；
比較該座位兩局的輸贏差。起手牌相同，運氣成分大多抵銷，比各打各的少很多局就能分出高下。
輸贏以「底 + 台」計（底 BASE_TAI = 2 台，對應訓練場 100/50）：放槍由放槍者付，自摸三家
各付；非莊家自摸時莊家多付莊家的 1 台（結算畫面：閒家各付 400、莊家付 450）。

策略代號：
    rule          規則式 AI（目前的預設參數）
    gold:<n>      規則式 AI，打出金牌的代價視同少 n 張有效進張
    blind         規則式 AI，但忽略對手宣告聽牌（看不到「聽」標記時的行為）
    fold:<n>      規則式 AI，有對手宣告聽牌時，自己向聽數 ≥n 就棄胡
    nodeclare     規則式 AI，但聽牌時不宣告（不拿聽牌 1 台，手牌不鎖、還能防守）
    ev[:<w>]      期望值打牌（agent.ev_agent），w 為放槍損失的權重（預設 RISK_WEIGHT）
    evhonor:<h>   期望值打牌，會加台的字牌刻子／對子每台算進攻收益的 h 倍（0 = 不算）
    evdeclare:<k>[:<s>] 期望值打牌，聽的牌剩 k 張以上才宣告（0 = 一律宣告）；s 為估計對手放槍危險時
                  「沒宣告的人已經聽牌」的比例（預設 0：其他 AI 都一律宣告時，沒宣告就是沒聽）
    mc[:<n>]      蒙地卡羅打牌（agent.mc_agent），n 為每個候選的模擬次數（預設 ROLLOUTS）
    mcclaim[:<n>] 規則式 AI，但吃碰槓用蒙地卡羅判斷（單獨評估吃碰的效果）
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from random import Random

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType
from game.simulator import PlayerPolicy, payments, play_game


def make_policy(spec: str) -> PlayerPolicy:
    name, _, arg = spec.partition(":")
    if name == "rule" and not arg:
        return choose_rule_action_for_player
    if name == "gold" and arg:
        return partial(_gold_policy, penalty=float(arg))
    if name == "blind" and not arg:
        return _blind_policy
    if name == "mcclaim":
        return partial(_mc_claim_policy, rollouts=int(arg) if arg else 40)
    if name == "mc":
        from agent.mc_agent import ROLLOUTS

        return partial(_mc_policy, rollouts=int(arg) if arg else ROLLOUTS)
    if name == "ev":
        from agent.ev_agent import RISK_WEIGHT

        return partial(_ev_policy, risk_weight=float(arg) if arg else RISK_WEIGHT)
    if name == "evdeclare" and arg:
        from agent.ev_agent import RISK_WEIGHT

        live, _, share = arg.partition(":")
        return partial(_ev_policy, risk_weight=RISK_WEIGHT, declare_min_live=int(live),
                       undeclared_share=float(share) if share else 0.0)
    if name == "evhonor" and arg:
        from agent.ev_agent import RISK_WEIGHT

        return partial(_ev_policy, risk_weight=RISK_WEIGHT, honor_share=float(arg))
    if name == "nodeclare" and not arg:
        return _no_declare_policy
    if name == "fold" and arg:
        return partial(_fold_policy, shanten=int(arg))
    raise ValueError(f"未知的策略代號：{spec}")


def _mc_claim_policy(state, player, rng, rollouts):
    from agent.mc_agent import choose_mc_claim
    from agent.opponent_model import default_calibration
    from game.rules import Phase

    if state.phase == Phase.RESPONSE and not state.robbing_kong and not state.declared[player]             and state.last_discard is not None:
        return choose_mc_claim(state, player, rng, default_calibration(), rollouts, 0.0, None)
    return choose_rule_action_for_player(state, player, rng)


def _mc_policy(state, player, rng, rollouts):
    from agent.mc_agent import choose_mc_action_for_player

    return choose_mc_action_for_player(state, player, rng, rollouts=rollouts, undeclared_share=0.0)


def _ev_policy(state, player, rng, risk_weight, honor_share=None, declare_min_live=None,
               undeclared_share=0.0):
    from agent.ev_agent import HONOR_TAI_SHARE, choose_ev_action_for_player

    # 模擬器裡的 AI 聽牌一律宣告，沒宣告的人不會已經聽牌
    return choose_ev_action_for_player(
        state, player, rng, risk_weight=risk_weight, undeclared_share=undeclared_share,
        honor_share=HONOR_TAI_SHARE if honor_share is None else honor_share,
        declare_min_live=declare_min_live)


def _no_declare_policy(state, player, rng):
    action = choose_rule_action_for_player(state, player, rng)
    if action is not None and action.kind == ActionType.DECLARE:
        return Action(ActionType.DISCARD, tile=action.tile)
    return action


def _fold_policy(state, player, rng, shanten):
    return choose_rule_action_for_player(state, player, rng, declared_fold_shanten=shanten)


def _blind_policy(state, player, rng):
    return choose_rule_action_for_player(state, player, rng, use_declared=False)


def _gold_policy(state, player, rng, penalty):
    return choose_rule_action_for_player(state, player, rng, gold_penalty=penalty)


RAKE = 0.15
"""實戰贏的錢遊戲抽 15%（eval.stats 由金幣餘額對出），輸的照付；比較策略時照實戰算。"""


def after_rake(net: float) -> float:
    return net * (1 - RAKE) if net > 0 else net


@dataclass(frozen=True)
class PairedGame:
    seed: int
    seat: int
    variant_net: float
    baseline_net: float
    variant_won: bool
    variant_dealt_in: bool
    baseline_won: bool = False
    baseline_dealt_in: bool = False


def play_pair(seed: int, variant: str, baseline: str) -> PairedGame:
    seat, dealer = seed % 4, (seed // 4) % 4
    base_policy, var_policy = make_policy(baseline), make_policy(variant)

    def mixed(state, player, rng):
        return (var_policy if player == seat else base_policy)(state, player, rng)

    reference = play_game(Random(seed), player_policy=base_policy, dealer=dealer, swap=True)
    trial = play_game(Random(seed), player_policy=mixed, dealer=dealer, swap=True)
    return PairedGame(seed, seat, after_rake(payments(trial)[seat]),
                      after_rake(payments(reference)[seat]),
                      trial.winner == seat, trial.discarder == seat,
                      reference.winner == seat, reference.discarder == seat)


@dataclass(frozen=True)
class Summary:
    games: int
    variant_mean: float
    baseline_mean: float
    difference: float
    margin: float
    """差值的 95% 信賴區間半寬。"""
    win_rate: float
    deal_in_rate: float
    baseline_win_rate: float = 0.0
    baseline_deal_in_rate: float = 0.0


def summarize(games: list[PairedGame]) -> Summary:
    n = len(games)
    if n < 2:
        raise ValueError("至少需要 2 局才能估計信賴區間")
    diffs = [game.variant_net - game.baseline_net for game in games]
    mean = sum(diffs) / n
    variance = sum((diff - mean) ** 2 for diff in diffs) / (n - 1)
    return Summary(
        games=n,
        variant_mean=sum(game.variant_net for game in games) / n,
        baseline_mean=sum(game.baseline_net for game in games) / n,
        difference=mean,
        margin=1.96 * (variance / n) ** 0.5,
        win_rate=sum(game.variant_won for game in games) / n,
        deal_in_rate=sum(game.variant_dealt_in for game in games) / n,
        baseline_win_rate=sum(game.baseline_won for game in games) / n,
        baseline_deal_in_rate=sum(game.baseline_dealt_in for game in games) / n,
    )


def describe(summary: Summary) -> str:
    return (f"{summary.games} 局：每局 {summary.variant_mean:+.3f} 台"
            f"（對照 {summary.baseline_mean:+.3f}），差 {summary.difference:+.3f}"
            f" ± {summary.margin:.3f}；胡牌率 {summary.win_rate:.1%}"
            f"（對照 {summary.baseline_win_rate:.1%}）、放槍率 {summary.deal_in_rate:.1%}"
            f"（對照 {summary.baseline_deal_in_rate:.1%}）")


def compare(variant: str, baseline: str, games: int, start: int = 0, workers: int = 1,
            out: Path | None = None, progress_every: int = 200) -> Summary:
    """out 有給時每局結果立即寫進 CSV，中斷後已跑完的部分不會遺失。"""
    make_policy(variant), make_policy(baseline)  # 先檢查代號
    task = partial(play_pair, variant=variant, baseline=baseline)
    seeds = range(start, start + games)
    results: list[PairedGame] = []
    writer = file = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        file = out.open("a", newline="", encoding="utf-8")
        writer = csv.writer(file)
    began = time.time()
    try:
        with Pool(workers) if workers > 1 else _Serial() as pool:
            for game in pool.imap_unordered(task, seeds, chunksize=4):
                results.append(game)
                if writer is not None:
                    writer.writerow([variant, baseline, game.seed, game.seat, game.variant_net,
                                     game.baseline_net, int(game.variant_won),
                                     int(game.variant_dealt_in), int(game.baseline_won),
                                     int(game.baseline_dealt_in)])
                    file.flush()
                if progress_every and len(results) % progress_every == 0 and len(results) > 1:
                    print(f"[{time.time() - began:.0f}s] {describe(summarize(results))}",
                          flush=True)
    finally:
        if file is not None:
            file.close()
    return summarize(results)


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(func, items, chunksize=1):
        return map(func, items)


def main() -> None:
    parser = argparse.ArgumentParser(description="配對比較兩種策略")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline", default="rule")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0, help="第一個種子")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, help="逐局結果附加寫入的 CSV")
    args = parser.parse_args()
    summary = compare(args.variant, args.baseline, args.games, args.start, args.workers,
                      args.out)
    print(f"{args.variant} 對 {args.baseline}：{describe(summary)}")


if __name__ == "__main__":
    main()
