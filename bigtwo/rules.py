"""大老二規則設定（明星三缺一，依遊戲內規則說明 2026-10-16）。

確定的（規則說明頁）：
- 持有梅花 3 的人先出，逆時針輪流；其他人要出同樣牌型且比上一家大，時間到算 PASS。
- 點數 2 > A > K > … > 3，花色 黑桃 > 紅心 > 方塊 > 梅花，先比點數再比花色。
- 牌型：單張、一對、三條、順子、葫蘆、鐵支（四張＋一張任意牌）、同花順。
  順子 A2345 最小、23456 最大；同點數順子比最大那張的花色。葫蘆比三張的點數。
- 特殊加倍（輸家負分再加一次本局倍率）：剩超過 9 張、手上有 2、13 張都沒出、
  贏家最後一手含梅花 3、有發牌權的人一路出完沒讓別人出過任何牌（滿貫全壘打）。

還沒確認、先做成開關的（看實戰錄影／結算畫面再改）：
- 牌型表沒有「同花」，`flush_enabled` 預設關。
- 鐵支、同花順能不能壓單張／對子／三條（「必須出同樣的牌型」→ 預設不能）。
- 第一手是否一定要含梅花 3（多數台灣規則要，預設要）。
- PASS 之後同一輪還能不能再出（預設可以）。
- 加倍是「多加一倍」還是「再乘一倍」：預設輸家扣 張數 ×（1 + 符合項數）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    flush_enabled: bool = False
    bombs_beat_smaller: bool = False
    first_must_include_club3: bool = True
    pass_locks_out: bool = False
    stacking: str = "add"   # "add"：張數 × (1 + k)；"double"：張數 × 2^k


DEFAULT_RULES = Rules()
