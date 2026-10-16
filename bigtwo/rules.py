"""大老二規則設定（明星三缺一，依遊戲內規則說明 2026-10-16）。

確定的（規則說明頁）：
- 持有梅花 3 的人先出，逆時針輪流；其他人要出同樣牌型且比上一家大，時間到算 PASS。
- 點數 2 > A > K > … > 3，花色 黑桃 > 紅心 > 方塊 > 梅花，先比點數再比花色。
- 牌型：單張、一對、三條、順子、葫蘆、鐵支（四張＋一張任意牌）、同花順。
  順子 A2345 最小、23456 最大；同點數順子比最大那張的花色。葫蘆比三張的點數。
- 特殊加倍（輸家負分再加一次本局倍率）：剩超過 9 張、手上有 2、13 張都沒出、
  贏家最後一手含梅花 3、有發牌權的人一路出完沒讓別人出過任何牌（滿貫全壘打）。

使用者確認（10/16）：
- 鐵支、同花順可以壓單張、對子、三條。
- 第一手不一定要含梅花 3（梅花 3 留到最後出完還有「最後脫手3」加倍）。
- PASS 之後同一輪還能再出。
- 加倍是一項乘一次兩倍：剩 11 張又有 2 → 11 × 2 × 2。

還沒確認、先做成開關的：
- 牌型表沒有「同花」，`flush_enabled` 預設關。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    flush_enabled: bool = False
    bombs_beat_smaller: bool = True
    first_must_include_club3: bool = False
    pass_locks_out: bool = False
    stacking: str = "double"   # "double"：張數 × 2^k；"add"：張數 × (1 + k)


DEFAULT_RULES = Rules()
