"""每局怎麼結束：記下每局的莊家；一局沒有結算畫面就換到下一局時，用莊家有沒有換判斷是流局還是漏認結算。

明星三缺一一圈的規則：莊家胡牌或流局就連莊，否則下家接莊（座位不動，畫面上「莊」換到下一家）。
流局沒有小結算畫面；別人胡牌、自己沒輸贏時照理有小結算，但曾經沒認到。10/19 15:19 那局開局前後
自己的門風從西變南（換莊）、金幣沒變，比較像漏認的結算而不是流局。
"""
from __future__ import annotations

SEAT_NAMES = {"me": "自己", "right": "下家", "top": "對家", "left": "上家"}
DRAW_TEXT = "流局（莊家沒換，連莊）"
"""寫進記錄檔的字樣，eval.stats 以它統計流局。"""
MISSED_TEXT = "結算畫面沒認到（莊家換了，有人胡）"


class HandBook:
    def __init__(self) -> None:
        self.dealer: str | None = None
        """這一局認到的莊家（me/right/top/left）。"""
        self._unsettled: tuple[str | None] | None = None
        """上一局沒有結算畫面就結束時，上一局的莊家（包成 tuple，區分「沒有」與「莊家不明」）。"""

    def hand_ended(self, settled: bool) -> None:
        """一局結束：settled 為有沒有認到結算畫面（換三張、牌河清空才發現的就是沒有）。"""
        self._unsettled = None if settled else (self.dealer,)
        self.dealer = None

    def see_dealer(self, dealer: str | None) -> list[str]:
        """這一局第一次認到莊家時回傳要記錄的訊息。"""
        if dealer is None or self.dealer is not None:
            return []
        self.dealer = dealer
        messages = [f"本局莊家：{SEAT_NAMES[dealer]}"]
        if self._unsettled is not None:
            previous = self._unsettled[0]
            self._unsettled = None
            if previous is None:
                messages.append("上一局沒有結算畫面，也不知道上一局的莊家，無法判斷是不是流局")
            elif previous == dealer:
                messages.append(f"上一局沒有結算畫面：{DRAW_TEXT}")
            else:
                messages.append(f"上一局沒有結算畫面：{MISSED_TEXT}，"
                                f"莊家 {SEAT_NAMES[previous]} → {SEAT_NAMES[dealer]}")
        return messages
