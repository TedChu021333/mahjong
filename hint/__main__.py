"""提示模式入口。

    python -m hint                         # 即時擷取螢幕（遊戲最大化、1920x1080）
    python -m hint --image train/吃牌畫面.png
    python -m hint --video train/遊戲流程.mp4 --step 1
    python -m hint --no-overlay            # 建議只印在終端機，不顯示浮動小視窗
    python -m hint --auto                  # 自動點擊，只在對手全是電腦的訓練場使用
    python -m hint --auto --ai ev          # 改用期望值打牌 AI（會估計各張牌的放槍風險）

連續兩幀辨識結果相同才給建議（避開理牌、摸牌等動畫），建議改變時才印出。
自動模式：同一個畫面只操作一次；點完 RETRY_AFTER 秒畫面沒變（例如出牌要點兩下）
才補點，最多 MAX_RETRIES 次。讀不到手牌時檢查打完後的「繼續」類按鈕（小結算繼續、再贏一局），
同樣連續兩幀一致才點；不再盲點任何座標（打一圈時按完「繼續」會直接發下一局，
盲點會點到手牌）。每局結算畫面在按「繼續」前存到 result/。
滑鼠甩到螢幕左上角會立刻中止。
"""
from __future__ import annotations

import argparse
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from game.rules import ActionType, forbidden_after_claim
from game.tiles import tile_code, tile_name
from hint.advisor import (
    POLICIES,
    STREAK_UNKNOWN_GUESS,
    Readers,
    decide,
    describe,
    danger_table,
    hand_after_claim,
    last_note,
    observe,
    use_network,
    use_policy,
)
from hint.hand_end import HandBook
from hint.meld_check import MeldCheck
from hint.river_watch import RiverWatch
from hint.river_watch import summary as river_summary
from perception.continue_button import ContinueButton
from perception.river_reader import RiverReader
from perception.capture import grab_screen, read_image, write_image
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CONTINUE_BUTTONS,
)
from perception.table_reader import autoplay_active, read_buttons

RETRY_AFTER = 1.5
MAX_RETRIES = 2
CLAIM_PROMPT_REPORT = 1.5
RIVER_INTERVAL = 0.3
"""牌河追蹤的門檻以幀數計、依約 0.3 秒一幀調校；擷取變快後牌河仍照這個間隔更新。"""
UNREADABLE_AFTER = 6.0
OWN_TAKEN_LIMIT = 20
"""自己的牌被判定吃碰拿走時存畫面，每次執行最多幾張（查是不是誤判）。"""
OWN_MISSING_LIMIT = 20
"""自己打出的牌約 5 秒牌河都沒看到時存畫面，每次執行最多幾張（10/21 約 5.5% 原因不明）。"""
MELD_SHOT_LIMIT = 150
"""收集對手副露的截圖（10/22 起，改善 perception.meld_reader 的槓與側躺的萬子）：每次執行最多幾張。"""
MELD_SHOT_DELAY = 1.5
"""有人吃碰後等副露擺好再存。"""
MELD_SHOT_INTERVAL = 8.0
"""場上有人有副露時，每隔幾秒存一張（加槓、暗槓不一定有吃碰事件）。"""
TABLE_INTERVAL = 1.0
"""別人的回合時放槍表最多幾秒重算一次（每次約 0.1 秒）。"""
END_SHOT_DELAYS = (1.0, 2.0, 4.0)
"""一局快打完（牌河記到 END_SHOT_MIN_DISCARDS 張以上）卻讀不到手牌、也還沒認到結算時，在這些秒數各存一張
「局末_」截圖：流局沒有結算畫面，要看清楚那時畫面上是什麼（10/19 只有 6 秒後的「讀不到」，已經在擲骰子）。"""
END_SHOT_MIN_DISCARDS = 40
END_SHOT_LIMIT = 45
"""每次執行最多存幾張局末截圖。"""
"""牌局中連續這麼多秒讀不到手牌（又不是聽牌代打）就存一張截圖，查版面認不出的原因。"""
CLAIM_MEMORY = 10.0
"""能吃碰時放大的牌可能先縮進牌河、按鈕卻還亮著；這段時間內沿用最近讀到的那張。"""
RESULT_DIR = Path("result")
"""每局結算畫面的截圖。"""


def save_result(frame: np.ndarray, prefix: str = "") -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = RESULT_DIR / f"{prefix}{stamp}.png"
    count = 1
    while path.exists():  # 小結算和大結算常在同一秒，不能互相覆蓋
        count += 1
        path = RESULT_DIR / f"{prefix}{stamp}_{count}.png"
    write_image(path, frame)
    return path


BANNER_COOLDOWN = 3.0
"""同一次吃碰只存一次。"""
BANNER_DELAYS = (0.6, 1.0)
"""吃碰時吃碰者那邊會出現「吃」（紫）／「碰」（綠灰）大字約 1.5 秒，牌河要確認幾幀才報「被拿走」，
所以存這麼多秒之前的畫面（收集起來做吃／碰與吃碰者的辨識）。"""
RECENT_SECONDS = 2.0
GAME_LOST_SECONDS = 60.0
"""自動模式看不到遊戲這麼久、使用者也這麼久沒操作電腦，就把遊戲切回前面
（10/3 03:03 VS Code 跑到遊戲前面，停了 8 小時）。"""
STUCK_IDLE_SECONDS = 5.0
"""點了幾次都沒反應時，遊戲不在前景、使用者也這麼久沒操作，就把遊戲切到前面再試
（10/3 02:23～03:03 出牌點了沒反應 9 次）。"""
UNKNOWN_SCREEN_SECONDS = 15.0
"""這麼久認不出畫面（不是手牌、按鈕、結算、代打）就存一張截圖：10/4 06:09 跳出升級畫面後停了 4 小時，
當時沒有截圖可以做辨識。"""
STUCK_SHOTS = 2
STUCK_SHOT_INTERVAL = 0.5
"""卡住時除了當下那張，再每 0.5 秒多存 2 張（查 10/3 出牌點了沒反應 34 次的原因）。"""
SAVE_CLAIM_FRAMES = False
"""收集吃碰動畫截圖（10/2 一天存了 1116 張、2.7GB，夠用了，預設關閉）。"""


def print_log(line: str, detail: str | None = None, echo: bool = True) -> None:
    if echo:
        print(line, flush=True)


class GameLog:
    """即時模式的記錄：終端機只印建議，記錄檔另外附上當時的手牌與金牌，方便事後對照結算截圖。"""

    def __init__(self, directory: Path = RESULT_DIR) -> None:
        self.path = directory / f"log_{time.strftime('%Y%m%d')}.txt"

    def __call__(self, line: str, detail: str | None = None, echo: bool = True) -> None:
        """echo=False 只寫進記錄檔（例如牌河事件，終端機不必每張都印）。"""
        if echo:
            print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"{time.strftime('%H:%M:%S')} {line}")
            file.write(f"　（{detail}）\n" if detail else "\n")


def hand_text(observation) -> str:
    text = "手牌 " + "".join(tile_name(tile) for tile in observation.hand)
    if observation.gold_tiles:
        text += "，金牌 " + "".join(tile_name(tile) for tile in observation.gold_tiles)
    if observation.buttons:
        text += "，按鈕 " + "、".join(sorted(observation.buttons))
    claim = observation.claim
    if claim is not None and claim.tile is not None and not observation.my_turn:
        # 事後分析吃碰（包括按「不要」的）要知道是哪張牌、誰打的（10/20 起）
        text += f"，可吃碰 {tile_name(claim.tile)}（{CLAIM_SOURCE_NAMES.get(claim.source, claim.source)}）"
    if observation.declared:
        # 事後分析放槍用：放給已宣告者的牌，要知道 AI 當下有沒有認到宣告（eval.deal_in）
        text += "，宣告 " + "、".join(DECLARED_NAMES[seat] for seat in sorted(observation.declared))
    return text


DECLARED_NAMES = {"left": "上家", "top": "對家", "right": "下家"}
CLAIM_SOURCE_NAMES = DECLARED_NAMES


def screen_frames(interval: float) -> Iterator[tuple[str, np.ndarray]]:
    while True:
        yield time.strftime("%H:%M:%S"), grab_screen()
        time.sleep(interval)


def video_frames(path: Path, step: float, start: float) -> Iterator[tuple[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"無法開啟影片：{path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(step * fps))
    index = 0
    while capture.grab():
        if index >= start * fps and index % every == 0:
            ok, frame = capture.retrieve()
            if ok:
                seconds = int(index / fps)
                yield f"{seconds // 60:02d}:{seconds % 60:02d}", frame
        index += 1


def run(frames: Iterator[tuple[str, np.ndarray]], readers: Readers,
        actuator=None, observe_fn=observe, clock=time.monotonic, show=None,
        find_continue=None, save_result=save_result, log=print_log, rivers=None,
        find_win=None, find_buttons=None, find_popup=None, find_autoplay=None,
        window=None, show_danger=None, stop_requested=None) -> None:
    """show_danger(text) 更新放槍表小視窗（輪到自己時手上放槍機率最高的幾張）。
    find_win(frame) 為「胡」按鈕是否亮著、find_buttons(frame) 讀按鈕列（都不依賴手牌辨識）；
    find_popup(frame) 找會蓋住牌桌的彈出面板（自動模式按關閉）。window（control.window.GameWindow）
    讓自動模式在看不到遊戲、使用者又沒在用電腦時把遊戲切回前面。"""
    previous = None
    last_advice = None
    acted_for = None
    acted_at = 0.0
    retries = 0
    pending_claim = None
    """最近一次建議的吃碰：(吃碰後應有的手牌, 不能打的牌, 用掉的手牌)。"""
    missed_for = None
    """已經存過「胡按鈕亮著卻沒選胡」截圖的畫面。"""
    last_claim = None
    """(最近讀到的放大牌, 時間)。"""
    decided_for = decided_action = decided_note = None
    stuck_for = None
    win_pressed_at = float("-inf")
    river_updated_at = float("-inf")
    claim_prompt_since = None
    popup_closed_at = float("-inf")
    banner_saved_at = float("-inf")
    recent = deque()
    """最近幾秒的畫面（牌河更新時記一張），吃碰時存下之前的畫面。"""
    claim_prompt_reported = False
    last_readable = None
    """最近一次讀到手牌的時間；存過「讀不到」截圖後設為 None，直到再讀到手牌。"""
    streak_reported = False
    game_seen_at = clock()
    """最近一次認出遊戲畫面（手牌、按鈕、結算、代打）的時間。"""
    focus_tried_at = float("-inf")
    unknown_saved_for = None
    """已經存過「不明畫面」截圖的那段期間（以 game_seen_at 區分）。"""
    stuck_shots_left, stuck_shot_at = 0, float("-inf")
    hands = HandBook()
    own_taken_saved = 0
    meld_shots, meld_shot_due, meld_shot_at = 0, None, clock()
    own_missing_saved = 0
    meld_check = MeldCheck()
    meld_rejected = None
    """最近一次記錄過「副露數不合理」的 (副露數, 手牌)，同一個畫面只記一次。"""
    declared_seen: set[str] = set()
    """這局已經記錄過宣告的對手。"""
    stop_after_round = False
    """按過 Esc：看到整圈結束的畫面（大結算等）就停，不按繼續。"""
    table_for = None
    """放槍表算過的局面（手牌、各家牌河張數、宣告者），沒變就不重算。"""
    table_at = float("-inf")

    def update_table(observation) -> None:
        nonlocal table_for, table_at
        if show_danger is None:
            return
        key = (observation.hand, tuple(len(river) for river in observation.discards),
               observation.declared)
        # 自己的回合手牌一變就算；別人的回合最多每 TABLE_INTERVAL 秒算一次
        if key == table_for or (not observation.my_turn and clock() - table_at < TABLE_INTERVAL):
            return
        table_for, table_at = key, clock()
        try:
            text = danger_table(observation)
        except ValueError:  # 局面不合法（例如張數讀錯）：保留上一次的表
            return
        if text is not None:
            show_danger(text)

    hand_seen_at = None
    """最近一次讀到手牌的時間（局末截圖用）。"""
    end_shots: set[float] = set()
    """這段讀不到的期間已經存過的局末截圖（秒數）。"""
    end_shots_saved = 0
    continue_seen_at = float("-inf")
    """卡住後還要存幾張後續截圖（每 STUCK_SHOT_INTERVAL 秒一張），看點擊後畫面怎麼變。"""
    for stamp, frame in frames:
        if stop_requested is not None and stop_requested():
            if stop_after_round:
                log(f"[{stamp}] 又按了 Esc，立刻停止")
                return
            # 10/21 使用者：按 Esc 改成打完這圈再停（直接停會丟下打到一半的牌局）
            stop_after_round = True
            log(f"[{stamp}] 按了 Esc，打完這圈後停止（要立刻停再按一次 Esc）")
            if show is not None:
                show("打完這圈後停止")
        try:
            if stuck_shots_left and clock() - stuck_shot_at >= STUCK_SHOT_INTERVAL:
                stuck_shots_left -= 1
                stuck_shot_at = clock()
                log(f"[{stamp}] 卡住後續截圖存到 {save_result(frame, '卡住_後續_')}", echo=False)
            if clock() - game_seen_at >= UNKNOWN_SCREEN_SECONDS \
                    and unknown_saved_for != game_seen_at:
                unknown_saved_for = game_seen_at  # 每段認不出的期間只存一張
                log(f"[{stamp}] {UNKNOWN_SCREEN_SECONDS:.0f} 秒認不出畫面（不是手牌、按鈕或結算），"
                    f"截圖存到 {save_result(frame, '不明畫面_')}")
            if (actuator is not None and window is not None
                    and clock() - game_seen_at >= GAME_LOST_SECONDS
                    and clock() - focus_tried_at >= GAME_LOST_SECONDS
                    and window.idle_seconds() >= GAME_LOST_SECONDS
                    and window.is_foreground() is False):
                focus_tried_at = clock()
                log(f"[{stamp}] {GAME_LOST_SECONDS:.0f} 秒看不到遊戲、也沒人操作電腦，"
                    f"把遊戲切回前面（{'成功' if window.focus() else '失敗'}）")
            if (actuator is not None and find_popup is not None
                    and clock() - popup_closed_at >= RETRY_AFTER):
                popup = find_popup(frame)
                if popup is not None:
                    game_seen_at = popup_closed_at = clock()
                    log(f"[{stamp}] 關閉「{popup.name}」面板", echo=False)
                    actuator.click(popup.center)
            observation = observe_fn(frame, readers)
            if observation is not None and not meld_check.accept(observation.meld_count,
                                                                 observation.buttons, clock()):
                if meld_rejected != (observation.meld_count, observation.hand):
                    meld_rejected = (observation.meld_count, observation.hand)
                    log(f"[{stamp}] 手牌只讀到 {len(observation.hand)} 張（副露 {meld_check.count} → "
                        f"{observation.meld_count} 組不合理，可能被動畫蓋住），不採用", echo=False)
                observation = None
            if observation is not None:
                if observation.claim is not None and observation.claim.tile is not None:
                    last_claim = (observation.claim, clock())
                elif observation.buttons is not None and not observation.my_turn \
                        and last_claim is not None and clock() - last_claim[1] <= CLAIM_MEMORY:
                    observation = replace(observation, claim=last_claim[0])
            if (rivers is not None and observation is not None
                    and clock() - river_updated_at >= RIVER_INTERVAL):
                river_updated_at = clock()
                recent.append((clock(), frame))
                while recent and clock() - recent[0][0] > RECENT_SECONDS:
                    recent.popleft()
                if observation.swap_prompt is not None and rivers.has_discards():
                    # 換三張＝新的一局。流局沒有小結算，牌河不會在按結算時重設（10/19 15:23 連兩局沒有結算，
                    # 三局的牌河疊在一起、牌牆估計剩 0 張，開局就當成最後一手在防守，拆了四五筒）
                    log(f"[{stamp}] 新的一局（上一局沒有結算畫面），重設牌河："
                        f"{river_summary(rivers)}", echo=False)
                    rivers.new_hand()
                    hands.hand_ended(settled=False)
                    declared_seen = set()
                resets = getattr(rivers, "reset_by_empty_table", 0)
                events = rivers.update(frame, observation)
                if getattr(rivers, "reset_by_empty_table", 0) != resets:
                    log(f"[{stamp}] 牌河整個清空，當成新的一局、重設牌河", echo=False)
                    hands.hand_ended(settled=False)
                    declared_seen = set()
                for event in events:
                    name = tile_name(event.tile) if event.tile is not None else "?"
                    log(f"[{stamp}] 牌河 {event.seat} {event.kind} {name}", echo=False)
                    if event.kind in ("claimed", "claimed_unseen"):
                        meld_shot_due = clock() + MELD_SHOT_DELAY
                    if event.seat == "me" and event.kind == "claimed" and own_taken_saved < OWN_TAKEN_LIMIT:
                        # 10/20 18:36～19:34 自己的牌被判定拿走 32 次、只還原 6 次，太多：存畫面查是真的吃碰還是被蓋住
                        own_taken_saved += 1
                        log(f"[{stamp}] 自己的牌被判定拿走，畫面存到 "
                            f"{save_result(frame, '自己的牌被拿走_')}", echo=False)
                    if SAVE_CLAIM_FRAMES and event.kind == "claimed" \
                            and clock() - banner_saved_at >= BANNER_COOLDOWN:
                        banner_saved_at = clock()
                        for delay in BANNER_DELAYS:
                            _, earlier = min(recent, key=lambda item: abs(clock() - delay - item[0]))
                            code = tile_code(event.tile) if event.tile is not None else "x"
                            path = save_result(earlier, f"吃碰動畫_{event.seat}_{code}_")
                            log(f"[{stamp}] 有人吃碰，{delay} 秒前的畫面存到 {path}", echo=False)
                overdue = getattr(rivers, "overdue_own_discard", None)
                missing, tile = overdue() if overdue is not None else (False, None)
                if missing and own_missing_saved < OWN_MISSING_LIMIT:
                    own_missing_saved += 1
                    name = tile_name(tile) if tile is not None else "?"
                    log(f"[{stamp}] 自己打的 {name} 約 5 秒牌河都沒看到，畫面存到 "
                        f"{save_result(frame, '自己的牌沒進牌河_')}", echo=False)
                counts = getattr(rivers, "meld_counts", None)
                has_melds = counts is not None and any(counts())
                due = meld_shot_due is not None and clock() >= meld_shot_due
                if meld_shots < MELD_SHOT_LIMIT and (due or (has_melds and clock() - meld_shot_at >= MELD_SHOT_INTERVAL)):
                    meld_shots += 1
                    meld_shot_due, meld_shot_at = None, clock()
                    log(f"[{stamp}] 副露截圖存到 {save_result(frame, '副露_')}", echo=False)
            if rivers is not None and observation is not None:
                observation = replace(observation, discards=rivers.discards(),
                                      melds=rivers.meld_counts(),
                                      exposed=rivers.exposed_tiles(),
                                      my_discards=rivers.my_discards,
                                      declared_safe=rivers.declared_safe(),
                                      my_pungs=tuple(rivers.my_pungs))
            if observation is not None and observation.declared - declared_seen:
                # 對手什麼時候聽牌：第一次認到「聽」時各家牌河已有幾張（10/20 起，量遊戲電腦的習慣用）
                sizes = "、".join(f"{name} {len(river)}" for name, river in
                                  zip(("自己", "下家", "對家", "上家"), observation.discards))
                for seat in sorted(observation.declared - declared_seen):
                    log(f"[{stamp}] {DECLARED_NAMES[seat]}宣告聽牌（第一次認到；牌河 {sizes}）", echo=False)
                declared_seen |= observation.declared
            if observation is not None:
                game_seen_at = last_readable = hand_seen_at = clock()
                end_shots.clear()
                for message in hands.see_dealer(observation.dealer):
                    log(f"[{stamp}] {message}", echo=False)
                if observation.dealer_streak is None and not streak_reported:
                    streak_reported = True  # 每次執行只存一張，補模板用
                    log(f"[{stamp}] 認不出連莊數（先當 {STREAK_UNKNOWN_GUESS}），截圖存到 "
                        f"{save_result(frame, '連莊_')}")
            elif find_autoplay is not None and find_autoplay(frame):
                game_seen_at = last_readable = clock()  # 聽牌後遊戲代打：「取消代打」蓋住手牌是正常的
            if (observation is None and hand_seen_at is not None and rivers is not None
                    and continue_seen_at < hand_seen_at and end_shots_saved < END_SHOT_LIMIT
                    and rivers.total_discards() >= END_SHOT_MIN_DISCARDS):
                for delay in END_SHOT_DELAYS:
                    if delay not in end_shots and clock() - hand_seen_at >= delay:
                        end_shots.add(delay)
                        end_shots_saved += 1
                        log(f"[{stamp}] 局末讀不到手牌、還沒有結算畫面（{delay:.0f} 秒），截圖存到 "
                            f"{save_result(frame, '局末_')}", echo=False)
                        break
            elif (last_readable is not None and clock() - last_readable >= UNREADABLE_AFTER
                  and not (decided_action is not None
                           and decided_action.kind == ActionType.DECLARE)):
                last_readable = None  # 每段讀不到的期間只存一張
                unknown_saved_for = game_seen_at  # 已經有截圖，不必再存「不明畫面」
                log(f"[{stamp}] 連續 {UNREADABLE_AFTER:.0f} 秒讀不到手牌，截圖存到 "
                    f"{save_result(frame, '讀不到_')}")
            # 吃碰槓按鈕亮著卻一直沒處理：存截圖查原因（手牌變暗讀不到、放大的牌讀不到…）
            lit = observation.buttons if observation is not None else \
                (find_buttons(frame) if find_buttons is not None else None)
            # 自己的回合「槓」亮著（能暗槓／加槓，AI 決定不槓就照常打牌）、選吃法的選項框出現時都不是
            # 等著回應的吃碰（10/19 三張「吃碰未處理」都是這兩種）
            responding = observation is None or (not observation.my_turn
                                                 and not observation.chow_panels)
            if lit and lit & {"chow", "pung", "kong"} and responding:
                if claim_prompt_since is None:
                    claim_prompt_since, claim_prompt_reported = clock(), False
                handled = observation is not None and decided_for == observation \
                    and decided_action is not None
                if not handled and not claim_prompt_reported \
                        and clock() - claim_prompt_since >= CLAIM_PROMPT_REPORT:
                    claim_prompt_reported = True
                    reason = "讀不到手牌" if observation is None else \
                        "放大的牌讀不到" if observation.claim is None else "沒有建議"
                    log(f"[{stamp}] 吃碰按鈕亮了 {CLAIM_PROMPT_REPORT} 秒還沒處理（{reason}），"
                        f"截圖存到 {save_result(frame, '吃碰未處理_')}")
            else:
                claim_prompt_since = None
            if observation is None and find_win is not None and find_win(frame):
                game_seen_at = clock()
                # 摸牌的手、出牌動畫常蓋住手牌，而且倒數只剩一兩秒：讀不到手牌也照樣胡
                # （曾因此錯過自摸，遊戲還會顯示「您剛剛錯過了胡牌時機」）
                if clock() - win_pressed_at >= RETRY_AFTER:
                    win_pressed_at = clock()
                    log(f"[{stamp}] 胡！（手牌被動畫擋住，直接按「胡」）")
                    if show is not None:
                        show("胡！")
                    if actuator is not None:
                        actuator.click(ACTION_BUTTON_CENTERS["win"])
                continue
            if observation is None and find_continue is not None:
                observation = find_continue(frame)
                if observation is not None:
                    game_seen_at = clock()
                    last_readable = None  # 這局結束了：結算、配桌期間讀不到手牌是正常的
            if observation is None or observation != previous:
                previous = observation
                continue
            if isinstance(observation, ContinueButton):
                continue_seen_at = clock()
                if observation == acted_for:
                    if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                        continue
                    retries += 1
                else:
                    retries = 0
                    log(f"[{stamp}] 結算截圖存到 {save_result(frame)}")
                    if rivers is not None:
                        log(f"[{stamp}] 本局牌河：{river_summary(rivers)}"
                            f"（累計自動收集 {rivers.saved} 張牌河模板、"
                            f"{getattr(rivers, 'rejected_labels', 0)} 次標註和牌不像沒採用）", echo=False)
                        rivers.new_hand()
                    hands.hand_ended(settled=True)
                    declared_seen = set()
                if stop_after_round and observation.name in ROUND_END_SCREENS:
                    log(f"[{stamp}] 這圈打完了（{observation.name}），程式停止")
                    return
                acted_for, acted_at = observation, clock()
                if actuator is None:  # 提示模式：只記錄，由人按繼續
                    retries = MAX_RETRIES
                    continue
                if retries == 0:
                    log(f"[{stamp}] 按「{observation.name}」")
                actuator.click(observation.center)
                continue
            if actuator is not None and actuator.follow_up(observation):
                log(f"[{stamp}] 選擇吃法")
                acted_for, acted_at, retries = observation, clock(), 0
                continue
            if observation.my_turn and pending_claim is not None:
                expected, forbidden, used = pending_claim
                if tuple(sorted(observation.hand)) == expected:
                    if rivers is not None and used is not None:
                        rivers.own_claim(used)  # 每次吃碰只記一次
                        observation = replace(observation, exposed=rivers.exposed_tiles())
                        pending_claim = (expected, forbidden, None)
                    observation = replace(observation, claimed=True, forbidden=forbidden)
                else:
                    pending_claim = None
            # 同一個畫面只決定一次：蒙地卡羅 AI 每次要約 2 秒且有隨機性，重算會讓建議閃動
            if observation != decided_for:
                decided_for, decided_action = observation, decide(observation)
                decided_note = last_note()
            action = decided_action
            if action is not None and action.kind in (ActionType.CHOW, ActionType.PUNG):
                used = list(action.tiles)
                used.remove(action.tile)
                pending_claim = (hand_after_claim(observation, action),
                                 forbidden_after_claim(action, action.tile), tuple(used))
            advice = describe(action) if action is not None else None
            buttons = observation.buttons or frozenset()
            # 吃法選項框會蓋在「胡」按鈕上被誤認成亮著（10/3 20:58 誤報），選吃法時不算
            if "win" in buttons and (action is None or action.kind != ActionType.WIN) \
                    and not observation.chow_panels and observation != missed_for:
                missed_for = observation
                log(f"[{stamp}] 「胡」按鈕亮著但 AI 沒選胡，截圖存到 {save_result(frame, '未胡_')}",
                    hand_text(observation))
            if advice != last_advice:
                if advice is not None:
                    note = f"（{decided_note}）" if decided_note else ""
                    log(f"[{stamp}] {advice}{note}", hand_text(observation))
                if show is not None:
                    show(advice)
            last_advice = advice
            # 放槍表一直亮著：別人的回合也算（現在這 16 張給下一手參考），吃碰按鈕亮著時先不算、不拖慢回應
            if (not observation.my_turn and not observation.buttons) or                     (observation.my_turn and (actuator is None or action is None)):
                update_table(observation)
            if actuator is None or action is None:
                continue
            if observation == acted_for:
                if retries >= MAX_RETRIES and clock() - acted_at >= RETRY_AFTER \
                        and stuck_for != observation:
                    stuck_for = observation  # 點了幾次畫面都沒變：存下來查原因（例如槓完沒反應）
                    log(f"[{stamp}] 「{advice}」點了 {MAX_RETRIES + 1} 次畫面都沒變，截圖存到 "
                        f"{save_result(frame, '卡住_')}", hand_text(observation))
                    stuck_shots_left, stuck_shot_at = STUCK_SHOTS, clock()
                    if window is not None and window.is_foreground() is False \
                            and window.idle_seconds() >= STUCK_IDLE_SECONDS:
                        # 遊戲不在前景時點擊可能沒送進去：切到前面、再點一輪
                        log(f"[{stamp}] 遊戲不在最前面，切回前面再試"
                            f"（{'成功' if window.focus() else '失敗'}）")
                        retries = 0
                if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                    continue
                retries += 1
            else:
                retries = 0
            if not actuator.perform(action, observation):
                log(f"[{stamp}] 畫面上找不到對應的按鈕或牌，略過", hand_text(observation))
            acted_for = observation
            acted_at = clock()
            if observation.my_turn:
                update_table(observation)  # 自動模式：點完再算，不拖慢出牌
        except Exception as error:  # 單一幀出錯不能讓整個迴圈停掉（浮動小視窗模式下看起來就像不動作）
            if type(error).__name__ == "FailSafeException":
                raise  # 滑鼠甩到左上角的緊急停止一定要生效
            detail = traceback.format_exc()
            try:
                where = save_result(frame, "錯誤_")
            except Exception:
                where = "（截圖失敗）"
            log(f"[{stamp}] 程式錯誤，已略過這一幀，截圖存到 {where}", detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="麻將提示模式")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--step", type=float, default=0.5, help="影片每隔幾秒取一幀")
    parser.add_argument("--start", type=float, default=0.0, help="影片從第幾秒開始")
    parser.add_argument("--interval", type=float, default=0.1, help="即時模式擷取間隔（秒）")
    parser.add_argument("--auto", action="store_true",
                        help="自動點擊（只在對手全是電腦的訓練場使用；滑鼠甩到左上角中止）")
    parser.add_argument("--no-overlay", action="store_true", help="不顯示浮動小視窗")
    parser.add_argument("--ai", choices=POLICIES, default="rule",
                        help="打牌 AI：rule 規則式（預設）、ev 期望值打牌（估計放槍風險）、"
                             "mc 蒙地卡羅（模擬比較，每步約 2 秒）、nn 學習型（需要 PyTorch，"
                             "用 .venv-train/Scripts/python 執行）")
    parser.add_argument("--model", default="models/rl3_0960.pt", help="--ai nn 用的網路權重")
    parser.add_argument("--efficiency-turns", type=int, default=0,
                        help="--ai nn 前幾手湊牌優先（同向聽時改打進張最多的牌）；0 = 照網路")
    parser.add_argument("--attack", action="store_true",
                        help="--ai nn 極致進攻：整局湊牌優先、有人宣告時只擋放槍超過 25% 的牌、進張變多就吃碰")
    parser.add_argument("--raw-network", action="store_true",
                        help="--ai nn 完全照網路（不限制按聽、有人宣告時不檢查放槍）")
    parser.add_argument("--workers", type=int, default=4,
                        help="蒙地卡羅平行模擬的程序數（預設 4；之前用 CPU 核心數 − 2＝14 個，"
                             "每手出牌時整台電腦滿載，10/3 出牌點了沒反應 34 次）")
    parser.add_argument("--move-seconds", type=float, default=3.0,
                        help="遊戲每手限時（秒）；3 秒場模擬約 1 秒，6 秒場給 6，模擬約 4 秒")
    parser.add_argument("--demo-overlay", action="store_true",
                        help="只展示小視窗外觀（可拖曳調整位置），不辨識畫面")
    args = parser.parse_args()
    use_policy(args.ai)
    if args.ai == "nn":
        use_network(args.model, args.efficiency_turns, guard=not args.raw_network, attack=args.attack)
    if args.ai == "mc":
        import os
        from multiprocessing import Pool

        from hint.advisor import use_worker_pool

        from hint.advisor import use_move_seconds

        workers = max(1, min(args.workers, (os.cpu_count() or 2) - 2))
        use_worker_pool(Pool(workers), workers)
        use_move_seconds(args.move_seconds)
        print(f"蒙地卡羅 AI：{workers} 個程序平行模擬，每手限時 {args.move_seconds:g} 秒",
              flush=True)
    if args.demo_overlay:
        from hint.overlay import demo

        demo()
        return
    if args.image:
        observation = observe(read_image(args.image), Readers())
        action = decide(observation) if observation else None
        print(describe(action) if action else "畫面無法辨識或不需要做決定")
    elif args.video:
        run(video_frames(args.video, args.step, args.start), Readers())
    else:
        live(args.interval, overlay=not args.no_overlay, auto=args.auto)


def confirm_training_room(ask=input) -> bool:
    """經典版一般場是與真人配桌，自動點擊只能用在對手全是電腦的訓練場。"""
    print("自動模式只能在「訓練場」（三家對手都是電腦）使用；與真人配桌的場請用提示模式。")
    answer = ask("確認目前是對手全為電腦的訓練場？輸入 y 繼續：")
    return answer.strip().lower() == "y"


def live(interval: float, overlay: bool, auto: bool) -> None:
    actuator = None
    if auto:
        if not confirm_training_room():
            raise SystemExit("已取消自動模式。")
        from control.actuator import Actuator

        actuator = Actuator()
        print("自動模式 3 秒後開始，請切回遊戲視窗。按 Esc 會打完這圈再停（再按一次立刻停），滑鼠甩到螢幕左上角可立即中止。", flush=True)
        time.sleep(3)
    mode = "自動模式" if auto else "提示模式"
    readers = Readers()
    from perception.continue_button import ContinueButtons, PopupClosers

    # 提示模式也偵測結算畫面：存結算截圖、重置牌河；只有自動模式會按
    buttons = ContinueButtons.from_directory()
    missing = [name for name in CONTINUE_BUTTONS if name not in buttons.references]
    if missing:
        print(f"缺少按鈕參考圖，認不出結算畫面：{'、'.join(missing)}", flush=True)
    find_continue = buttons.find
    popups = PopupClosers.from_directory()
    log = GameLog()
    print(f"記錄檔：{log.path}", flush=True)
    from hint import advisor

    log(f"[{time.strftime('%H:%M:%S')}] {mode}啟動，AI {advisor._policy}，{program_version()}"
        f"{network_options(advisor._network)}", echo=False)
    rivers = RiverWatch(RiverReader.from_directory())
    esc_pressed()  # 清掉啟動前按過 Esc 的紀錄，免得一開始就停
    game_window = None
    if auto:
        from control.window import GameWindow

        game_window = GameWindow()  # 看不到遊戲、又沒人用電腦時把遊戲切回前面
    if not overlay:
        print(f"{mode}啟動，按 Esc 打完這圈後結束（再按一次立刻結束），或 Ctrl+C", flush=True)
        run(screen_frames(interval), readers, actuator=actuator, find_continue=find_continue,
            log=log, rivers=rivers, find_win=win_is_lit, find_buttons=read_buttons,
            find_popup=popups.find, find_autoplay=autoplay_active, window=game_window,
            stop_requested=esc_pressed)
        return
    from hint.overlay import Overlay

    print(f"{mode}啟動（浮動小視窗），按 Esc 打完這圈後結束（再按一次立刻結束），或 Ctrl+C", flush=True)
    window = Overlay()
    window.run(lambda: run(screen_frames(interval), readers, actuator=actuator,
                           show=window.show, find_continue=find_continue, log=log,
                           rivers=rivers, find_win=win_is_lit, find_buttons=read_buttons,
                           find_popup=popups.find, find_autoplay=autoplay_active,
                           window=game_window, show_danger=window.show_table,
                           stop_requested=esc_pressed))


ROUND_END_SCREENS = frozenset({"大結算", "繼續戰鬥", "破產再玩一局", "再拚一局", "再玩一局"})
"""整圈結束才會出現的畫面與按鈕（「再贏一局」是流局後的下一局，不算）。"""


_esc_down = False


def esc_pressed() -> bool:
    """Esc 剛被按下嗎（不管哪個視窗在前面）：GetAsyncKeyState 的最高位元＝現在按著，
    最低位元＝上次查詢之後按過（擷取一幀約 0.1 秒，短按也抓得到）。10/20 使用者要求按 Esc 停止程式。
    按著不放只算一次：第二次按 Esc 是「立刻停」，按著不放不能被當成按了兩次。"""
    import ctypes

    global _esc_down
    state = ctypes.windll.user32.GetAsyncKeyState(0x1B)
    down = bool(state & 0x8000)
    pressed = (down and not _esc_down) or (bool(state & 1) and not down)
    _esc_down = down
    return pressed


def program_version() -> str:
    """記錄檔用：目前的 commit 與有沒有未提交的修改，統計時才能分開不同版本（10/20 起）。"""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "版本不明"
    return f"版本 {commit or '不明'}{'（有未提交的修改）' if dirty else ''}"


def network_options(network) -> str:
    if network is None:
        return ""
    return (f"，湊牌優先 {network.efficiency_turns} 手"
            f"，護欄{'開' if getattr(network, 'guard', False) else '關'}"
            f"{'，極致進攻' if getattr(network, 'attack', False) else ''}")


def win_is_lit(frame: np.ndarray) -> bool:
    buttons = read_buttons(frame)
    return buttons is not None and "win" in buttons


if __name__ == "__main__":
    main()
