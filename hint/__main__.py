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
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from game.rules import ActionType, forbidden_after_claim
from game.tiles import tile_name
from hint.advisor import (
    POLICIES,
    STREAK_UNKNOWN_GUESS,
    Readers,
    decide,
    describe,
    hand_after_claim,
    observe,
    use_policy,
)
from hint.river_watch import RiverWatch
from hint.river_watch import summary as river_summary
from perception.continue_button import ContinueButton
from perception.river_reader import RiverReader
from perception.capture import grab_screen, read_image, write_image
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CONTINUE_BUTTONS,
)
from perception.table_reader import read_buttons

RETRY_AFTER = 1.5
MAX_RETRIES = 2
CLAIM_PROMPT_REPORT = 1.5
RIVER_INTERVAL = 0.3
"""牌河追蹤的門檻以幀數計、依約 0.3 秒一幀調校；擷取變快後牌河仍照這個間隔更新。"""
UNREADABLE_AFTER = 6.0
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
    return text


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
        find_win=None, find_buttons=None, find_popup=None) -> None:
    """find_win(frame) 為「胡」按鈕是否亮著、find_buttons(frame) 讀按鈕列（都不依賴手牌辨識）；
    find_popup(frame) 找會蓋住牌桌的彈出面板（自動模式按關閉）。"""
    previous = None
    last_advice = None
    acted_for = None
    acted_at = 0.0
    retries = 0
    pending_claim = None
    """最近一次建議的吃碰：(吃碰後應有的手牌, 不能打的牌)。"""
    missed_for = None
    """已經存過「胡按鈕亮著卻沒選胡」截圖的畫面。"""
    last_claim = None
    """(最近讀到的放大牌, 時間)。"""
    decided_for = decided_action = None
    stuck_for = None
    win_pressed_at = float("-inf")
    river_updated_at = float("-inf")
    claim_prompt_since = None
    popup_closed_at = float("-inf")
    claim_prompt_reported = False
    last_readable = None
    """最近一次讀到手牌的時間；存過「讀不到」截圖後設為 None，直到再讀到手牌。"""
    streak_reported = False
    for stamp, frame in frames:
        try:
            if (actuator is not None and find_popup is not None
                    and clock() - popup_closed_at >= RETRY_AFTER):
                popup = find_popup(frame)
                if popup is not None:
                    popup_closed_at = clock()
                    log(f"[{stamp}] 關閉「{popup.name}」面板", echo=False)
                    actuator.click(popup.center)
            observation = observe_fn(frame, readers)
            if observation is not None:
                if observation.claim is not None and observation.claim.tile is not None:
                    last_claim = (observation.claim, clock())
                elif observation.buttons is not None and not observation.my_turn \
                        and last_claim is not None and clock() - last_claim[1] <= CLAIM_MEMORY:
                    observation = replace(observation, claim=last_claim[0])
            if (rivers is not None and observation is not None
                    and clock() - river_updated_at >= RIVER_INTERVAL):
                river_updated_at = clock()
                for event in rivers.update(frame, observation):
                    name = tile_name(event.tile) if event.tile is not None else "?"
                    log(f"[{stamp}] 牌河 {event.seat} {event.kind} {name}", echo=False)
            if rivers is not None and observation is not None:
                observation = replace(observation, discards=rivers.discards(),
                                      melds=rivers.meld_counts())
            if observation is not None:
                last_readable = clock()
                if observation.dealer_streak is None and not streak_reported:
                    streak_reported = True  # 每次執行只存一張，補模板用
                    log(f"[{stamp}] 認不出連莊數（先當 {STREAK_UNKNOWN_GUESS}），截圖存到 "
                        f"{save_result(frame, '連莊_')}")
            elif (last_readable is not None and clock() - last_readable >= UNREADABLE_AFTER
                  and not (decided_action is not None
                           and decided_action.kind == ActionType.DECLARE)):
                last_readable = None  # 每段讀不到的期間只存一張
                log(f"[{stamp}] 連續 {UNREADABLE_AFTER:.0f} 秒讀不到手牌，截圖存到 "
                    f"{save_result(frame, '讀不到_')}")
            # 吃碰槓按鈕亮著卻一直沒處理：存截圖查原因（手牌變暗讀不到、放大的牌讀不到…）
            lit = observation.buttons if observation is not None else \
                (find_buttons(frame) if find_buttons is not None else None)
            if lit and lit & {"chow", "pung", "kong"}:
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
                    last_readable = None  # 這局結束了：結算、配桌期間讀不到手牌是正常的
            if observation is None or observation != previous:
                previous = observation
                continue
            if isinstance(observation, ContinueButton):
                if observation == acted_for:
                    if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                        continue
                    retries += 1
                else:
                    retries = 0
                    log(f"[{stamp}] 結算截圖存到 {save_result(frame)}")
                    if rivers is not None:
                        log(f"[{stamp}] 本局牌河：{river_summary(rivers)}"
                            f"（累計自動收集 {rivers.saved} 張牌河模板）", echo=False)
                        rivers.new_hand()
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
                expected, forbidden = pending_claim
                if tuple(sorted(observation.hand)) == expected:
                    observation = replace(observation, claimed=True, forbidden=forbidden)
                else:
                    pending_claim = None
            # 同一個畫面只決定一次：蒙地卡羅 AI 每次要約 2 秒且有隨機性，重算會讓建議閃動
            if observation != decided_for:
                decided_for, decided_action = observation, decide(observation)
            action = decided_action
            if action is not None and action.kind in (ActionType.CHOW, ActionType.PUNG):
                pending_claim = (hand_after_claim(observation, action),
                                 forbidden_after_claim(action, action.tile))
            advice = describe(action) if action is not None else None
            buttons = observation.buttons or frozenset()
            if "win" in buttons and (action is None or action.kind != ActionType.WIN)                 and observation != missed_for:
                missed_for = observation
                log(f"[{stamp}] 「胡」按鈕亮著但 AI 沒選胡，截圖存到 {save_result(frame, '未胡_')}",
                    hand_text(observation))
            if advice != last_advice:
                if advice is not None:
                    log(f"[{stamp}] {advice}", hand_text(observation))
                if show is not None:
                    show(advice)
            last_advice = advice
            if actuator is None or action is None:
                continue
            if observation == acted_for:
                if retries >= MAX_RETRIES and clock() - acted_at >= RETRY_AFTER                         and stuck_for != observation:
                    stuck_for = observation  # 點了幾次畫面都沒變：存下來查原因（例如槓完沒反應）
                    log(f"[{stamp}] 「{advice}」點了 {MAX_RETRIES + 1} 次畫面都沒變，截圖存到 "
                        f"{save_result(frame, '卡住_')}", hand_text(observation))
                if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                    continue
                retries += 1
            else:
                retries = 0
            if not actuator.perform(action, observation):
                log(f"[{stamp}] 畫面上找不到對應的按鈕或牌，略過", hand_text(observation))
            acted_for = observation
            acted_at = clock()
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
                             "mc 蒙地卡羅（模擬比較，每步約 2 秒）")
    parser.add_argument("--demo-overlay", action="store_true",
                        help="只展示小視窗外觀（可拖曳調整位置），不辨識畫面")
    args = parser.parse_args()
    use_policy(args.ai)
    if args.ai == "mc":
        import os
        from multiprocessing import Pool

        from hint.advisor import use_worker_pool

        workers = max(1, (os.cpu_count() or 2) - 2)
        use_worker_pool(Pool(workers), workers)
        print(f"蒙地卡羅 AI：{workers} 個程序平行模擬", flush=True)
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
        print("自動模式 3 秒後開始，請切回遊戲視窗。滑鼠甩到螢幕左上角可立即中止。", flush=True)
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
    rivers = RiverWatch(RiverReader.from_directory())
    if not overlay:
        print(f"{mode}啟動，Ctrl+C 結束", flush=True)
        run(screen_frames(interval), readers, actuator=actuator, find_continue=find_continue,
            log=log, rivers=rivers, find_win=win_is_lit, find_buttons=read_buttons,
            find_popup=popups.find)
        return
    from hint.overlay import Overlay

    print(f"{mode}啟動（浮動小視窗），Ctrl+C 結束", flush=True)
    window = Overlay()
    window.run(lambda: run(screen_frames(interval), readers, actuator=actuator,
                           show=window.show, find_continue=find_continue, log=log,
                           rivers=rivers, find_win=win_is_lit, find_buttons=read_buttons,
                           find_popup=popups.find))


def win_is_lit(frame: np.ndarray) -> bool:
    buttons = read_buttons(frame)
    return buttons is not None and "win" in buttons


if __name__ == "__main__":
    main()
