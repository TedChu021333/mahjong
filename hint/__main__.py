"""提示模式入口。

    python -m hint                         # 即時擷取螢幕（遊戲最大化、1920x1080）
    python -m hint --image train/吃牌畫面.png
    python -m hint --video train/遊戲流程.mp4 --step 1

連續兩幀辨識結果相同才給建議（避開理牌、摸牌等動畫），建議改變時才印出。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from hint.advisor import Readers, advise, observe
from perception.capture import grab_screen, read_image


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


def run(frames: Iterator[tuple[str, np.ndarray]], readers: Readers) -> None:
    previous = None
    last_advice = None
    for stamp, frame in frames:
        observation = observe(frame, readers)
        if observation is None or observation != previous:
            previous = observation
            continue
        advice = advise(observation)
        if advice is not None and advice != last_advice:
            print(f"[{stamp}] {advice}", flush=True)
        last_advice = advice


def main() -> None:
    parser = argparse.ArgumentParser(description="麻將提示模式")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--step", type=float, default=0.5, help="影片每隔幾秒取一幀")
    parser.add_argument("--start", type=float, default=0.0, help="影片從第幾秒開始")
    parser.add_argument("--interval", type=float, default=0.3, help="即時模式擷取間隔（秒）")
    args = parser.parse_args()
    readers = Readers()
    if args.image:
        observation = observe(read_image(args.image), readers)
        print(advise(observation) if observation else "畫面無法辨識或不需要做決定")
    elif args.video:
        run(video_frames(args.video, args.step, args.start), readers)
    else:
        print("提示模式啟動，Ctrl+C 結束", flush=True)
        run(screen_frames(args.interval), readers)


if __name__ == "__main__":
    main()
