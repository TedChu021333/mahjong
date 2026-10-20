"""清理牌河模板：找出放錯資料夾或不是牌的模板，移到隔離資料夾（不刪除）。

    python -m perception.clean_river_templates            # 列出可疑的張數
    python -m perception.clean_river_templates --move     # 移到 train/river_templates_quarantine/

牌河模板大多是提示迴圈自動收集的（標註來自手牌變化、放大的牌），標註套錯牌、或偵測到的根本不是牌
（吃碰動畫、暗條）時就會混進錯的圖；辨識取每種牌最像的一張模板，一張北放進三萬，真的北就可能讀成三萬。
10/19 6753 張中 1447 張可疑（目視抽查全對），自己打的牌原本只讀對約一半。

做法：用和辨識時一樣的比對（TileClassifier：上下半張分開、取較低者），每張模板找最像的 K 張（不含自己），
同資料夾的不到 MIN_AGREE 張、或最像的一張分數不到 MIN_SCORE 的就是可疑（人工標註的 manual_* 不算）。
移走後再算一輪，直到沒有可疑的。
約 6000 張一輪 2～3 分鐘。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from game.tiles import parse
from perception.capture import read_image
from perception.config import RIVER_TEMPLATE_DIR
from perception.hand_reader import TEMPLATE_MARGIN, TileClassifier, _normalize, _unit_rows

K = 7
MIN_AGREE = 3
MIN_SCORE = 0.5
QUARANTINE_DIR = RIVER_TEMPLATE_DIR.parent / "river_templates_quarantine"


def nearest(images: list[np.ndarray], codes: list[str], k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """每張圖最像的 k 張（不含自己）的索引與分數。"""
    classifier = TileClassifier([(parse(code)[0], image) for code, image in zip(codes, images)],
                                min_score=0)
    halves = [half.astype(np.float32) for half in classifier._halves]
    m = TEMPLATE_MARGIN
    k = min(k, len(images) - 1)
    neighbors = np.zeros((len(images), k), np.int64)
    scores = np.zeros((len(images), k), np.float32)
    for i, image in enumerate(images):
        target = _normalize(image)
        middle = (target.shape[0] - 2 * m) // 2
        per = None
        for region, templates in zip((target[:middle + 2 * m], target[middle:]), halves):
            height, width = region.shape[0] - 2 * m, region.shape[1] - 2 * m
            windows = np.lib.stride_tricks.sliding_window_view(region, (height, width, 3))
            windows = _unit_rows(windows.reshape(-1, height, width, 3)).astype(np.float32)
            # 這裡是大矩陣，用 BLAS 比 einsum 快（辨識時的小矩陣才要避開 BLAS）
            best = (windows @ templates.T).max(axis=0)
            per = best if per is None else np.minimum(per, best)
        per[i] = -np.inf
        top = np.argpartition(-per, k)[:k] if k < len(per) else np.arange(len(per))
        top = top[np.argsort(-per[top])][:k]
        neighbors[i], scores[i] = top, per[top]
    return neighbors, scores


def suspicious(codes: list[str], neighbors: np.ndarray, scores: np.ndarray,
               min_agree: int = MIN_AGREE, min_score: float = MIN_SCORE) -> np.ndarray:
    labels = np.array(codes)
    agree = (labels[neighbors] == labels[:, None]).sum(axis=1)
    return (agree < min_agree) | (scores[:, 0] < min_score)


def load(directory: Path) -> tuple[list[Path], list[str], list[np.ndarray]]:
    paths = sorted(directory.glob("*/*.png"))
    return paths, [path.parent.name for path in paths], [read_image(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="找出放錯資料夾或不是牌的牌河模板")
    parser.add_argument("--dir", type=Path, default=RIVER_TEMPLATE_DIR)
    parser.add_argument("--move", action="store_true", help="移到隔離資料夾（預設只列出）")
    args = parser.parse_args()
    quarantine = args.dir.parent / f"{args.dir.name}_quarantine"
    rounds = 0
    while True:
        rounds += 1
        paths, codes, images = load(args.dir)
        # 人工標註的（manual_ 開頭）不動：下家角度的牌樣本少，相似的模板不夠多也會被判可疑（10/20 19 張全是這種）
        flagged = suspicious(codes, *nearest(images, codes)) &             np.array([not path.name.startswith("manual_") for path in paths])
        print(f"第 {rounds} 輪：{len(paths)} 張中可疑 {int(flagged.sum())} 張")
        if not args.move or not flagged.any():
            break
        for path, flag in zip(paths, flagged):
            if flag:
                target = quarantine / path.parent.name / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
    if args.move:
        print(f"可疑的模板已移到 {quarantine}")


if __name__ == "__main__":
    main()
