"""從已標註的截圖切出暗手牌，存成模板。

用法（標註依畫面由左到右，格式同 game.tiles.parse）：
    python -m perception.extract_templates train/完整正常畫面.png 257778m114p2335588s

輸出：train/templates/<牌代碼>/<截圖檔名>_<位置>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from game.tiles import parse, tile_code
from perception.capture import read_image, write_image
from perception.config import TEMPLATE_DIR
from perception.hand_reader import crop_tile, locate_hand_tiles


def extract_templates(image: np.ndarray, labels: list[int], name: str,
                      output_dir: Path = TEMPLATE_DIR) -> list[Path]:
    boxes = locate_hand_tiles(image)
    if len(boxes) != len(labels):
        raise ValueError(f"畫面偵測到 {len(boxes)} 張暗手牌，但標註有 {len(labels)} 張")
    paths = []
    for index, (box, tile) in enumerate(zip(boxes, labels)):
        path = output_dir / tile_code(tile) / f"{name}_{index:02d}.png"
        write_image(path, crop_tile(image, box))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="從截圖切出牌面模板")
    parser.add_argument("image", type=Path)
    parser.add_argument("labels", help="暗手牌由左到右，如 257778m114p2335588s")
    parser.add_argument("--output", type=Path, default=TEMPLATE_DIR)
    args = parser.parse_args()
    paths = extract_templates(read_image(args.image), parse(args.labels),
                              args.image.stem, args.output)
    print(f"已輸出 {len(paths)} 張模板到 {args.output}")


if __name__ == "__main__":
    main()
