"""辨識打完後的「繼續」類按鈕（小結算的繼續、繼續下一場）。

每個按鈕用一張參考截圖做正規化相關比對；按鈕位置固定，只在它的區域附近搜尋。

    python -m perception.continue_button <截圖> <按鈕名稱>   # 從截圖裁出參考圖
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from perception.capture import read_image, write_image
from perception.config import (
    BUTTON_TEMPLATE_DIR,
    CONTINUE_BUTTONS,
    CONTINUE_MIN_SCORE,
    CONTINUE_SEARCH_MARGIN,
)


@dataclass(frozen=True)
class ContinueButton:
    name: str
    center: tuple[int, int]


def crop(image: np.ndarray, region: tuple[int, int, int, int], margin: int = 0) -> np.ndarray:
    left, top, right, bottom = region
    height, width = image.shape[:2]
    return image[max(0, top - margin):min(height, bottom + margin),
                 max(0, left - margin):min(width, right + margin)]


class ContinueButtons:
    def __init__(self, references: dict[str, np.ndarray]) -> None:
        self.references = {name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                           for name, image in references.items()}

    @classmethod
    def from_directory(cls, directory: Path = BUTTON_TEMPLATE_DIR) -> "ContinueButtons":
        return cls({name: read_image(directory / f"{name}.png")
                    for name in CONTINUE_BUTTONS if (directory / f"{name}.png").exists()})

    def find(self, image: np.ndarray) -> ContinueButton | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for name, reference in self.references.items():
            area = crop(gray, CONTINUE_BUTTONS[name], CONTINUE_SEARCH_MARGIN)
            if area.shape[0] < reference.shape[0] or area.shape[1] < reference.shape[1]:
                continue
            scores = cv2.matchTemplate(area, reference, cv2.TM_CCOEFF_NORMED)
            if scores.max() >= CONTINUE_MIN_SCORE:
                left, top, right, bottom = CONTINUE_BUTTONS[name]
                return ContinueButton(name, ((left + right) // 2, (top + bottom) // 2))
        return None


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in CONTINUE_BUTTONS:
        raise SystemExit(f"用法：python -m perception.continue_button <截圖> "
                         f"<{'|'.join(CONTINUE_BUTTONS)}>")
    path = BUTTON_TEMPLATE_DIR / f"{sys.argv[2]}.png"
    write_image(path, crop(read_image(sys.argv[1]), CONTINUE_BUTTONS[sys.argv[2]]))
    print(f"已存參考圖：{path}")


if __name__ == "__main__":
    main()
