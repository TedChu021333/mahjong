"""視覺模組的固定環境設定。"""
from pathlib import Path


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
TRAIN_DIR = Path("train")
"""訓練截圖、標註與辨識資料的統一根目錄。"""

# Regions use original 1920x1080 screenshot pixels: (left, top, right, bottom).
MY_HAND_REGION = (10, 818, 1763, 996)
OPEN_MELD_REGION_BY_COUNT = {
	1: (3, 855, 336, 994),
	2: (0, 854, 657, 996),
}
