"""視覺模組的固定環境設定。"""
from pathlib import Path


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
TRAIN_DIR = Path("train")
"""訓練截圖、標註與辨識資料的統一根目錄。"""
TEMPLATE_DIR = TRAIN_DIR / "templates"
"""牌面模板：train/templates/<牌代碼，如 5m、1z>/*.png。"""

# Regions use original 1920x1080 screenshot pixels: (left, top, right, bottom).
MY_HAND_REGION = (10, 818, 1763, 996)
OPEN_MELD_REGION_BY_COUNT = {
    1: (3, 855, 336, 994),
    2: (0, 854, 657, 996),
    3: (0, 854, 978, 996),  # 由胡牌畫面推估（每組約 321px）
}

# 暗手牌排在固定格線上（副露在左、暗手牌緊接其後），每張間距固定；
# 張數不足時右側格子會空著（見 train/第十七張.png），所以每格要個別檢查。
HAND_TILE_PITCH = 109.07
HAND_GRID_RIGHT = 1760.5
"""最右邊一格的右緣。"""
HAND_TILE_TOP = 828
HAND_TILE_BOTTOM = 998
HAND_FACE_PROBE_ROWS = (852, 857)
"""判斷每一格是否有暗手牌的橫條：在花牌展示區（壓住手牌頂端到 y≈850）
之下、副露牌面頂端（y≈858）之上，只有暗手牌會是白色牌面。"""
MAX_HAND_TILES = 16
"""格線上的格數（不含摸進的第 17 張）。"""
DRAWN_TILE_REGION = (1792, 823, 1902, 995)
"""摸進的第 17 張在畫面上的位置。"""
DRAWN_TILE_CROP_X = (1798, 1905)
"""第 17 張辨識用的裁切左右界：它左側多露出幾像素的牌邊，照格線方式裁會
讓模板比對分數從 0.98 掉到 0.6；y 沿用 HAND_TILE_TOP/BOTTOM。"""
TILE_RAISE = 48
"""被選取的牌（如換三張）上移的距離，由牌面下緣量出；副露牌下緣只差
約 21px，只接受接近此值的位移，避免把副露當成上移的暗手牌。"""

# 吃碰槓聽胡按鈕列（由 train/遊戲流程.mp4 量測）：可按的按鈕是亮的，不能按的變暗。
ACTION_BUTTON_CENTERS = {
    "chow": (428, 688),
    "pung": (640, 688),
    "kong": (850, 688),
    "declare": (1060, 688),
    "win": (1270, 688),
}
CANCEL_BUTTON_REGION = (1470, 650, 1560, 740)
"""「取消」紅色按鈕；出現代表按鈕列正在顯示。"""

# 別家打出、可以吃碰胡的牌會放大顯示在圓圈裡，位置依打牌的人而定。
CLAIM_TILE_POSITIONS = {
    "left": (615, 341),   # 上家
    "right": (1216, 341),  # 下家
}
CLAIM_TILE_SIZE = (106, 153)
CLAIM_TILE_MARGINS = (2, 12, 2, 4)
"""裁切時左、上、右、下多留的像素，讓比例接近手牌模板（網格搜尋結果）。"""

# 金牌面板：一排 4 格，超過換第二排。
GOLD_SLOT_ORIGIN = (437, 38)
GOLD_SLOT_PITCH = (56.7, 82)
GOLD_SLOT_SIZE = (52, 75)
GOLD_MAX_SLOTS = 8
GOLD_TEMPLATE_BOX = (0, 12, 106, 170)
"""金牌小圖只有牌面，比對時手牌模板只取這塊（網格搜尋結果）。"""

# 換三張畫面：「不換」出現代表正在選牌；「換牌」在選了牌之後才會亮（橘色）。
SWAP_KEEP_BUTTON = (757, 655)
SWAP_CONFIRM_BUTTON = (1163, 655)

# 多種吃法時的選項框（由 train/吃牌2.png 量測）：每框三張正面牌，依順子由小到大排列。
CHOW_PANEL_ROWS = (650, 790)
CHOW_PANEL_WIDTH = (230, 340)
