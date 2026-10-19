import numpy as np

from perception.clean_river_templates import nearest, suspicious


def _tile(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 6, 3)).astype(np.uint8)
    return np.kron(base, np.ones((10, 10, 1), np.uint8))  # 80x60 的方塊圖


def test_misfiled_template_is_flagged():
    rng = np.random.default_rng(0)
    images, codes = [], []
    for code, seed in (("1m", 1), ("2m", 2)):
        for _ in range(6):
            noisy = np.clip(_tile(seed).astype(int) + rng.integers(-10, 10, (80, 60, 3)), 0, 255)
            images.append(noisy.astype(np.uint8))
            codes.append(code)
    images.append(_tile(1))  # 一萬的圖放進二萬資料夾
    codes.append("2m")
    flagged = suspicious(codes, *nearest(images, codes))
    assert flagged.tolist() == [False] * 12 + [True]
