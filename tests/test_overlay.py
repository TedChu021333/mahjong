from hint.overlay import wrap_parts


def test_meld_label_wraps_at_the_screen_edge_without_splitting_a_meld():
    text = "下家　碰 白　吃 七八九筒　槓 二筒"
    assert wrap_parts(text, len, 100) == text                      # 放得下就一行
    assert wrap_parts(text, len, 10) == "下家　碰 白\n吃 七八九筒\n槓 二筒"
    assert wrap_parts(text, len, 1) == "下家\n碰 白\n吃 七八九筒\n槓 二筒"  # 一段比寬度還長也不拆
