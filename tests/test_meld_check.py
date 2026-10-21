from hint.meld_check import CLAIM_GRACE, PERSIST_SECONDS, MeldCheck


def test_partial_hand_after_discard_is_rejected():
    # 10/21 13:58:12：打完九筒後只讀到「六條五條」兩張，等於 5 組副露
    check = MeldCheck()
    assert check.accept(0, None, 0.0)
    assert not check.accept(5, None, 1.0)
    assert not check.accept(5, None, 1.3)
    assert check.accept(0, None, 1.6)
    assert check.count == 0


def test_one_more_meld_after_claim_buttons():
    check = MeldCheck()
    assert check.accept(0, frozenset({"pung"}), 0.0)
    assert check.accept(1, None, 2.0)
    assert check.count == 1
    # 一次只能多 1
    assert not check.accept(3, frozenset({"chow"}), 3.0)


def test_claim_buttons_long_ago_do_not_count():
    check = MeldCheck()
    check.accept(0, frozenset({"kong"}), 0.0)
    assert not check.accept(1, None, CLAIM_GRACE + 1)


def test_persistent_count_is_accepted():
    # 漏看了按鈕：同一個副露數持續夠久就當真
    check = MeldCheck()
    check.accept(0, None, 0.0)
    assert not check.accept(1, None, 10.0)
    assert not check.accept(1, None, 10.0 + PERSIST_SECONDS / 2)
    assert check.accept(1, None, 10.0 + PERSIST_SECONDS)
    assert check.count == 1


def test_fewer_melds_means_new_hand():
    check = MeldCheck()
    check.accept(0, frozenset({"pung"}), 0.0)
    check.accept(1, None, 1.0)
    assert check.accept(0, None, 60.0)
    assert check.count == 0
