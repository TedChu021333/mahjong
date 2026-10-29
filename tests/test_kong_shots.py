from hint.kong_shots import KongShots


def test_shot_is_taken_once_after_the_gold_count_stays_higher():
    shots = KongShots(delay=2.0, limit=5)
    assert not shots.update(2, 0.0)
    assert not shots.update(3, 1.0)      # 剛多一種，等動畫結束
    assert not shots.update(3, 2.5)
    assert shots.update(3, 3.1)
    assert not shots.update(3, 6.0)      # 同一次槓只存一張


def test_flicker_and_new_hand_do_not_trigger():
    shots = KongShots(delay=2.0, limit=5)
    shots.update(3, 0.0)
    assert not shots.update(4, 1.0)
    assert not shots.update(3, 2.0)      # 少讀一種又回來：重新計時
    assert not shots.update(4, 3.5)
    assert not shots.update(4, 4.0)
    shots.new_hand()
    assert not shots.update(2, 10.0)
    assert not shots.update(2, 20.0)


def test_limit():
    shots = KongShots(delay=0.0, limit=1)
    shots.update(2, 0.0)
    shots.update(3, 1.0)
    assert shots.update(3, 1.0)
    shots.update(4, 2.0)
    assert not shots.update(4, 2.0)


def test_start_and_confirm_flags_ignore_the_screenshot_limit():
    shots = KongShots(delay=2.0, limit=0)
    shots.update(2, 0.0)
    shots.update(3, 1.0)
    assert shots.started and not shots.confirmed
    shots.update(3, 2.0)
    assert not shots.started and not shots.confirmed
    assert not shots.update(3, 3.5)      # 張數用完不存圖
    assert shots.confirmed
    shots.update(3, 4.0)
    assert not shots.confirmed
