from perception.river_reader import SEATS, RiverTile
from perception.river_tracker import CONFIRM_FRAMES, RiverTracker


def tile(value, x, y=0, score=0.9):
    return RiverTile(value, score, (x, y, x + 50, y + 50))


def row(*tiles):
    """一排牌，每張間距 60。"""
    return [tile(value, 60 * index) for index, value in enumerate(tiles)]


def reading(**seats):
    return {seat: seats.get(seat, []) for seat in SEATS}


def feed(tracker, frames):
    events = []
    for frame in frames:
        events += tracker.update(frame)
    return [(e.kind, e.seat, e.tile) for e in events]


def test_discard_is_confirmed_after_consecutive_frames():
    tracker = RiverTracker()
    assert feed(tracker, [reading(top=row(5))] * (CONFIRM_FRAMES - 1)) == []
    assert feed(tracker, [reading(top=row(5))]) == [("discard", "top", 5)]
    assert feed(tracker, [reading(top=row(5))] * 3) == []  # 不重複
    assert tracker.discards["top"] == [5]


def test_short_lived_objects_are_not_discards():
    tracker = RiverTracker()
    blip = reading(right=[tile(9, 300, 60)])  # 牌滑進牌河的動畫、放大顯示等
    assert feed(tracker, [blip] * (CONFIRM_FRAMES - 1) + [reading()] * 3) == []
    assert tracker.rivers["right"] == []


def test_brief_occlusion_is_ignored():
    tracker = RiverTracker()
    feed(tracker, [reading(right=row(1, 2, 3))] * CONFIRM_FRAMES)
    events = feed(tracker, [reading(right=[])] * (CONFIRM_FRAMES + 1)
                  + [reading(right=row(1, 2, 3))])
    assert events == [] and tracker.tiles("right") == [1, 2, 3]


def test_claim_removes_the_latest_tile_but_keeps_it_as_discarded():
    tracker = RiverTracker()
    feed(tracker, [reading(left=row(7))] * CONFIRM_FRAMES + [reading(left=row(7, 8))] * CONFIRM_FRAMES)
    assert feed(tracker, [reading(left=row(7))] * CONFIRM_FRAMES) == [("claimed", "left", 8)]
    assert tracker.tiles("left") == [7] and tracker.discards["left"] == [7, 8]


def test_new_column_at_the_top_is_still_the_latest_discard():
    # 上家、下家一欄一欄排：新的一欄從最上面開始，位置排序會插在中間
    tracker = RiverTracker()
    column = [tile(1, 0, 0), tile(2, -10, 60), tile(3, -20, 120)]
    feed(tracker, [reading(left=column[:1])] * CONFIRM_FRAMES)
    feed(tracker, [reading(left=column[:2])] * CONFIRM_FRAMES)
    feed(tracker, [reading(left=column)] * CONFIRM_FRAMES)
    new_column = column + [tile(4, 70, 0)]
    assert feed(tracker, [reading(left=new_column)] * CONFIRM_FRAMES) == [("discard", "left", 4)]
    assert tracker.tiles("left") == [1, 2, 3, 4]


def test_unknown_tiles_are_filled_in_later():
    tracker = RiverTracker()
    feed(tracker, [reading(top=row(None))] * CONFIRM_FRAMES)
    feed(tracker, [reading(top=row(4))])
    assert tracker.tiles("top") == [4] and tracker.discards["top"] == [4]


def test_long_occlusion_does_not_duplicate_discards():
    # 吃碰按鈕列會蓋住自己的牌河好幾秒
    tracker = RiverTracker()
    feed(tracker, [reading(me=row(1, 2))] * CONFIRM_FRAMES)
    assert feed(tracker, [reading()] * 20 + [reading(me=row(1, 2))] * 5) == []
    assert tracker.tiles("me") == [1, 2] and tracker.discards["me"] == [1, 2]


def test_false_claim_is_restored_when_the_tile_reappears():
    # 只有最新一張被放大動畫蓋住，看起來像被吃碰
    tracker = RiverTracker()
    feed(tracker, [reading(right=row(1, 2))] * CONFIRM_FRAMES)
    events = feed(tracker, [reading(right=row(1))] * CONFIRM_FRAMES + [reading(right=row(1, 2))])
    assert events == [("claimed", "right", 2), ("restored", "right", 2)]
    assert tracker.tiles("right") == [1, 2] and tracker.discards["right"] == [1, 2]


def test_claimed_tile_does_not_come_back_after_a_new_discard():
    tracker = RiverTracker()
    feed(tracker, [reading(top=row(1, 2))] * CONFIRM_FRAMES)
    feed(tracker, [reading(top=row(1))] * CONFIRM_FRAMES)       # 2 被吃碰
    feed(tracker, [reading(top=row(1, 3))] * CONFIRM_FRAMES)    # 原位置打出新的 3
    assert tracker.tiles("top") == [1, 3] and tracker.discards["top"] == [1, 2, 3]


def test_reset_starts_a_new_hand():
    tracker = RiverTracker()
    feed(tracker, [reading(top=row(4))] * CONFIRM_FRAMES)
    tracker.reset()  # 結算畫面出現時由提示迴圈呼叫
    assert tracker.rivers["top"] == [] and tracker.discards["top"] == []
