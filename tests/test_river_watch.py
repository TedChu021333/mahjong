from game.tiles import parse
from hint.advisor import Observation
from hint.river_watch import RiverWatch, summary
from perception.river_reader import SEATS, RiverTile
from perception.river_tracker import CONFIRM_FRAMES
from perception.table_reader import ClaimTile


def P(value):
    return parse(value)[0]


class FakeReader:
    """依序回傳預先排好的牌河讀數。"""

    def __init__(self):
        self.current = {seat: [] for seat in SEATS}

    def read(self, frame):
        return {seat: list(tiles) for seat, tiles in self.current.items()}


def watch_with(reader):
    saved = []
    watch = RiverWatch(reader, save=lambda path, image: saved.append(path))
    return watch, saved


def frames(watch, observation, count=CONFIRM_FRAMES):
    events = []
    for _ in range(count):
        events += watch.update(None, observation)
    return events


def waiting(hand="11223789m99m4p56779s", claim=None):
    return Observation(tuple(parse(hand)), None, claim, ())


def test_own_discard_is_labelled_from_the_hand_difference(tmp_path):
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, Observation(tuple(parse("11223789m99m4p56779s9p")), None, None, ()))
    reader.current["me"] = [RiverTile(None, 0.2, (580, 651, 650, 717))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert watch.tracker.discards["me"] == [P("9p")]
    assert len(saved) == 1 and saved[0].parent.name == "9p"
    assert watch.discards()[0] == (P("9p"),)


def test_opponent_discard_is_labelled_from_the_enlarged_tile():
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, waiting(claim=ClaimTile(P("5z"), 0.9, "top")))
    reader.current["top"] = [RiverTile(P("6z"), 0.75, (660, 284, 717, 334))]  # 認錯
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert watch.tracker.discards["top"] == [P("5z")]
    assert [p.parent.name for p in saved] == ["5z"]
    assert watch.discards()[2] == (P("5z"),)
    assert "top 中" in summary(watch)


def test_known_tiles_are_not_saved_again():
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, waiting(claim=ClaimTile(P("1s"), 0.9, "right")))
    reader.current["right"] = [RiverTile(P("1s"), 0.95, (1151, 398, 1212, 450))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert saved == [] and watch.discards()[1] == (P("1s"),)


def test_new_hand_clears_everything():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = [RiverTile(P("3m"), 0.9, (569, 397, 630, 449))]
    frames(watch, waiting())
    watch.new_hand()
    assert watch.discards() == ((), (), (), ())
