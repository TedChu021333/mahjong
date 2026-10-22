from perception.label_meld_corners import CORNERS, SEATS, TILES, CornerSession


def test_session_collects_four_corners_of_the_first_and_last_tile_per_row():
    session = CornerSession()
    assert "上家" in session.prompt() and "左上" in session.prompt()
    per_seat = len(TILES) * len(CORNERS)
    assert per_seat == 8
    for i in range(per_seat):
        session.add(i, i)
    session.undo()
    assert len(session.points) == per_seat - 1
    for i in range(per_seat + 1):
        session.add(i, i)
    assert session.done and "存檔" in session.prompt()
    data = session.to_json()
    assert set(data) == {seat for seat, _ in SEATS}
    assert len(data["left"]["first"]) == 4 and len(data["right"]["last"]) == 4
