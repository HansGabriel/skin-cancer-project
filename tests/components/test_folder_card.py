from services.storage import Folder


def test_folder_dataclass_for_card():
    f = Folder("id1", "Arm scans", "#B58CF0", "2026-01-01T00:00:00+00:00")
    assert f.color == "#B58CF0"
