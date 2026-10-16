import numpy as np

from gameplay_events.flash import FlashDetector


def _frame(yellow, red):
    """A flat view whose yellow and red scores match the requested values."""
    # yellow = (r+g)/2 - b, redness = r - (g+b)/2. Set g = b = 40 and solve r.
    blue = np.full((20, 20), 40, dtype=np.uint8)
    green = blue.copy()
    # Use the yellow offset directly by raising red and green together, then extra red.
    red_channel = np.full((20, 20), np.clip(40 + yellow + red, 0, 255), dtype=np.uint8)
    green_channel = np.full((20, 20), np.clip(40 + yellow - red, 0, 255), dtype=np.uint8)
    return np.dstack([blue, green_channel, red_channel])


def test_a_yellow_jump_is_a_pickup_flash_and_a_red_jump_is_not():
    detector = FlashDetector()
    saw = False
    for index in range(10):
        yellow = 5 if index < 7 else 40
        saw = detector.update(_frame(yellow, 0), index / 30) or saw
    assert saw

    damage = FlashDetector()
    saw_damage = False
    for index in range(10):
        yellow = 5 if index < 7 else 40
        red = 0 if index < 7 else 80
        saw_damage = damage.update(_frame(yellow, red), index / 30) or saw_damage
    assert not saw_damage


def test_real_map04_shell_pickup_flashes():
    from pathlib import Path

    import cv2

    video = Path("data/clips/Doom2_UV_lv04_028.avi")
    if not video.exists():
        return
    capture = cv2.VideoCapture(str(video))
    detector = FlashDetector()
    onsets = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (320, 200), interpolation=cv2.INTER_AREA)
        if detector.update(frame[:168], index / 30):
            onsets.append(index)
        index += 1
        if index > 250:
            break
    capture.release()
    assert any(185 <= frame <= 200 for frame in onsets)
    assert not any(frame < 170 for frame in onsets)
