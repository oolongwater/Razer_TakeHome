import numpy as np
from ultralytics.trackers.byte_tracker import KalmanFilterXYAH, STrack

import gameplay_events.detect_track  # noqa: F401  applies the first-frame activation patch


def test_a_new_track_is_visible_on_its_first_frame():
    track = STrack(np.array([100.0, 80.0, 20.0, 40.0, 0.0]), 0.4, 0)
    track.activate(KalmanFilterXYAH(), frame_id=5)
    assert track.is_activated


def test_overlay_adds_a_side_panel():
    from gameplay_events.events import Detection, EventBuilder
    from gameplay_events.pipeline import _draw_overlay

    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    detection = Detection(0, 0.0, 3, "enemy", 0.8, (20, 30, 50, 80))
    builder = EventBuilder()
    builder.update(0, 0.0, [detection])
    canvas = _draw_overlay(
        frame,
        [detection],
        {"health": 100, "ammo": 50},
        [],
        [],
            builder,
        [{"text": "DETECTED", "bbox": detection.bbox, "until": 1.0}],
    )
    assert canvas.shape == (600, 1280, 3)
