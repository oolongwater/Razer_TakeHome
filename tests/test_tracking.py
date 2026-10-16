import numpy as np
from ultralytics.trackers.byte_tracker import KalmanFilterXYAH, STrack

import gameplay_events.detect_track  # noqa: F401  applies the first-frame activation patch


def test_a_new_track_is_visible_on_its_first_frame():
    track = STrack(np.array([100.0, 80.0, 20.0, 40.0, 0.0]), 0.4, 0)
    track.activate(KalmanFilterXYAH(), frame_id=5)
    assert track.is_activated
