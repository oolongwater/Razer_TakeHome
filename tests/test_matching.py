from gameplay_events.matching import events_from_intervals, match_events
from gameplay_events.radar import bearing_degrees, proximity_label


def test_matching_is_one_to_one_and_type_specific():
    predicted = [
        {"timestamp": 1.0, "event": "enemy_detected"},
        {"timestamp": 1.2, "event": "enemy_detected"},
        {"timestamp": 3.0, "event": "player_health_changed"},
    ]
    ground_truth = [
        {"timestamp": 1.1, "event": "enemy_detected"},
        {"timestamp": 2.5, "event": "enemy_detected"},
        {"timestamp": 3.2, "event": "item_picked_up"},
    ]
    scores = match_events(predicted, ground_truth, tolerance=0.5)
    assert scores["enemy_detected"]["tp"] == 1
    assert scores["enemy_detected"]["fp"] == 1
    assert scores["enemy_detected"]["fn"] == 1
    assert scores["player_health_changed"]["fp"] == 1
    assert scores["item_picked_up"]["fn"] == 1
    assert scores["overall"]["tp"] == 1


def test_outside_tolerance_does_not_match():
    predicted = [{"timestamp": 0.0, "event": "enemy_disappeared"}]
    ground_truth = [{"timestamp": 1.2, "event": "enemy_disappeared"}]
    scores = match_events(predicted, ground_truth, tolerance=0.5)
    assert scores["enemy_disappeared"]["tp"] == 0
    assert scores["enemy_disappeared"]["fp"] == 1
    assert scores["enemy_disappeared"]["fn"] == 1


def test_intervals_merge_short_gaps_and_skip_the_clip_end():
    rows = [
        {"object": "e1", "class_name": "enemy", "first_frame": 0, "last_frame": 5, "picked_up": 0},
        {"object": "e1", "class_name": "enemy", "first_frame": 10, "last_frame": 20, "picked_up": 0},
        {"object": "h1", "class_name": "health_item", "first_frame": 30, "last_frame": 34, "picked_up": 1},
        {"object": "a1", "class_name": "ammo_item", "first_frame": 40, "last_frame": 50, "picked_up": 0},
        {"object": "e2", "class_name": "enemy", "first_frame": 0, "last_frame": 1, "picked_up": 0},
        {"object": "e3", "class_name": "enemy", "first_frame": 85, "last_frame": 90, "picked_up": 0},
    ]
    events = events_from_intervals(rows, fps=30, clip_end=3.0)
    kinds = [event["event"] for event in events]
    # e1's two intervals merge. e2 is under 3 frames. e3 is kept but does not disappear.
    # a1 leaves without a pickup and only adds its detection.
    assert kinds.count("enemy_detected") == 2
    assert kinds.count("enemy_disappeared") == 1
    assert kinds.count("item_detected") == 2
    assert kinds.count("item_picked_up") == 1
    assert "item_disappeared" not in kinds


def test_bearing_is_zero_ahead_and_signed():
    assert abs(bearing_degrees(160)) < 1e-6
    assert bearing_degrees(320) > 40
    assert bearing_degrees(0) < -40


def test_proximity_bands():
    assert proximity_label(50) == "near"
    assert proximity_label(20) == "mid"
    assert proximity_label(8) == "far"
