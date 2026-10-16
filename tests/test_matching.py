from gameplay_events.matching import match_events
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
        {"timestamp": 3.2, "event": "player_ammo_changed"},
    ]
    scores = match_events(predicted, ground_truth, tolerance=0.5)
    assert scores["enemy_detected"]["tp"] == 1
    assert scores["enemy_detected"]["fp"] == 1
    assert scores["enemy_detected"]["fn"] == 1
    assert scores["player_health_changed"]["fp"] == 1
    assert scores["player_ammo_changed"]["fn"] == 1
    assert scores["overall"]["tp"] == 1


def test_outside_tolerance_does_not_match():
    predicted = [{"timestamp": 0.0, "event": "enemy_disappeared"}]
    ground_truth = [{"timestamp": 1.2, "event": "enemy_disappeared"}]
    scores = match_events(predicted, ground_truth, tolerance=0.5)
    assert scores["enemy_disappeared"]["tp"] == 0
    assert scores["enemy_disappeared"]["fp"] == 1
    assert scores["enemy_disappeared"]["fn"] == 1


def test_bearing_is_zero_ahead_and_signed():
    assert abs(bearing_degrees(160)) < 1e-6
    assert bearing_degrees(320) > 40
    assert bearing_degrees(0) < -40


def test_proximity_bands():
    assert proximity_label(50) == "near"
    assert proximity_label(20) == "mid"
    assert proximity_label(8) == "far"
