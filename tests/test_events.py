from gameplay_events.events import (
    ENEMY_DETECTED,
    ENEMY_DISAPPEARED,
    ITEM_DETECTED,
    ITEM_PICKED_UP,
    Detection,
    EventBuilder,
    HudChange,
)


def detection(frame, time, track_id, class_name, bbox, confidence=0.9):
    return Detection(frame, time, track_id, class_name, confidence, bbox)


def enemy_box():
    return (200, 40, 230, 90)


def run_frames(builder, specs):
    """specs: list of (frame, time, detections, hud_changes[, flash])."""
    for spec in specs:
        frame, time, detections, changes = spec[:4]
        flash = spec[4] if len(spec) > 4 else False
        builder.update(frame, time, detections, changes, flash=flash)


def test_single_frame_flicker_is_ignored():
    builder = EventBuilder()
    run_frames(builder, [(0, 0.0, [detection(0, 0.0, 1, "enemy", enemy_box())], [])])
    builder.finish(0.0)
    assert builder.events == []


def test_three_of_five_confirms_one_enemy_at_onset():
    builder = EventBuilder()
    frames = []
    for index in range(4):
        hit = [detection(index, index / 30, 7, "enemy", enemy_box(), 0.8)] if index != 1 else []
        frames.append((index, index / 30, hit, []))
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ENEMY_DETECTED]
    assert len(detected) == 1
    assert detected[0].timestamp == 0.0
    assert detected[0].track_id == 7
    assert detected[0].class_name == "enemy"


def test_enemy_disappears_at_last_seen_time():
    builder = EventBuilder()
    frames = []
    for index in range(3):
        frames.append((index, index / 30, [detection(index, index / 30, 3, "enemy", enemy_box())], []))
    # Gap longer than the expiry window, then an empty frame.
    frames.append((20, 0.7, [], []))
    run_frames(builder, frames)
    gone = [event for event in builder.events if event.event == ENEMY_DISAPPEARED]
    assert len(gone) == 1
    assert abs(gone[0].timestamp - 2 / 30) < 1e-6


def test_majority_vote_picks_the_class():
    builder = EventBuilder()
    boxes = (40, 40, 55, 60)
    frames = [
        (0, 0.0, [detection(0, 0.0, 1, "ammo_item", boxes)], []),
        (1, 0.03, [detection(1, 0.03, 1, "health_item", boxes)], []),
        (2, 0.06, [detection(2, 0.06, 1, "health_item", boxes)], []),
    ]
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ITEM_DETECTED]
    assert len(detected) == 1
    assert detected[0].class_name == "health_item"


def test_item_leaving_the_view_is_not_a_pickup():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "ammo_item", far)], []) for index in range(3)]
    frames.append((20, 0.8, [], []))
    run_frames(builder, frames)
    builder.finish(0.8)
    assert [event.event for event in builder.events] == [ITEM_DETECTED]


def test_item_expiring_in_the_pickup_zone_without_a_flash_is_not_a_pickup():
    builder = EventBuilder()
    near = (130, 130, 160, 160)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "health_item", near)], []) for index in range(3)]
    frames.append((20, 0.8, [], []))
    run_frames(builder, frames)
    assert not any(event.event == ITEM_PICKED_UP for event in builder.events)


def test_hud_increase_picks_up_an_item_outside_the_zone():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "ammo_item", far)], []) for index in range(3)]
    frames.append((4, 0.2, [], [HudChange(0.2, "ammo", 20, 24)], True))
    frames.append((20, 0.9, [], []))
    run_frames(builder, frames)
    pickups = [event for event in builder.events if event.event == ITEM_PICKED_UP]
    assert len(pickups) == 1


def test_weapon_switch_without_a_flash_is_not_a_pickup():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "ammo_item", far)], []) for index in range(3)]
    frames.append((4, 0.2, [], [HudChange(0.2, "ammo", 4, 50)]))
    frames.append((20, 0.9, [], []))
    run_frames(builder, frames)
    assert not any(event.event == ITEM_PICKED_UP for event in builder.events)


def test_hud_increase_after_the_item_expired_still_counts():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "ammo_item", far)], []) for index in range(3)]
    frames.append((18, 0.6, [], []))
    frames.append((27, 0.9, [], [HudChange(0.9, "ammo", 20, 24)], True))
    run_frames(builder, frames)
    pickups = [event for event in builder.events if event.event == ITEM_PICKED_UP]
    assert len(pickups) == 1
    assert abs(pickups[0].timestamp - 0.9) < 1e-6


def test_health_loss_does_not_pick_up_an_item():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "health_item", far)], []) for index in range(3)]
    frames.append((4, 0.2, [], [HudChange(0.2, "health", 100, 80)]))
    frames.append((20, 0.9, [], []))
    run_frames(builder, frames)
    assert not any(event.event == ITEM_PICKED_UP for event in builder.events)


def test_one_hud_increase_picks_up_only_one_item():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = []
    for index in range(3):
        frames.append(
            (
                index,
                index / 30,
                [
                    detection(index, index / 30, 4, "ammo_item", far),
                    detection(index, index / 30, 5, "ammo_item", far),
                ],
                [],
            )
        )
    frames.append((4, 0.2, [], [HudChange(0.2, "ammo", 20, 24)], True))
    frames.append((20, 0.9, [], []))
    run_frames(builder, frames)
    pickups = [event for event in builder.events if event.event == ITEM_PICKED_UP]
    assert len(pickups) == 1


def test_hud_increase_before_the_item_was_seen_does_not_count():
    builder = EventBuilder()
    far = (10, 20, 24, 36)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "ammo_item", far)], []) for index in range(3)]
    frames[0] = (0, 0.0, [detection(0, 0.0, 4, "ammo_item", far)], [HudChange(0.0, "ammo", 20, 24)])
    frames.append((20, 0.8, [], []))
    run_frames(builder, frames)
    assert not any(event.event == ITEM_PICKED_UP for event in builder.events)


def test_zone_and_hud_do_not_double_count():
    builder = EventBuilder()
    near = (130, 130, 160, 160)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "health_item", near)], []) for index in range(3)]
    frames.append((4, 0.2, [], [HudChange(0.2, "health", 80, 100)], True))
    frames.append((20, 0.9, [], []))
    run_frames(builder, frames)
    pickups = [event for event in builder.events if event.event == ITEM_PICKED_UP]
    assert len(pickups) == 1


def test_a_fragment_keeps_the_parent_id():
    builder = EventBuilder()
    box = enemy_box()
    frames = [(index, index / 30, [detection(index, index / 30, 1, "enemy", box)], []) for index in range(3)]
    frames.append((6, 0.2, [detection(6, 0.2, 2, "enemy", box)], []))
    frames.append((7, 0.23, [detection(7, 0.23, 2, "enemy", box)], []))
    frames.append((8, 0.27, [detection(8, 0.27, 2, "enemy", box)], []))
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ENEMY_DETECTED]
    assert len(detected) == 1
    assert detected[0].track_id == 1
    assert builder.display_id(2) == 1


def test_a_new_enemy_does_not_absorb_one_still_in_the_frame():
    builder = EventBuilder()
    left = (20, 40, 50, 90)
    right = (200, 40, 230, 90)
    frames = [(index, index / 30, [detection(index, index / 30, 1, "enemy", left)], []) for index in range(3)]
    for index in range(5, 8):
        frames.append(
            (
                index,
                index / 30,
                [
                    detection(index, index / 30, 2, "enemy", right),
                    detection(index, index / 30, 1, "enemy", left),
                ],
                [],
            )
        )
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ENEMY_DETECTED]
    assert {event.track_id for event in detected} == {1, 2}


def test_two_enemies_at_once_are_not_merged():
    builder = EventBuilder()
    left = (20, 40, 50, 90)
    right = (200, 40, 230, 90)
    frames = []
    for index in range(3):
        frames.append(
            (
                index,
                index / 30,
                [
                    detection(index, index / 30, 1, "enemy", left),
                    detection(index, index / 30, 2, "enemy", right),
                ],
                [],
            )
        )
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ENEMY_DETECTED]
    assert len(detected) == 2


def test_expiry_waits_while_a_successor_might_confirm():
    builder = EventBuilder()
    box = enemy_box()
    frames = [(index, index / 30, [detection(index, index / 30, 1, "enemy", box)], []) for index in range(3)]
    frames.append((16, 16 / 30, [detection(16, 16 / 30, 2, "enemy", box)], []))
    frames.append((17, 17 / 30, [detection(17, 17 / 30, 2, "enemy", box)], []))
    run_frames(builder, frames)
    assert not any(event.event == ENEMY_DISAPPEARED for event in builder.events)


def test_flash_plus_zone_item_is_a_pickup():
    builder = EventBuilder()
    near = (130, 130, 160, 160)
    frames = [(index, index / 30, [detection(index, index / 30, 4, "health_item", near)], []) for index in range(3)]
    frames.append((6, 0.2, [], [], True))
    frames.append((20, 0.6, [], []))
    run_frames(builder, frames)
    pickups = [event for event in builder.events if event.event == ITEM_PICKED_UP]
    assert len(pickups) == 1
    assert pickups[0].class_name == "health_item"


def test_a_gap_longer_than_expiry_is_a_new_enemy():
    builder = EventBuilder()
    box = enemy_box()
    frames = [(index, index / 30, [detection(index, index / 30, 1, "enemy", box)], []) for index in range(3)]
    frames.append((30, 1.0, [], []))
    for index in range(31, 34):
        frames.append((index, index / 30, [detection(index, index / 30, 2, "enemy", box)], []))
    run_frames(builder, frames)
    detected = [event for event in builder.events if event.event == ENEMY_DETECTED]
    assert len(detected) == 2
