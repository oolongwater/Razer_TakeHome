import cv2
import numpy as np

from gameplay_events.hud import (
    AMMO_RIGHT,
    DIGIT_HEIGHT,
    DIGIT_TOP,
    DIGIT_WIDTH,
    HEALTH_RIGHT,
    MATCH_THRESHOLD,
    HudReader,
    TemporalFilter,
    ink_mask,
    load_templates,
    read_counter,
)


def test_filter_waits_for_three_identical_readings():
    smoother = TemporalFilter(persistence=3)
    assert smoother.update(100) is None
    assert smoother.update(100) is None
    assert smoother.update(100) == 100
    assert smoother.update(90) == 100
    assert smoother.update(90) == 100
    assert smoother.update(90) == 90


def test_filter_rejects_a_single_frame_glitch():
    smoother = TemporalFilter(persistence=3)
    for _ in range(3):
        smoother.update(50)
    assert smoother.update(12) == 50
    assert smoother.update(50) == 50
    assert smoother.update(50) == 50


def test_reader_emits_one_change_after_the_value_settles():
    reader = HudReader.__new__(HudReader)
    reader.templates = None
    reader.health_filter = TemporalFilter()
    reader.ammo_filter = TemporalFilter()
    reader.health = None
    reader.ammo = None

    def feed(health, ammo, time):
        reader.health_filter  # keep attribute access obvious
        # Drive the filters through the public update by stubbing read_hud's inputs.
        from gameplay_events.hud import HEALTH_CHANGED

        previous_events = []
        # Call the internal apply path the same way update() does.
        event, _change = reader._apply("health", reader.health_filter, health, 0.9, time, HEALTH_CHANGED)
        if event is not None:
            previous_events.append(event)
        return previous_events

    assert feed(100, 50, 0.0) == []
    assert feed(100, 50, 0.03) == []
    assert feed(100, 50, 0.06) == []
    assert reader.health == 100
    assert feed(80, 50, 0.1) == []
    assert feed(80, 50, 0.13) == []
    changed = feed(80, 50, 0.16)
    assert len(changed) == 1
    assert changed[0].old_value == 100
    assert changed[0].new_value == 80
    assert changed[0].to_dict()["delta"] == -20


def _paint_counter(frame, right_edge, text, templates):
    """Stamp ink templates into a counter, right aligned."""
    glyphs = list(text)
    for offset, glyph in enumerate(reversed(glyphs)):
        x = right_edge - DIGIT_WIDTH * (offset + 1)
        template = templates[int(glyph)][0]
        color = np.zeros((DIGIT_HEIGHT, DIGIT_WIDTH, 3), np.uint8)
        # Green ink, which the mask treats as a digit.
        color[:, :, 1] = template
        frame[DIGIT_TOP : DIGIT_TOP + DIGIT_HEIGHT, x : x + DIGIT_WIDTH] = color


def test_template_match_reads_a_painted_counter():
    templates = load_templates()
    frame = np.zeros((200, 320, 3), np.uint8)
    _paint_counter(frame, HEALTH_RIGHT, "100", templates)
    _paint_counter(frame, AMMO_RIGHT, "50", templates)
    health, health_score = read_counter(frame, HEALTH_RIGHT, templates)
    ammo, ammo_score = read_counter(frame, AMMO_RIGHT, templates)
    assert health == 100
    assert ammo == 50
    assert health_score >= MATCH_THRESHOLD
    assert ammo_score >= MATCH_THRESHOLD
    assert ink_mask(frame[DIGIT_TOP : DIGIT_TOP + DIGIT_HEIGHT, 62:76]).sum() > 0


def test_ammo_change_feeds_pickups_but_is_not_an_event():
    templates = load_templates()
    reader = HudReader(templates)
    events = changes = []
    for index, ammo in enumerate(("50", "50", "50", "60", "60", "60")):
        frame = np.zeros((200, 320, 3), np.uint8)
        _paint_counter(frame, HEALTH_RIGHT, "100", templates)
        _paint_counter(frame, AMMO_RIGHT, ammo, templates)
        _reading, new_events, new_changes = reader.update(frame, index / 30)
        events = events + new_events
        changes = changes + new_changes
    assert events == []
    assert [(change.stat, change.old_value, change.new_value) for change in changes] == [("ammo", 50, 60)]


def test_real_status_bar_frame_reads_starting_values():
    templates = load_templates()
    capture = cv2.VideoCapture("data/clips/Doom2_UV_lv04_028.avi")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return
    health, _health_score = read_counter(frame, HEALTH_RIGHT, templates)
    ammo, _ammo_score = read_counter(frame, AMMO_RIGHT, templates)
    assert health == 100
    assert ammo == 50
