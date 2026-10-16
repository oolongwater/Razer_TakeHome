"""Read health and ammo from the vanilla Doom status bar.

Digits sit in fixed 14x16 cells. Each cell is matched with
``cv2.matchTemplate`` against ten ink templates cropped from a clip.
A reading is accepted only after the same value appears three frames
in a row.
"""

from pathlib import Path

import cv2
import numpy as np

from gameplay_events.events import AMMO_CHANGED, HEALTH_CHANGED, GameplayEvent, HudChange

DIGIT_WIDTH = 14
DIGIT_HEIGHT = 16
DIGIT_TOP = 171
DIGIT_COUNT = 3
AMMO_RIGHT = 44
HEALTH_RIGHT = 90
MATCH_THRESHOLD = 0.45
BLANK_INK_PIXELS = 12
PERSISTENCE = 3

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def ink_mask(bgr):
    """Bright gold, green, or red status-bar ink on the gray stone."""
    blue, green, red = cv2.split(bgr)
    blue = blue.astype(np.int16)
    green = green.astype(np.int16)
    red = red.astype(np.int16)
    gold = (red > 120) & (red > green + 20) & (green > blue)
    green_ink = (green > 80) & (green > red + 15) & (green > blue + 10)
    red_ink = (red > 100) & (red > green + 30) & (red > blue + 30)
    return ((gold | green_ink | red_ink).astype(np.uint8)) * 255


def load_templates(directory=None):
    """Load digit templates. A digit may have more than one image.

    The status bar recolors the same glyph (gold ammo, green health, red
    when the value is low). One extra crop per confused digit covers that.
    """
    directory = Path(directory) if directory is not None else _TEMPLATE_DIR
    grouped = {digit: [] for digit in range(10)}
    for path in sorted(directory.glob("*.png")):
        if not path.stem or not path.stem[0].isdigit():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        grouped[int(path.stem[0])].append(image)
    missing = [digit for digit, images in grouped.items() if not images]
    if missing:
        raise FileNotFoundError(f"missing digit templates: {missing}")
    return grouped


class TemporalFilter:
    """Accept a value only after it repeats for ``persistence`` frames."""

    def __init__(self, persistence=PERSISTENCE):
        self.persistence = persistence
        self.value = None
        self._pending = None
        self._count = 0

    def update(self, reading):
        if reading is None:
            self._pending = None
            self._count = 0
            return self.value
        if reading == self._pending:
            self._count += 1
        else:
            self._pending = reading
            self._count = 1
        if self._count >= self.persistence:
            self.value = reading
        return self.value


def _read_cell(ink, templates):
    if int(ink.sum() // 255) < BLANK_INK_PIXELS:
        return None, 1.0
    search = np.zeros((DIGIT_HEIGHT + 4, DIGIT_WIDTH + 4), np.uint8)
    search[2 : 2 + DIGIT_HEIGHT, 2 : 2 + DIGIT_WIDTH] = ink
    best_digit = None
    best_score = -1.0
    for digit, variants in templates.items():
        if not isinstance(variants, list):
            variants = [variants]
        for template in variants:
            if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_digit = digit
    if best_score < MATCH_THRESHOLD:
        return None, best_score
    return best_digit, best_score


def read_counter(frame_bgr, right_edge, templates):
    """Read one right-aligned three-digit counter.

    Returns ``(value, confidence)``. ``value`` is ``None`` when a cell
    is unreadable. Confidence is the weakest digit score.
    """
    digits = []
    scores = []
    for index in range(DIGIT_COUNT):
        x = right_edge - DIGIT_WIDTH * (DIGIT_COUNT - index)
        cell = frame_bgr[DIGIT_TOP : DIGIT_TOP + DIGIT_HEIGHT, x : x + DIGIT_WIDTH]
        digit, score = _read_cell(ink_mask(cell), templates)
        digits.append(digit)
        scores.append(score)
    if any(digit is None and score < MATCH_THRESHOLD for digit, score in zip(digits, scores)):
        return None, min(scores)
    # Leading blanks are spaces. A blank after a digit is a failed read.
    seen_digit = False
    characters = []
    used_scores = []
    for digit, score in zip(digits, scores):
        if digit is None:
            if seen_digit:
                return None, min(scores)
            continue
        seen_digit = True
        characters.append(str(digit))
        used_scores.append(score)
    if not characters:
        return None, min(scores)
    return int("".join(characters)), min(used_scores)


def read_hud(frame_bgr, templates):
    health, health_score = read_counter(frame_bgr, HEALTH_RIGHT, templates)
    ammo, ammo_score = read_counter(frame_bgr, AMMO_RIGHT, templates)
    return {
        "health": health,
        "health_confidence": health_score,
        "ammo": ammo,
        "ammo_confidence": ammo_score,
    }


class HudReader:
    def __init__(self, templates=None):
        self.templates = templates if templates is not None else load_templates()
        self.health_filter = TemporalFilter()
        self.ammo_filter = TemporalFilter()
        self.health = None
        self.ammo = None

    def update(self, frame_bgr, time):
        reading = read_hud(frame_bgr, self.templates)
        events = []
        changes = []
        for stat, smoother, raw_value, confidence, event_name in (
            ("health", self.health_filter, reading["health"], reading["health_confidence"], HEALTH_CHANGED),
            ("ammo", self.ammo_filter, reading["ammo"], reading["ammo_confidence"], AMMO_CHANGED),
        ):
            event, change = self._apply(stat, smoother, raw_value, confidence, time, event_name)
            if event is not None:
                events.append(event)
                changes.append(change)
        return reading, events, changes

    def _apply(self, stat, smoother, raw_value, confidence, time, event_name):
        previous = smoother.value
        accepted = smoother.update(raw_value)
        if stat == "health":
            self.health = accepted
        else:
            self.ammo = accepted
        if accepted is None or previous is None or accepted == previous:
            return None, None
        event = GameplayEvent(
            timestamp=time,
            event=event_name,
            confidence=confidence,
            old_value=previous,
            new_value=accepted,
        )
        change = HudChange(time=time, stat=stat, old_value=previous, new_value=accepted)
        return event, change
