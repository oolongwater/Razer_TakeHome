"""Pickup flashes tint the gameplay view yellow for a few frames.

A damage flash is redder than it is yellow, so it does not count.
"""

import numpy as np

YELLOW_JUMP = 20.0
DEBOUNCE_SECONDS = 0.3


class FlashDetector:
    def __init__(self):
        self.history = []
        self.last_onset = None

    def update(self, view_bgr, time):
        view = view_bgr.astype(np.float32)
        blue = view[:, :, 0]
        green = view[:, :, 1]
        red = view[:, :, 2]
        yellow = float(((red + green) / 2.0 - blue).mean())
        redness = float((red - (green + blue) / 2.0).mean())
        self.history.append((yellow, redness))
        if len(self.history) < 7:
            return False
        window = self.history[-7:-2]
        base_yellow = sum(item[0] for item in window) / len(window)
        base_red = sum(item[1] for item in window) / len(window)
        yellow_jump = yellow - base_yellow
        red_jump = redness - base_red
        if yellow_jump < YELLOW_JUMP or red_jump >= yellow_jump:
            return False
        if self.last_onset is not None and time - self.last_onset < DEBOUNCE_SECONDS:
            return False
        self.last_onset = time
        return True
