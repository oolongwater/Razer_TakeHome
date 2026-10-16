"""Turn tracks and HUD readings into gameplay events.

A track is confirmed after it is seen in 3 of 5 frames. Its class is the
majority vote of those detections. It expires after half a second unseen.
"""

from collections import Counter
from dataclasses import dataclass, field

ENEMY_DETECTED = "enemy_detected"
ENEMY_DISAPPEARED = "enemy_disappeared"
ITEM_DETECTED = "item_detected"
ITEM_PICKED_UP = "item_picked_up"
HEALTH_CHANGED = "player_health_changed"
AMMO_CHANGED = "player_ammo_changed"

CONFIRM_HITS = 3
CONFIRM_WINDOW = 5
EXPIRE_SECONDS = 0.5
PICKUP_WINDOW_SECONDS = 1.0

# Bottom-centre of the 320x168 view, where the player walks into an item.
PICKUP_X_MIN = 100
PICKUP_X_MAX = 220
PICKUP_CENTER_Y_MIN = 120

STAT_FOR_CLASS = {"health_item": "health", "ammo_item": "ammo"}


@dataclass
class Detection:
    frame_index: int
    time: float
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple


@dataclass
class HudChange:
    time: float
    stat: str
    old_value: int
    new_value: int
    used: bool = False

    @property
    def increased(self):
        return self.new_value > self.old_value


@dataclass
class GameplayEvent:
    timestamp: float
    event: str
    confidence: float
    track_id: int | None = None
    class_name: str | None = None
    bearing: float | None = None
    proximity: str | None = None
    old_value: int | None = None
    new_value: int | None = None

    def to_dict(self):
        raw = {
            "timestamp": round(float(self.timestamp), 3),
            "event": self.event,
            "confidence": round(float(self.confidence), 3),
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bearing": None if self.bearing is None else round(float(self.bearing), 1),
            "proximity": self.proximity,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": None
            if self.old_value is None or self.new_value is None
            else self.new_value - self.old_value,
        }
        return {key: value for key, value in raw.items() if value is not None}


@dataclass
class _Track:
    track_id: int
    hits: list = field(default_factory=list)
    confirmed: bool = False
    class_name: str | None = None
    pickup_emitted: bool = False

    @property
    def last(self):
        return self.hits[-1]

    def vote_class(self):
        names = [hit.class_name for hit in self.hits]
        return Counter(names).most_common(1)[0][0]

    def mean_confidence(self, frame_index):
        recent = [
            hit.confidence
            for hit in self.hits
            if hit.frame_index >= frame_index - CONFIRM_WINDOW + 1
        ]
        chosen = recent or [hit.confidence for hit in self.hits]
        return sum(chosen) / len(chosen)


def in_pickup_zone(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return PICKUP_X_MIN <= center_x <= PICKUP_X_MAX and center_y >= PICKUP_CENTER_Y_MIN


class EventBuilder:
    def __init__(self, locate=None):
        self.locate = locate
        self.tracks = {}
        self.events = []
        self.hud_changes = []
        # Items that left the view without a pickup, kept until the HUD window closes.
        self.expired_items = []

    def update(self, frame_index, time, detections, hud_changes=()):
        self.expired_items = [
            track
            for track in self.expired_items
            if time - track.last.time <= PICKUP_WINDOW_SECONDS
        ]
        for change in hud_changes:
            self.hud_changes.append(change)
            self._match_pickups(change)

        seen = set()
        for detection in detections:
            seen.add(detection.track_id)
            track = self.tracks.get(detection.track_id)
            if track is None:
                track = _Track(track_id=detection.track_id)
                self.tracks[detection.track_id] = track
            track.hits.append(detection)
            self._maybe_confirm(track, frame_index)

        for track_id, track in list(self.tracks.items()):
            if track_id in seen:
                continue
            if time - track.last.time < EXPIRE_SECONDS:
                continue
            self._expire(track)
            del self.tracks[track_id]

    def finish(self, time):
        """Close tracks that have already gone stale. Still-visible tracks stay open."""
        for track in list(self.tracks.values()):
            if time - track.last.time >= EXPIRE_SECONDS:
                self._expire(track)
        self.tracks.clear()

    def _maybe_confirm(self, track, frame_index):
        if track.confirmed:
            return
        window_start = frame_index - CONFIRM_WINDOW + 1
        hits = [hit for hit in track.hits if hit.frame_index >= window_start]
        if len(hits) < CONFIRM_HITS:
            return
        track.confirmed = True
        track.class_name = track.vote_class()
        onset = track.hits[0]
        event_name = ENEMY_DETECTED if track.class_name == "enemy" else ITEM_DETECTED
        bearing = proximity = None
        if track.class_name == "enemy" and self.locate is not None:
            bearing, proximity = self.locate(onset.bbox)
        self.events.append(
            GameplayEvent(
                timestamp=onset.time,
                event=event_name,
                confidence=track.mean_confidence(frame_index),
                track_id=track.track_id,
                class_name=track.class_name,
                bearing=bearing,
                proximity=proximity,
            )
        )

    def _expire(self, track):
        if not track.confirmed:
            return
        last = track.last
        if track.class_name == "enemy":
            bearing = proximity = None
            if self.locate is not None:
                bearing, proximity = self.locate(last.bbox)
            self.events.append(
                GameplayEvent(
                    timestamp=last.time,
                    event=ENEMY_DISAPPEARED,
                    confidence=last.confidence,
                    track_id=track.track_id,
                    class_name=track.class_name,
                    bearing=bearing,
                    proximity=proximity,
                )
            )
            return
        if track.pickup_emitted:
            return
        # A counter increase explains at most one item. The zone rule does not
        # need an increase, but it consumes one that follows so a second track
        # cannot claim the same change.
        if in_pickup_zone(last.bbox):
            self._claim_following_increase(track)
            self._emit_pickup(track, last.time)
            return
        if self._claim_following_increase(track):
            self._emit_pickup(track, last.time)
            return
        self.expired_items.append(track)

    def _match_pickups(self, change):
        """Attach this increase to the single closest item it follows."""
        if change.used or not change.increased:
            return
        best = None
        best_gap = None
        candidates = list(self.tracks.values()) + list(self.expired_items)
        for track in candidates:
            if track.pickup_emitted or not track.confirmed:
                continue
            if STAT_FOR_CLASS.get(track.class_name) != change.stat:
                continue
            gap = change.time - track.last.time
            if 0 <= gap <= PICKUP_WINDOW_SECONDS and (best_gap is None or gap < best_gap):
                best = track
                best_gap = gap
        if best is None:
            return
        change.used = True
        self._emit_pickup(best, best.last.time)

    def _claim_following_increase(self, track):
        """Mark the nearest later counter increase as used by this track."""
        stat = STAT_FOR_CLASS.get(track.class_name)
        if stat is None:
            return False
        last_time = track.last.time
        best = None
        best_gap = None
        for change in self.hud_changes:
            if change.used or change.stat != stat or not change.increased:
                continue
            gap = change.time - last_time
            if 0 <= gap <= PICKUP_WINDOW_SECONDS and (best_gap is None or gap < best_gap):
                best = change
                best_gap = gap
        if best is None:
            return False
        best.used = True
        return True

    def _emit_pickup(self, track, timestamp):
        track.pickup_emitted = True
        self.events.append(
            GameplayEvent(
                timestamp=timestamp,
                event=ITEM_PICKED_UP,
                confidence=sum(hit.confidence for hit in track.hits) / len(track.hits),
                track_id=track.track_id,
                class_name=track.class_name,
            )
        )
