"""Turn tracks and HUD readings into gameplay events.

A track is confirmed after it is seen in 3 of 5 frames. Its class is the
majority vote of those detections. It expires after half a second unseen.
"""

from collections import Counter
from dataclasses import dataclass, field

ENEMY_DETECTED = "enemy_detected"
ENEMY_DISAPPEARED = "enemy_disappeared"
ITEM_DETECTED = "item_detected"
ITEM_DISAPPEARED = "item_disappeared"
ITEM_PICKED_UP = "item_picked_up"
HEALTH_CHANGED = "player_health_changed"
AMMO_CHANGED = "player_ammo_changed"

CONFIRM_HITS = 3
CONFIRM_WINDOW = 5
EXPIRE_SECONDS = 0.5
PICKUP_WINDOW_SECONDS = 1.0
FLASH_PAIR_SECONDS = 0.3

# Bottom-centre of the 320x168 view, where the player walks into an item.
PICKUP_X_MIN = 100
PICKUP_X_MAX = 220
PICKUP_CENTER_Y_MIN = 120

STAT_FOR_CLASS = {"health_item": "health", "ammo_item": "ammo"}
CLASS_FOR_STAT = {"health": "health_item", "ammo": "ammo_item"}


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


def _retarget(detection, track_id):
    return Detection(
        detection.frame_index,
        detection.time,
        track_id,
        detection.class_name,
        detection.confidence,
        detection.bbox,
    )


def _box_height(bbox):
    return max(bbox[3] - bbox[1], 1.0)


def in_pickup_zone(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return PICKUP_X_MIN <= center_x <= PICKUP_X_MAX and center_y >= PICKUP_CENTER_Y_MIN


@dataclass
class _Flash:
    time: float
    consumed: bool = False


class EventBuilder:
    def __init__(self, locate=None):
        self.locate = locate
        self.tracks = {}
        self.events = []
        self.hud_changes = []
        self.flashes = []
        self.alias = {}
        # Items that left the view, kept so a flash just after can still claim them.
        self.expired_items = []

    def display_id(self, track_id):
        seen = set()
        while track_id in self.alias and track_id not in seen:
            seen.add(track_id)
            track_id = self.alias[track_id]
        return track_id

    def update(self, frame_index, time, detections, hud_changes=(), flash=False):
        self.expired_items = [
            track
            for track in self.expired_items
            if time - track.last.time <= PICKUP_WINDOW_SECONDS
        ]
        if flash:
            self.flashes.append(_Flash(time))
        for change in hud_changes:
            self.hud_changes.append(change)

        seen = set()
        for detection in detections:
            canonical = self.display_id(detection.track_id)
            if canonical != detection.track_id and canonical in self.tracks:
                parent = self.tracks[canonical]
                parent.hits.append(_retarget(detection, canonical))
                seen.add(canonical)
                continue
            track = self.tracks.get(detection.track_id)
            if track is None:
                track = _Track(track_id=detection.track_id)
                self.tracks[detection.track_id] = track
            track.hits.append(detection)
            self._maybe_confirm(track, frame_index, seen)
            if detection.track_id in self.alias:
                seen.add(self.display_id(detection.track_id))
            elif detection.track_id in self.tracks:
                seen.add(detection.track_id)

        for track_id, track in list(self.tracks.items()):
            if track_id in seen:
                continue
            if time - track.last.time < EXPIRE_SECONDS:
                continue
            if track.confirmed and self._awaiting_successor(track, frame_index):
                continue
            self._expire(track)
            del self.tracks[track_id]
        self._match_flashes(time)

    def finish(self, time):
        """Close tracks that have already gone stale. Still-visible tracks stay open."""
        for track in list(self.tracks.values()):
            if time - track.last.time >= EXPIRE_SECONDS:
                self._expire(track)
        self.tracks.clear()
        self._match_flashes(time + FLASH_PAIR_SECONDS)

    def _maybe_confirm(self, track, frame_index, seen):
        if track.confirmed:
            return
        window_start = frame_index - CONFIRM_WINDOW + 1
        hits = [hit for hit in track.hits if hit.frame_index >= window_start]
        if len(hits) < CONFIRM_HITS:
            return
        track.class_name = track.vote_class()
        parent = self._stitch_parent(track, seen)
        if parent is not None:
            parent.hits.extend(track.hits)
            self.alias[track.track_id] = parent.track_id
            del self.tracks[track.track_id]
            return
        track.confirmed = True
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

    def _stitch_parent(self, track, seen):
        onset = track.hits[0]
        best = None
        best_gap = None
        for parent in self.tracks.values():
            if parent.track_id == track.track_id or not parent.confirmed:
                continue
            if parent.track_id in seen or parent.class_name != track.class_name:
                continue
            gap = onset.time - parent.last.time
            if gap < 0 or gap > EXPIRE_SECONDS:
                continue
            ratio = _box_height(onset.bbox) / _box_height(parent.last.bbox)
            if ratio < 0.5 or ratio > 2:
                continue
            if best_gap is None or gap < best_gap:
                best = parent
                best_gap = gap
        return best

    def _awaiting_successor(self, track, frame_index):
        for other in self.tracks.values():
            if other.track_id == track.track_id or other.confirmed or not other.hits:
                continue
            gap = other.hits[0].time - track.last.time
            if gap < 0 or gap > EXPIRE_SECONDS:
                continue
            if other.vote_class() != track.class_name:
                continue
            if frame_index - other.last.frame_index > CONFIRM_WINDOW:
                continue
            return True
        return False

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
        self.expired_items.append(track)
        self.events.append(
            GameplayEvent(
                timestamp=last.time,
                event=ITEM_DISAPPEARED,
                confidence=last.confidence,
                track_id=self.display_id(track.track_id),
                class_name=track.class_name,
            )
        )

    def _match_flashes(self, time):
        for flash in self.flashes:
            if flash.consumed:
                continue
            paired = self._increase_near(flash.time)
            if paired is not None:
                change = paired
                change.used = True
                flash.consumed = True
                track = self._recent_item(change.stat, flash.time)
                self._emit_pickup(track, flash.time, CLASS_FOR_STAT[change.stat])
                continue
            if time - flash.time < FLASH_PAIR_SECONDS:
                continue
            zone_track = self._zone_item_before(flash.time)
            flash.consumed = True
            if zone_track is not None:
                self._emit_pickup(zone_track, flash.time, zone_track.class_name)
        self.flashes = [
            flash for flash in self.flashes if not flash.consumed or time - flash.time <= FLASH_PAIR_SECONDS
        ]

    def _increase_near(self, flash_time):
        best = None
        best_gap = None
        for change in self.hud_changes:
            if change.used or not change.increased or change.stat not in CLASS_FOR_STAT:
                continue
            gap = abs(change.time - flash_time)
            if gap <= FLASH_PAIR_SECONDS and (best_gap is None or gap < best_gap):
                best = change
                best_gap = gap
        return best

    def _item_tracks(self):
        return list(self.tracks.values()) + list(self.expired_items)

    def _recent_item(self, stat, flash_time):
        best = None
        best_gap = None
        for track in self._item_tracks():
            if not track.confirmed or track.pickup_emitted:
                continue
            if STAT_FOR_CLASS.get(track.class_name) != stat:
                continue
            gap = flash_time - track.last.time
            if gap < 0 or gap > PICKUP_WINDOW_SECONDS:
                continue
            if best_gap is None or gap < best_gap:
                best = track
                best_gap = gap
        return best

    def _zone_item_before(self, flash_time):
        best = None
        best_gap = None
        for track in self._item_tracks():
            if not track.confirmed or track.pickup_emitted:
                continue
            if track.class_name not in STAT_FOR_CLASS or not in_pickup_zone(track.last.bbox):
                continue
            gap = flash_time - track.last.time
            if gap < 0 or gap > FLASH_PAIR_SECONDS:
                continue
            if best_gap is None or gap < best_gap:
                best = track
                best_gap = gap
        return best

    def _emit_pickup(self, track, timestamp, class_name):
        if track is not None:
            if track.pickup_emitted:
                return
            track.pickup_emitted = True
            class_name = track.class_name
            track_id = self.display_id(track.track_id)
            confidence = sum(hit.confidence for hit in track.hits) / len(track.hits)
            self.events = [
                event
                for event in self.events
                if not (
                    event.event == ITEM_DISAPPEARED
                    and event.track_id == track_id
                    and abs(event.timestamp - track.last.time) < 1e-6
                )
            ]
        else:
            track_id = None
            confidence = 0.5
        self.events.append(
            GameplayEvent(
                timestamp=timestamp,
                event=ITEM_PICKED_UP,
                confidence=confidence,
                track_id=track_id,
                class_name=class_name,
            )
        )
