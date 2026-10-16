"""YOLO11n detection plus ByteTrack identities on the gameplay view."""

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import STrack

from gameplay_events.events import Detection

# ByteTrack leaves a brand-new track inactive until a second frame matches it,
# and it deletes that track if the next frame does not. A monster seen clearly
# for one frame, then at a low score, never reaches the event builder.
_byte_activate = STrack.activate


def _activate_on_first_sighting(self, kalman_filter, frame_id):
    _byte_activate(self, kalman_filter, frame_id)
    self.is_activated = True


STrack.activate = _activate_on_first_sighting

VIEW_HEIGHT = 168
MIN_SPRITE_PX = 8
WEAPON_MIN_WIDTH = 60
WEAPON_CENTER_X_MIN = 90
WEAPON_CENTER_X_MAX = 230


def is_weapon_sprite(bbox, view_height=VIEW_HEIGHT):
    """The player's gun sits on the bottom edge and is not a world object."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    center_x = (x1 + x2) / 2.0
    touches_bottom = y2 >= view_height - 2 and y1 >= view_height * 0.62
    centered = WEAPON_CENTER_X_MIN <= center_x <= WEAPON_CENTER_X_MAX
    return touches_bottom and centered and width >= WEAPON_MIN_WIDTH


def is_too_small(bbox, class_name):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if class_name == "enemy":
        return height < MIN_SPRITE_PX
    return max(width, height) < MIN_SPRITE_PX


class DetectorTracker:
    def __init__(self, weights, conf=0.1, imgsz=320, view_height=VIEW_HEIGHT, device="mps"):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.view_height = view_height
        self.device = device

    def observe(self, view_bgr, frame_index, time):
        results = self.model.track(
            view_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        names = result.names
        detections = []
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()
        for box, class_id, score, track_id in zip(xyxy, classes, scores, track_ids):
            bbox = tuple(float(value) for value in box)
            class_name = names[int(class_id)]
            if is_weapon_sprite(bbox, self.view_height) or is_too_small(bbox, class_name):
                continue
            detections.append(
                Detection(
                    frame_index=frame_index,
                    time=time,
                    track_id=int(track_id),
                    class_name=class_name,
                    confidence=float(score),
                    bbox=bbox,
                )
            )
        return detections
