"""YOLO11n detection plus ByteTrack identities on the gameplay view."""

from ultralytics import YOLO

from gameplay_events.events import Detection

VIEW_HEIGHT = 168
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


class DetectorTracker:
    def __init__(self, weights, conf=0.25, imgsz=320, view_height=VIEW_HEIGHT, device="mps"):
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
            if is_weapon_sprite(bbox, self.view_height):
                continue
            detections.append(
                Detection(
                    frame_index=frame_index,
                    time=time,
                    track_id=int(track_id),
                    class_name=names[int(class_id)],
                    confidence=float(score),
                    bbox=bbox,
                )
            )
        return detections
