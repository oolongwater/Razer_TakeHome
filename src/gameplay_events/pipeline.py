"""Run detection, tracking, OCR, and the radar on one gameplay video."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from gameplay_events.detect_track import VIEW_HEIGHT, DetectorTracker
from gameplay_events.events import EventBuilder
from gameplay_events.flash import FlashDetector
from gameplay_events.hud import HudReader, status_bar_visible
from gameplay_events.radar import RADAR_RADIUS, draw_radar, enemy_blips, locate

CANONICAL_WIDTH = 320
CANONICAL_HEIGHT = 200
RENDER_SCALE = 3
PANEL_WIDTH = 320
TAG_SECONDS = 1.0

PLAYER_COLOR = (240, 240, 240)
CLASS_COLOR = {
    "enemy": (40, 40, 220),
    "health_item": (40, 180, 40),
    "ammo_item": (40, 200, 220),
}
CLASS_LABEL = {
    "enemy": "Enemy",
    "health_item": "Health item",
    "ammo_item": "Ammo item",
}
TAG_TEXT = {
    "enemy_detected": "DETECTED",
    "item_detected": "DETECTED",
    "enemy_disappeared": "GONE",
    "item_picked_up": "PICKED UP",
}


def canonicalize(frame):
    if frame.shape[1] != CANONICAL_WIDTH or frame.shape[0] != CANONICAL_HEIGHT:
        frame = cv2.resize(frame, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def write_events(events, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [event.to_dict() for event in events]
    rows.sort(key=lambda row: (row["timestamp"], row["event"]))
    json_path = output_dir / "events.json"
    csv_path = output_dir / "events.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = [
        "timestamp",
        "event",
        "confidence",
        "track_id",
        "class_name",
        "bearing",
        "proximity",
        "old_value",
        "new_value",
        "delta",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def _scaled_box(bbox):
    return [int(round(value * RENDER_SCALE)) for value in bbox]


def _draw_overlay(frame, detections, hud_state, recent_events, blips, builder, tags):
    view = cv2.resize(
        frame,
        (CANONICAL_WIDTH * RENDER_SCALE, CANONICAL_HEIGHT * RENDER_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    for detection in detections:
        display_id = builder.display_id(detection.track_id)
        track = builder.tracks.get(display_id)
        pending = track is None or not track.confirmed
        color = (160, 160, 160) if pending else CLASS_COLOR.get(detection.class_name, (200, 200, 200))
        x1, y1, x2, y2 = _scaled_box(detection.bbox)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 1 if pending else 2)
        if pending:
            label = f"pending {display_id}"
        else:
            label = f"{CLASS_LABEL[detection.class_name]} {display_id} {detection.confidence:.2f}"
        cv2.putText(view, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for tag in tags:
        if tag["bbox"] is None:
            cv2.putText(view, tag["text"], (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 2, cv2.LINE_AA)
            continue
        x1, y1, x2, y2 = _scaled_box(tag["bbox"])
        cv2.putText(view, tag["text"], (x1, min(view.shape[0] - 8, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    origin = (
        CANONICAL_WIDTH * RENDER_SCALE - RADAR_RADIUS - 12,
        RADAR_RADIUS + 12,
    )
    draw_radar(view, blips, origin, radius=RADAR_RADIUS)
    health = "--" if hud_state.get("health") is None else str(hud_state["health"])
    cv2.putText(
        view,
        f"Player HP {health}",
        (8, CANONICAL_HEIGHT * RENDER_SCALE - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        PLAYER_COLOR,
        1,
        cv2.LINE_AA,
    )
    panel = np.full((view.shape[0], PANEL_WIDTH, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "events", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    legend_y = 48
    legend = [("Player", PLAYER_COLOR)] + [(CLASS_LABEL[name], color) for name, color in CLASS_COLOR.items()]
    for name, color in legend:
        cv2.rectangle(panel, (12, legend_y - 10), (26, legend_y + 2), color, -1)
        cv2.putText(panel, name, (34, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
        legend_y += 18
    counts = {}
    for event in recent_events:
        counts[event.event] = counts.get(event.event, 0) + 1
    count_y = legend_y + 8
    for name in sorted(counts):
        cv2.putText(
            panel,
            f"{counts[name]:3d}  {name}",
            (12, count_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 200, 180),
            1,
            cv2.LINE_AA,
        )
        count_y += 14
    row_y = max(count_y + 16, 220)
    cv2.putText(panel, "last events", (12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    row_y += 18
    for event in recent_events[-12:]:
        ident = "-" if event.track_id is None else str(event.track_id)
        if event.old_value is not None and event.new_value is not None:
            detail = f"{event.old_value}->{event.new_value}"
        else:
            detail = f"{event.confidence:.2f}"
        text = f"{event.timestamp:5.1f} {event.event} #{ident} {detail}"
        cv2.putText(panel, text[:42], (12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 220, 255), 1, cv2.LINE_AA)
        row_y += 16
        if row_y > panel.shape[0] - 8:
            break
    return np.hstack([view, panel])


def analyze_video(video_path, weights, output_dir, conf=0.1, imgsz=416, read_hud=True, save_video=True):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = DetectorTracker(weights, conf=conf, imgsz=imgsz)
    builder = EventBuilder(locate=locate)
    flashes = FlashDetector()
    hud = HudReader() if read_hud else None
    writer = None
    recent = []
    tags = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = canonicalize(frame)
        time = frame_index / fps
        view = frame[:VIEW_HEIGHT]
        playing = status_bar_visible(frame) if read_hud else True
        detections = tracker.observe(view, frame_index, time) if playing else []
        hud_events = []
        hud_changes = []
        hud_state = {"health": None, "ammo": None}
        flash = False
        if hud is not None and playing:
            reading, hud_events, hud_changes = hud.update(frame, time)
            hud_state = {"health": hud.health, "ammo": hud.ammo, "raw": reading}
            flash = flashes.update(view, time)
        seen_boxes = {track_id: track.last.bbox for track_id, track in builder.tracks.items()}
        before_ids = {id(event) for event in builder.events}
        builder.update(frame_index, time, detections, hud_changes, flash=flash)
        new_builder_events = [event for event in builder.events if id(event) not in before_ids]
        new_events = hud_events + new_builder_events
        recent.extend(new_events)
        for detection in detections:
            seen_boxes[builder.display_id(detection.track_id)] = detection.bbox
        for event in new_builder_events:
            label = TAG_TEXT.get(event.event)
            if label is None:
                continue
            tags.append(
                {
                    "text": label,
                    "bbox": seen_boxes.get(event.track_id),
                    "until": time + TAG_SECONDS,
                }
            )
        tags[:] = [tag for tag in tags if tag["until"] >= time]
        if save_video:
            blips = enemy_blips(detections)
            canvas = _draw_overlay(frame, detections, hud_state, recent, blips, builder, tags)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_dir / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
        if frame_index % 200 == 0:
            print(f"{video_path.name} frame {frame_index}", flush=True)
        frame_index += 1

    last_time = max(frame_index - 1, 0) / fps
    builder.finish(last_time)
    capture.release()
    if writer is not None:
        writer.release()
    hud_events = [event for event in recent if event.event.startswith("player_")]
    events = sorted(hud_events + builder.events, key=lambda event: (event.timestamp, event.event))
    write_events(events, output_dir)
    return events


def main():
    parser = argparse.ArgumentParser(description="Detect Doom gameplay events in a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/yolo11n-doom.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--no-hud", action="store_true", help="Video has no status bar (full-screen view).")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = Path("outputs") / args.video.stem
    events = analyze_video(
        args.video,
        args.weights,
        output,
        conf=args.conf,
        imgsz=args.imgsz,
        read_hud=not args.no_hud,
    )
    print(f"{len(events)} events -> {output}")


if __name__ == "__main__":
    main()
