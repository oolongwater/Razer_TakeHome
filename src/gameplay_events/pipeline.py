"""Run detection, tracking, OCR, and the radar on one gameplay video."""

import argparse
import csv
import json
from pathlib import Path

import cv2

from gameplay_events.detect_track import VIEW_HEIGHT, DetectorTracker
from gameplay_events.events import EventBuilder
from gameplay_events.flash import FlashDetector
from gameplay_events.hud import HudReader, status_bar_visible
from gameplay_events.radar import RADAR_RADIUS, draw_radar, enemy_blips, locate

CANONICAL_WIDTH = 320
CANONICAL_HEIGHT = 200
RENDER_SCALE = 3

CLASS_COLOR = {
    "enemy": (40, 40, 220),
    "health_item": (40, 180, 40),
    "ammo_item": (40, 200, 220),
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


def _draw_overlay(frame, detections, hud_state, recent_events, blips):
    canvas = cv2.resize(
        frame,
        (CANONICAL_WIDTH * RENDER_SCALE, CANONICAL_HEIGHT * RENDER_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    for detection in detections:
        color = CLASS_COLOR.get(detection.class_name, (200, 200, 200))
        x1, y1, x2, y2 = [int(round(value * RENDER_SCALE)) for value in detection.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name[:5]} {detection.track_id}"
        cv2.putText(
            canvas,
            label,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    origin = (
        CANONICAL_WIDTH * RENDER_SCALE - RADAR_RADIUS - 12,
        RADAR_RADIUS + 12,
    )
    draw_radar(canvas, blips, origin, radius=RADAR_RADIUS)
    health = "--" if hud_state.get("health") is None else str(hud_state["health"])
    ammo = "--" if hud_state.get("ammo") is None else str(hud_state["ammo"])
    cv2.putText(
        canvas,
        f"HP {health}  AMMO {ammo}",
        (8, CANONICAL_HEIGHT * RENDER_SCALE - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    y = CANONICAL_HEIGHT * RENDER_SCALE - 18
    for event in recent_events[-3:]:
        text = f"{event.timestamp:5.1f}  {event.event}"
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1, cv2.LINE_AA)
        y -= 16
    return canvas


def analyze_video(video_path, weights, output_dir, conf=0.1, read_hud=True, save_video=True):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = DetectorTracker(weights, conf=conf)
    builder = EventBuilder(locate=locate)
    flashes = FlashDetector()
    hud = HudReader() if read_hud else None
    writer = None
    screenshots = 0
    recent = []
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
        before = len(builder.events)
        builder.update(frame_index, time, detections, hud_changes, flash=flash)
        new_events = hud_events + builder.events[before:]
        recent.extend(new_events)
        if save_video or screenshots < 8:
            blips = enemy_blips(detections)
            canvas = _draw_overlay(frame, detections, hud_state, recent, blips)
            if save_video:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (canvas.shape[1], canvas.shape[0]),
                    )
                writer.write(canvas)
            if new_events and screenshots < 8:
                cv2.imwrite(str(output_dir / f"frame_{frame_index:05d}.png"), canvas)
                screenshots += 1
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
        read_hud=not args.no_hud,
    )
    print(f"{len(events)} events -> {output}")


if __name__ == "__main__":
    main()
