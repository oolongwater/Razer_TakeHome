"""Detection scores on held-out CocoDoom maps, plus event and OCR accuracy."""

import argparse
import csv
import json
from pathlib import Path

import cv2

from gameplay_events.hud import HudReader, load_templates, read_hud
from gameplay_events.matching import events_from_intervals, match_events


def detection_scores(weights, data, device):
    from ultralytics import YOLO

    metrics = YOLO(weights).val(data=data, split="test", imgsz=320, device=device, verbose=False)
    names = metrics.names
    rows = []
    for index, name in names.items():
        precision = float(metrics.box.p[index])
        recall = float(metrics.box.r[index])
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": rows,
    }


def load_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def event_scores(prediction_dir, annotation_dir, clips_dir):
    dev = {}
    report = {}
    for annotation_path in sorted(Path(annotation_dir).glob("*_objects.csv")):
        clip = annotation_path.name.replace("_objects.csv", "")
        prediction_path = Path(prediction_dir) / clip / "events.json"
        if not prediction_path.exists():
            continue
        fps, clip_end = _clip_timing(clips_dir, clip)
        ground_truth = events_from_intervals(load_csv(annotation_path), fps, clip_end)
        hud_path = Path(annotation_dir) / f"{clip}_hud_events.csv"
        if hud_path.exists():
            for row in load_csv(hud_path):
                ground_truth.append({"timestamp": float(row["timestamp"]), "event": row["event"]})
        predicted = json.loads(prediction_path.read_text())
        rows = match_events(predicted, ground_truth)
        rows["view_overall"] = _without_hud(rows)
        bucket = dev if clip.endswith("lv08_022") else report
        bucket[clip] = rows
    return {"dev": dev, "report": report}


def _without_hud(rows):
    true_positives = false_positives = false_negatives = 0
    for name, row in rows.items():
        if name == "overall" or name.startswith("player_"):
            continue
        true_positives += row["tp"]
        false_positives += row["fp"]
        false_negatives += row["fn"]
    precision = (
        true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    )
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def ocr_scores(clips_dir, annotation_dir):
    """Compare raw and filtered readings with hand-labeled frames."""
    templates = load_templates()
    rows = []
    for annotation_path in sorted(Path(annotation_dir).glob("*_hud.csv")):
        clip = annotation_path.name.replace("_hud.csv", "")
        video_path = _find_video(clips_dir, clip)
        labels = load_csv(annotation_path)
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        reader = HudReader(templates)
        label_by_frame = {}
        for row in labels:
            frame_index = int(round(float(row["timestamp"]) * fps))
            label_by_frame[frame_index] = row
        last_needed = max(label_by_frame)
        raw_correct = raw_total = filtered_correct = filtered_total = 0
        frame_index = 0
        while frame_index <= last_needed:
            ok, frame = capture.read()
            if not ok:
                break
            reading = read_hud(frame, templates)
            reader.update(frame, frame_index / fps)
            label = label_by_frame.get(frame_index)
            if label is not None:
                for stat in ("health", "ammo"):
                    expected_text = str(label.get(stat, "")).strip()
                    if expected_text == "":
                        continue
                    expected = int(expected_text)
                    raw_total += 1
                    filtered_total += 1
                    if reading[stat] == expected:
                        raw_correct += 1
                    accepted = reader.health if stat == "health" else reader.ammo
                    if accepted == expected:
                        filtered_correct += 1
            frame_index += 1
        capture.release()
        rows.append(
            {
                "clip": clip,
                "raw_accuracy": raw_correct / raw_total if raw_total else 0.0,
                "filtered_accuracy": filtered_correct / filtered_total if filtered_total else 0.0,
                "labeled_values": raw_total,
            }
        )
    return rows


def _find_video(clips_dir, clip):
    path = Path(clips_dir) / f"{clip}.avi"
    if not path.exists():
        raise FileNotFoundError(f"No video {path}")
    return path


def _clip_timing(clips_dir, clip):
    capture = cv2.VideoCapture(str(_find_video(clips_dir, clip)))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    capture.release()
    return fps, max(frames - 1, 0) / fps


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection, events, and OCR.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolo11n-doom.pt"))
    parser.add_argument("--data", default="data/yolo/data.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--predictions", type=Path, default=Path("outputs"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations"))
    parser.add_argument("--clips", type=Path, default=Path("data/clips"))
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()

    report = {}
    if not args.skip_detection:
        print("detection on held-out maps...")
        report["detection"] = detection_scores(args.weights, args.data, args.device)
    if any(args.annotations.glob("*_objects.csv")):
        report["events"] = event_scores(args.predictions, args.annotations, args.clips)
    if any(args.annotations.glob("*_hud.csv")):
        report["ocr"] = ocr_scores(args.clips, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
