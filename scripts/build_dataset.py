"""Export the CocoDoom map split to a three-class YOLO dataset.

Reads the published ``map-train.json``, ``map-val.json``, and ``map-test.json``
files. Enemy boxes flatter than ``MIN_ENEMY_ASPECT`` are dropped so corpses,
which keep their monster name, are not trained as live enemies.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from gameplay_events.classes import CLASS_NAMES, CLASS_OF, MIN_ENEMY_ASPECT, class_index

SPLITS = ("train", "val", "test")


def yolo_row(class_id, bbox, width, height):
    x, y, box_w, box_h = bbox
    center_x = min(max((x + box_w / 2) / width, 0.0), 1.0)
    center_y = min(max((y + box_h / 2) / height, 0.0), 1.0)
    norm_w = min(max(box_w / width, 0.0), 1.0)
    norm_h = min(max(box_h / height, 0.0), 1.0)
    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def keep_annotation(class_name, bbox):
    _x, _y, box_w, box_h = bbox
    if box_w < 2 or box_h < 2:
        return False
    if class_name == "enemy" and box_h / box_w < MIN_ENEMY_ASPECT:
        return False
    return True


def export_split(coco, image_root, split_root, split_name):
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    by_image = {}
    kept_counts = Counter()
    dropped_flat = 0
    for annotation in coco["annotations"]:
        doom_name = categories.get(annotation["category_id"])
        class_name = CLASS_OF.get(doom_name)
        if class_name is None:
            continue
        bbox = annotation["bbox"]
        if not keep_annotation(class_name, bbox):
            if class_name == "enemy":
                dropped_flat += 1
            continue
        by_image.setdefault(annotation["image_id"], []).append((class_name, bbox))
        kept_counts[class_name] += 1

    image_dir = split_root / "images" / split_name
    label_dir = split_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    for image in coco["images"]:
        source = image_root / image["file_name"]
        if not source.exists():
            missing += 1
            continue
        stem = image["file_name"].replace("/", "_").replace(".png", "")
        target = image_dir / f"{stem}.png"
        if not target.exists():
            os.symlink(source.resolve(), target)
        rows = []
        for class_name, bbox in by_image.get(image["id"], []):
            rows.append(yolo_row(class_index(class_name), bbox, image["width"], image["height"]))
        (label_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        written += 1
    return {"images": written, "missing": missing, "boxes": dict(kept_counts), "dropped_flat_enemies": dropped_flat}


def write_yaml(split_root):
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    text = (
        f"path: {split_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    (split_root / "data.yaml").write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Build a YOLO dataset from CocoDoom.")
    parser.add_argument("--cocodoom", type=Path, default=Path("data/cocodoom"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()
    summary = {}
    for split_name in SPLITS:
        annotation_path = args.cocodoom / f"map-{split_name}.json"
        print(f"loading {annotation_path}")
        coco = json.loads(annotation_path.read_text())
        summary[split_name] = export_split(coco, args.cocodoom, args.output, split_name)
        print(split_name, summary[split_name])
    write_yaml(args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
