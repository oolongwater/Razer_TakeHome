"""Contact sheets and boundary crops for hand labeling.

These images are the raw gameplay view. They never include detector boxes.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

VIEW_HEIGHT = 168
MIN_SPRITE = 8


def _view(frame):
    if frame.shape[1] != 320 or frame.shape[0] != 200:
        frame = cv2.resize(frame, (320, 200), interpolation=cv2.INTER_AREA)
    view = frame[:VIEW_HEIGHT].copy()
    cv2.rectangle(view, (2, 2), (2 + MIN_SPRITE, 6), (255, 255, 255), -1)
    return view


def _load(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def write_sheets(video_path, output_dir, step, columns, thumb_width):
    frames = _load(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_height = int(round(thumb_width * VIEW_HEIGHT / 320))
    thumbs = []
    for index in range(0, len(frames), step):
        thumb = cv2.resize(_view(frames[index]), (thumb_width, thumb_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            thumb,
            str(index),
            (2, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    blank = np.zeros_like(thumbs[0])
    sheet_index = 0
    for start in range(0, len(thumbs), columns * 6):
        cells = thumbs[start : start + columns * 6]
        rows = []
        for row_start in range(0, len(cells), columns):
            row = cells[row_start : row_start + columns]
            row += [blank] * (columns - len(row))
            rows.append(cv2.hconcat(row))
        sheet = cv2.vconcat(rows)
        # Keep each sheet to a height a person can scan.
        cv2.imwrite(str(output_dir / f"sheet_{sheet_index:02d}.png"), sheet)
        sheet_index += 1
    print(f"{len(frames)} frames, {sheet_index} sheets -> {output_dir}")


def write_boundaries(video_path, intervals_path, output_dir, pad):
    frames = _load(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(intervals_path).open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for edge in ("first_frame", "last_frame"):
            center = int(row[edge])
            crops = []
            for index in range(center - pad, center + pad + 1):
                if index < 0 or index >= len(frames):
                    continue
                crop = cv2.resize(_view(frames[index]), (160, 84), interpolation=cv2.INTER_NEAREST)
                color = (0, 255, 0) if index == center else (0, 255, 255)
                cv2.putText(
                    crop,
                    str(index),
                    (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                crops.append(crop)
            if not crops:
                continue
            name = f"{row['object']}_{edge}_{center}.png"
            cv2.imwrite(str(output_dir / name), cv2.hconcat(crops))
    print(f"{len(rows)} intervals -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Raw-frame labeling sheets. No model overlays.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--intervals", type=Path, default=None)
    parser.add_argument("--pad", type=int, default=8)
    args = parser.parse_args()
    if args.intervals is None:
        write_sheets(args.video, args.output, args.step, args.columns, args.thumb_width)
    else:
        write_boundaries(args.video, args.intervals, args.output, args.pad)


if __name__ == "__main__":
    main()
