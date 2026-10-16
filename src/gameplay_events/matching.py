"""One-to-one matching of predicted events to ground truth."""

from gameplay_events.events import (
    CONFIRM_HITS,
    ENEMY_DETECTED,
    ENEMY_DISAPPEARED,
    EXPIRE_SECONDS,
    ITEM_DETECTED,
    ITEM_PICKED_UP,
)

TOLERANCE_SECONDS = 0.5

_DETECTED = {"enemy": ENEMY_DETECTED, "health_item": ITEM_DETECTED, "ammo_item": ITEM_DETECTED}


def events_from_intervals(rows, fps, clip_end):
    """Turn visibility intervals into the same events the pipeline emits.

    Intervals of one object less than ``EXPIRE_SECONDS`` apart are one
    presence. A presence shorter than ``CONFIRM_HITS`` frames is ignored.
    Only enemies disappear, and an enemy presence that ends within
    ``EXPIRE_SECONDS`` of the clip does not, matching ``finish``. An item
    ends with ``item_picked_up`` when ``picked_up`` is set.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["object"]), []).append(row)
    events = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["first_frame"]))
        spans = []
        for row in ordered:
            first = int(row["first_frame"])
            last = int(row["last_frame"])
            picked = int(row["picked_up"])
            class_name = row["class_name"]
            if (
                spans
                and spans[-1]["class_name"] == class_name
                and (first / fps) - (spans[-1]["last"] / fps) < EXPIRE_SECONDS
            ):
                spans[-1]["last"] = max(spans[-1]["last"], last)
                spans[-1]["picked_up"] = max(spans[-1]["picked_up"], picked)
                continue
            spans.append(
                {
                    "class_name": class_name,
                    "first": first,
                    "last": last,
                    "picked_up": picked,
                }
            )
        for span in spans:
            if span["last"] - span["first"] + 1 < CONFIRM_HITS:
                continue
            events.append({"timestamp": span["first"] / fps, "event": _DETECTED[span["class_name"]]})
            end_time = span["last"] / fps
            if span["class_name"] != "enemy":
                if span["picked_up"]:
                    events.append({"timestamp": end_time, "event": ITEM_PICKED_UP})
            elif clip_end - end_time >= EXPIRE_SECONDS:
                events.append({"timestamp": end_time, "event": ENEMY_DISAPPEARED})
    events.sort(key=lambda event: (event["timestamp"], event["event"]))
    return events


def _timestamp(item):
    if isinstance(item, dict):
        return float(item["timestamp"])
    return float(item.timestamp)


def _name(item):
    if isinstance(item, dict):
        return item["event"]
    return item.event


def match_events(predicted, ground_truth, tolerance=TOLERANCE_SECONDS):
    """Match each ground-truth event to the closest unused prediction.

    Matching requires the same event type and a time gap of at most
    ``tolerance`` seconds. Returns per-type counts and an overall row.
    """
    types = sorted({_name(item) for item in predicted} | {_name(item) for item in ground_truth})
    rows = {}
    for event_type in types:
        predictions = [item for item in predicted if _name(item) == event_type]
        truths = [item for item in ground_truth if _name(item) == event_type]
        used = set()
        true_positives = 0
        for truth in sorted(truths, key=_timestamp):
            best_index = None
            best_gap = None
            truth_time = _timestamp(truth)
            for index, prediction in enumerate(predictions):
                if index in used:
                    continue
                gap = abs(_timestamp(prediction) - truth_time)
                if gap <= tolerance and (best_gap is None or gap < best_gap):
                    best_index = index
                    best_gap = gap
            if best_index is not None:
                used.add(best_index)
                true_positives += 1
        false_positives = len(predictions) - true_positives
        false_negatives = len(truths) - true_positives
        rows[event_type] = _rates(true_positives, false_positives, false_negatives)
    total_tp = sum(row["tp"] for row in rows.values())
    total_fp = sum(row["fp"] for row in rows.values())
    total_fn = sum(row["fn"] for row in rows.values())
    rows["overall"] = _rates(total_tp, total_fp, total_fn)
    return rows


def _rates(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
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
