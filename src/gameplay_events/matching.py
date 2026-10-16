"""One-to-one matching of predicted events to ground truth."""

TOLERANCE_SECONDS = 0.5


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
