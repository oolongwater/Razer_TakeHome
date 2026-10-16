# Doom gameplay event detection

A Python pipeline that watches a public Doom II speedrun and writes structured gameplay events: an enemy appearing or leaving, an item being picked up, and the health or ammo counter changing. Each event has a timestamp and a confidence. Enemy events also carry a relative bearing and a near/mid/far proximity, drawn as a small radar on the annotated video.

## Approach

The frame is split into two regions. At the vanilla 320x200 resolution, rows 0–167 are the gameplay view and rows 168–199 are the status bar. Other resolutions are resized to 320x200 first.

Two branches run on every frame and are merged at the end:

1. **View.** A fine-tuned YOLO11n detector finds enemies, health items, and ammo items. ByteTrack gives each box an id. A short state machine turns those tracks into events.
2. **Status bar.** Fixed digit cells are read with OpenCV template matching. A value is kept only after it repeats for three frames, and a change becomes a HUD event.

```
gameplay view  ->  YOLO11n  ->  ByteTrack  ->  event rules  --+
                                                               +-->  events.json / events.csv
status bar     ->  template OCR  ->  3-frame filter  ->  HUD  --+     annotated.mp4 + radar
```

### Classes

CocoDoom labels sprites with Chocolate Doom's object names. Those names are mapped to three classes:

| Class | Examples |
| --- | --- |
| `enemy` | imp, demon, baron, cacodemon, lost soul, and the other Doom II monsters |
| `health_item` | stimpack, medikit, health bonus, soulsphere, megasphere |
| `ammo_item` | clips, shells, rockets, cells, and their boxes |

CocoDoom keeps the monster's name on a corpse. Corpse sprites are wide and flat, so any enemy box with height/width below 0.6 is left out of training. A kill then shows up as the track disappearing rather than as a second class. Wide live monsters (a spider mastermind seen from the front, a mancubus side-on) can be dropped by the same rule. That is a known limit of the filter.

Projectiles, barrels, armor, weapons on the ground, and the player's gun sprite are not classes.

### How an event is derived

A track is **confirmed** once it is seen in 3 of the last 5 frames. Its class is the majority vote of its detections, and its confidence is the mean detector score over those frames. The event time is the first frame of the track, not the frame that crossed the threshold.

| Event | When it fires |
| --- | --- |
| `enemy_detected` | An enemy track is confirmed. |
| `enemy_disappeared` | A confirmed enemy track has not been seen for 0.5 s. The time is the last frame it was seen. |
| `item_detected` | A health or ammo item track is confirmed. |
| `item_picked_up` | An item track ends in the bottom-centre of the view (where the player walks into it), or the matching counter goes up within 1 s after the item was last seen. One increase is paired with one item. An item that only slides off the side of the screen does not count. |
| `player_health_changed` | The filtered health reading changes. `old_value`, `new_value`, and `delta` are stored. |
| `player_ammo_changed` | The same, for the ammo counter of the weapon in hand. |

A single-frame detection, or a one-frame misread of a digit, does not emit an event.

The player's gun is not a world object. A box that sits on the bottom edge, is centred, and is at least 60 pixels wide is dropped before tracking.

### Status bar

Health and ammo use the vanilla three-digit cells: ammo occupies x = 2–44 and health x = 48–90, both at y = 171–187, each digit 14×16 and right-aligned. Each cell is turned into an ink mask (gold, green, or red on the gray stone) and matched with `cv2.matchTemplate` against digit templates cropped from a clip. Several digits have a second crop, because the same glyph is redrawn in a different palette and one template is not enough (a green zero was being read as an eight, a gold two as an eight).

A new reading is accepted only after the same number appears on three consecutive frames. Confidence is the weakest digit's match score.

### Relative radar

Bearing is `atan((x_center - 160) / 160)` in degrees. Ahead is 0 and right is positive, which matches Doom's 90-degree horizontal field of view. Proximity is the box height: near at 40 pixels or more, mid at 18 or more, otherwise far. The radar is an inset on the annotated video, and the same bearing and proximity are attached to enemy events. This is a direction and a rough distance, not a map position.

## Model and tools

- **Detector:** Ultralytics YOLO11n, fine-tuned from the COCO checkpoint. Image size 320, 12 epochs, batch 16, on Apple MPS. CocoDoom's training frames are sampled every few tics and are near-duplicates, and a full pass is about 18 minutes per epoch on an M2, so training uses a random 15% of the training images. The held-out test maps are not subsampled.
- **Tracker:** Ultralytics ByteTrack (`bytetrack.yaml`). BoT-SORT was tried on the first eight seconds of MAP04. Median track length moved from 3 frames to 5, which still does not hold an identity through a turn, so the pipeline stays on ByteTrack.
- **OCR:** OpenCV template matching. No Tesseract.
- **Everything else:** NumPy, pandas, pytest.

## Data

All of it is public. Nothing here was rendered for this project.

- **[CocoDoom v1.0.1 standard archive](https://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.1.tar.gz)** (Oxford VGG, about 2.3 GB). Doom II frames from human speedrun demos with COCO boxes. These frames have no status bar, so they train and score the detector only. The published map split is `(map_id - 1) % 4`: 0 and 1 train, 2 validation, 3 test. Test maps include MAP04, MAP08, MAP12, MAP16, MAP20, MAP24, MAP28, and MAP32.
- **[Doom2_UVlevels](https://archive.org/details/Doom2_UVlevels)** on the Internet Archive. Compet-N UV speedruns at 320×200, 35 tic recordings stored at 30 fps, with the classic status bar. The three clips are held-out maps:
  - `Doom2_UV_lv04_028.avi` (MAP04, The Focus)
  - `Doom2_UV_lv12_043.avi` (MAP12, The Factory)
  - `Doom2_UV_lv20_034.avi` (MAP20, Gotcha!)

## Results

Detection scores are precision, recall, and F1 at IoU 0.5 on the CocoDoom test maps (5,804 images, maps the detector never trained on), plus Ultralytics mAP50. Event scores match each hand-labeled event to one prediction of the same type within ±0.5 s. OCR scores compare health and ammo read off the status bar once a second, for the raw digit match and for the 3-frame filter.

`scripts/evaluate.py` writes `outputs/metrics.json`. Each clip also has `outputs/<clip>/events.json`, `events.csv`, `annotated.mp4`, and a few screenshots.

### Detection

mAP50 is 0.461 and mAP50-95 is 0.284.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| enemy | 0.684 | 0.480 | 0.564 |
| health_item | 0.755 | 0.421 | 0.541 |
| ammo_item | 0.669 | 0.282 | 0.397 |

Recall is the weak side. Small sprites and the flat-corpse filter both drop boxes the test set still counts. These precision and recall figures are Ultralytics' validation scores, taken at the confidence that maximises F1. The video pipeline keeps detections at confidence 0.25. mAP50 averages across confidence thresholds.

### Events

HUD changes match the hand labels on all three clips: health and ammo event F1 is 1.00 (MAP04 12 health + 18 ammo, MAP12 5 health, MAP20 15 health + 19 ammo).

Enemy and item events are lower, because a fast turn splits one monster into several track ids. Each new id is another `enemy_detected` / `enemy_disappeared`, so precision falls while recall stays higher.

The overall row counts every event type. The view row leaves out health and ammo changes, which are the easy matches, so it is the score for enemies and items alone.

| Clip | Overall P | Overall R | Overall F1 | View P | View R | View F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAP04 | 0.347 | 0.806 | 0.485 | 0.204 | 0.667 | 0.313 |
| MAP12 | 0.230 | 0.472 | 0.309 | 0.174 | 0.387 | 0.240 |
| MAP20 | 0.458 | 0.790 | 0.580 | 0.205 | 0.536 | 0.297 |

Per type, `enemy_detected` F1 is 0.46 / 0.50 / 0.50 on the three clips. `enemy_disappeared` is 0.25 / 0.00 / 0.20: the track expires within a few tenths of a second, and the hand label marks the moment the monster actually leaves the fight, which is often more than 0.5 s later. `item_picked_up` F1 is 0.36 / 0.00 / 0.00. A pickup whose sprite was never confirmed is missed. Separate track ids that each expire in the bottom-centre still each emit a pickup, because that rule does not wait for the counter.

`item_detected` was only hand-labeled when an item was sitting still in view (one medikit on MAP12, which was found). Other item sprites were not given their own rows, so extra `item_detected` predictions are unmatched.

### OCR

| Clip | Raw accuracy | Filtered accuracy | Labeled values |
| --- | ---: | ---: | ---: |
| MAP04 | 1.000 | 0.931 | 58 |
| MAP12 | 1.000 | 0.973 | 75 |
| MAP20 | 0.986 | 0.903 | 72 |

The filter is a little less accurate on these once-a-second samples because it waits two extra frames before accepting a new number, so a sample that lands on the change still shows the previous value. That same wait is what keeps a one-frame misread from becoming an event. The digit templates, including the alternate crops, were cut from these same three clips, so the OCR numbers are in-sample.

Hand labels:

- `annotations/*_hud.csv` — health and ammo read off the status bar at each whole second. Blank ammo means the fist is selected and the counter is empty. Intermission screens are omitted.
- `annotations/*_events.csv` — HUD changes, plus enemies and pickups read from the view at half-second steps. A monster that is only on screen for a flash, or that is a few pixels tall, may be missing from this list. A HUD change is timestamped at the first frame where the raw template read holds the new value for three frames. Those values were checked against the once-a-second visual reads in `*_hud.csv`. The event time is therefore close to the filter's own accept time (about two frames earlier), so the HUD event F1 mostly checks that the filter emits the same transitions the raw reader stabilized on, at the right moment.

## Failure cases

- **Damage flash.** A red pain flash repaints the status bar and the ink mask misses the digits. The filter holds the previous value, so a one-frame flash does not emit a change, but a flash that lasts through the sample frame is a raw-OCR miss.
- **Fist selected.** The ammo cell is blank. The filter keeps the last number, so it does not announce that the counter disappeared, and it can lag behind a later weapon switch.
- **Palette twins.** A few glyphs (2/8, 3/9, 5/6) stay close even with a second template. A misread that lasts three frames becomes a false HUD event.
- **Corpse filter.** Flat enemy boxes are not trained, so a live monster seen edge-on can be missed, and a death is only a disappearance. There is no `enemy_killed` event.
- **Gun and muzzle flash.** The gun filter drops a wide box on the bottom edge. A monster standing in that same spot, or a flash that covers it, can be dropped or can split the track.
- **Fast turns.** ByteTrack has no camera-motion compensation. A quick turn gives a new id, which looks like one enemy disappearing and another appearing.
- **Items off to the side.** An item that leaves the view without a later counter increase is not a pickup. A pickup whose sprite was never confirmed for three frames is missed even if the counter moves. One increase is paired with one item. A weapon switch that raises the ammo number can still confirm one nearby ammo sprite. Several ids that die in the bottom-centre each count as their own pickup.
- **Radar.** Bearing assumes a 90-degree view and a centred player. Proximity is sprite height, so a lost soul and a baron at the same distance are not the same band.

## Reproduce

```bash
# Python 3.12
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

bash scripts/fetch_data.sh          # CocoDoom (~2.3 GB) and the three clips
python scripts/build_dataset.py     # YOLO labels under data/yolo
python scripts/train.py             # writes models/yolo11n-doom.pt
python -m gameplay_events.pipeline data/clips/Doom2_UV_lv04_028.avi
python scripts/evaluate.py
pytest
```

`gameplay-events VIDEO --output outputs/NAME` is the same entry point. `--no-hud` skips the status bar, for a full-screen view with no counters.
