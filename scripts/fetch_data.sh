#!/bin/bash
# Download the public inputs. CocoDoom is about 2.3 GB; the clips are a few MB.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw" "$ROOT/data/clips"

COCO_URL="https://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.1.tar.gz"
COCO_DEST="$ROOT/data/raw/cocodoom-v1.0.1.tar.gz"
if [[ ! -f "$COCO_DEST" ]]; then
  curl -L --fail --retry 5 -o "$COCO_DEST" "$COCO_URL"
fi
if [[ ! -d "$ROOT/data/cocodoom/run1" ]]; then
  tar -xzf "$COCO_DEST" -C "$ROOT/data"
fi

BASE="https://archive.org/download/Doom2_UVlevels"
for name in Doom2_UV_lv04_028.avi Doom2_UV_lv08_022.avi Doom2_UV_lv12_043.avi Doom2_UV_lv20_034.avi; do
  if [[ ! -f "$ROOT/data/clips/$name" ]]; then
    curl -L --fail --retry 5 -o "$ROOT/data/clips/$name" "$BASE/$name"
  fi
done
echo "data ready"
