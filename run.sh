#!/usr/bin/env bash
#
# Quick demo driver for the GPU-accelerated batch image processing pipeline.
#
#   1. Builds bin/image_processor if it is missing.
#   2. Populates data/input/ with a few sample images if it is empty.
#   3. Runs every kernel over the input directory in benchmark mode.
#
# NOTE: The USC SIPI test images are distributed as TIFF, which stb_image
# cannot decode. This script downloads them and converts each to PNG using
# ImageMagick (`magick`/`convert`) or macOS `sips`. If neither is available it
# falls back to generated PNGs so the demo still runs.

set -euo pipefail

cd "$(dirname "$0")"

BIN="bin/image_processor"
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"

# ---------------------------------------------------------------------------
# 1. Build if needed.
# ---------------------------------------------------------------------------
if [[ ! -x "$BIN" ]]; then
  echo "==> $BIN not found; building with make..."
  make
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 2. Fetch sample images if the input folder is empty.
# ---------------------------------------------------------------------------
shopt -s nullglob
existing=("$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.bmp)
shopt -u nullglob

if [[ ${#existing[@]} -eq 0 ]]; then
  echo "==> $INPUT_DIR is empty; fetching sample images..."

  # Pick a downloader.
  if command -v curl >/dev/null 2>&1; then
    DL() { curl -fsSL -o "$1" "$2"; }
  elif command -v wget >/dev/null 2>&1; then
    DL() { wget -q -O "$1" "$2"; }
  else
    echo "    neither curl nor wget is installed; cannot download." >&2
    DL() { return 1; }
  fi

  # Pick a TIFF -> PNG converter.
  convert_to_png() {  # $1 = src tiff, $2 = dst png
    if command -v magick >/dev/null 2>&1; then
      magick "$1" "$2"
    elif command -v convert >/dev/null 2>&1; then
      convert "$1" "$2"
    elif command -v sips >/dev/null 2>&1; then
      sips -s format png "$1" --out "$2" >/dev/null
    else
      return 1
    fi
  }

  SIPI_BASE="https://sipi.usc.edu/database/misc"
  # name:sipi_id
  SAMPLES=("baboon:4.2.03" "peppers:4.2.07" "airplane:4.2.05" "boat:boat.512")

  got_any=0
  for pair in "${SAMPLES[@]}"; do
    name="${pair%%:*}"
    id="${pair##*:}"
    tiff="$INPUT_DIR/$name.tiff"
    png="$INPUT_DIR/$name.png"
    echo "    downloading $name ($id.tiff)..."
    if DL "$tiff" "$SIPI_BASE/$id.tiff"; then
      if convert_to_png "$tiff" "$png"; then
        rm -f "$tiff"
        got_any=1
        echo "      -> $png"
      else
        echo "      !! no TIFF converter (ImageMagick/sips); left $tiff in place" >&2
        echo "         (stb_image cannot read TIFF — convert it to PNG to use it)" >&2
      fi
    else
      echo "      !! download failed for $name" >&2
    fi
  done

  # Fallback: synthesize a PNG so the demo always has something to process.
  if [[ $got_any -eq 0 ]]; then
    echo "    no usable images fetched; generating a synthetic PNG fallback..."
    if command -v magick >/dev/null 2>&1; then
      magick -size 512x512 gradient:black-white "$INPUT_DIR/gradient.png"
    elif command -v convert >/dev/null 2>&1; then
      convert -size 512x512 gradient:black-white "$INPUT_DIR/gradient.png"
    else
      echo "    !! cannot generate a fallback image either." >&2
      echo "    Please place a .png/.jpg/.bmp into $INPUT_DIR and re-run." >&2
      exit 1
    fi
  fi
else
  echo "==> Using ${#existing[@]} existing image(s) in $INPUT_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Run the pipeline over the whole input directory.
# ---------------------------------------------------------------------------
echo "==> Running pipeline..."
"./$BIN" -i "$INPUT_DIR/" -o "$OUTPUT_DIR/" -k all -b

echo "Done — check $OUTPUT_DIR/ for results"
