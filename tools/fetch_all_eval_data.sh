#!/usr/bin/env bash
# Fetches the benchmark datasets used by the evaluation workflows into eval-data/<name>/
# (same seed -> same images everywhere). A dataset that fails to fetch is skipped with a
# warning, not fatal.
# Usage: tools/fetch_all_eval_data.sh [samples_per_class] [name ...]
#   names: defactify, mj-dalle-sd-nbp (the default set)
set -u
PER_CLASS="${1:-250}"
shift || true
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(defactify mj-dalle-sd-nbp)

fetch() {
  case "$1" in
    defactify)
      # MS-COCO real photos + SD 2.1 / SDXL / SD3 / DALL-E 3 / Midjourney v6 made from the
      # same captions. Label_A = real/AI, Label_B = source id; the id -> name order follows
      # the dataset card's listing order, and fetch_eval_data.py fails if Label_A and
      # Label_B ever disagree about which images are real.
      python tools/fetch_eval_data.py \
        --dataset Rajarshi-Roy-research/Defactify_Image_Dataset \
        --out eval-data/defactify --per-class "$PER_CLASS" \
        --label-col Label_A --ai-values 1 --real-values 0 \
        --generator-col Label_B --generator-names real,sd21,sdxl,sd3,dalle3,midjourney6
      ;;
    mj-dalle-sd-nbp)
      # Midjourney / DALL-E / SD / Nano Banana Pro vs real (labels from ClassLabel names).
      python tools/fetch_eval_data.py \
        --dataset julienlucas/midjourney-dalle-sd-nanobananapro-dataset \
        --out eval-data/mj-dalle-sd-nbp --per-class "$PER_CLASS"
      ;;
    *)
      echo "::error::unknown dataset $1"
      return 1
      ;;
  esac
}

for name in "${NAMES[@]}"; do
  fetch "$name" || echo "::warning::$name fetch failed"
done
