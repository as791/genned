#!/usr/bin/env bash
# Fetches the benchmark datasets used by the evaluation workflows into eval-data/<name>/
# (same seed -> same images everywhere). A dataset that fails to fetch is skipped with a
# warning, not fatal.
# Usage: tools/fetch_all_eval_data.sh [samples_per_class] [name ...]
#   names: defactify, mj-dalle-sd-nbp (the default set), openfake, phone-photos (real only),
#          unsplash-phone (real only), cf-eval, fairface (real only, with groups),
#          magicbrush (real photos and their AI edits)
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
    openfake)
      # OpenFake (ComplexDataLab, CC-BY-SA-4.0; proprietary-generator subsets non-commercial,
      # used here for evaluation only). Its test split holds the newest generators
      # (2025-26: GPT-Image-1.5/2, Midjourney 7, Flux.2, Nano Banana Pro, Seedream 5,
      # Z-Image, Recraft, frames from Veo 3 / Sora 2 ...) against ImageNet and DOCCI photos;
      # its train split holds older generators, so train-on-train / test-on-test measures
      # generalization to newer generators. Capped per generator for an even spread.
      python tools/fetch_eval_data.py \
        --dataset ComplexDataLab/OpenFake --config core --split test --strict-split \
        --out eval-data/openfake --per-class "$PER_CLASS" \
        --label-col label --ai-values fake --real-values real --generator-col model \
        --max-per-generator $(( (PER_CLASS + 9) / 10 )) --max-scan 40000
      ;;
    phone-photos)
      # Real only: smartphone photos from LAION-Mobile (arXiv 2609.11134; a community MDS
      # mirror of ~822k images, upstream copyright, evaluation only), kept only when their
      # EXIF make is a phone brand. Measures how often real phone photos read HIGH; AUC is
      # n/a. PER_CLASS photos. Drawn from the "test" shards only: training (Phase 3 train)
      # uses the "train" shards, so these photos are never trained on.
      python tools/fetch_mds_photos.py \
        --dataset sumathiselvan/LAION-Mobile-streaming \
        --out eval-data/phone-photos --count "$PER_CLASS" --split test
      ;;
    unsplash-phone)
      # Real only, evaluation only: phone photos listed in the Unsplash Lite dataset (camera
      # make is a phone brand), downloaded from Unsplash's image server. A second source,
      # independent of LAION-Mobile, which the detector is also trained on. Never trained
      # on. Unsplash Dataset terms: internal use only, no redistribution, so the photos
      # stay on the runner.
      python tools/fetch_unsplash_phone.py --out eval-data/unsplash-phone --count "$PER_CLASS"
      ;;
    cf-eval)
      # Community Forensics-Eval (OwensLab, CVPR 2025; CC-BY-NC-SA-4.0, evaluation only): the
      # authors' own test set, thousands of generators (GANs, diffusion, commercial tools)
      # against real photos. 206 GB in model order, so 16 files spread over the 413 are read
      # and each generator is capped, for a wide spread of generators.
      python tools/fetch_eval_data.py \
        --dataset OwensLab/CommunityForensics-Eval --data-files 'data/CompEval-*.parquet' --spread 16 \
        --out eval-data/cf-eval --per-class "$PER_CLASS" \
        --image-col image_data --label-col label --ai-values 1 --real-values 0 --generator-col model_name \
        --max-per-generator 12 --max-scan 40000
      ;;
    fairface)
      # Real only, for fairness: FairFace face photos (CC-BY-4.0) balanced over ethnicity x
      # gender, with each photo's group in groups.csv (tools/group_report.py). 4 x PER_CLASS
      # photos, so each of the 14 groups gets enough for an interval.
      python tools/fetch_fairface.py --out eval-data/fairface --count $(( PER_CLASS * 4 ))
      ;;
    magicbrush)
      # AI-edited real photos: MagicBrush (CC-BY-4.0, evaluation only), real MS-COCO photos
      # and the same photo after one DALL-E 2 edit. "ai" here means "edited by AI"; both
      # classes are saved as PNG at the same size (tools/fetch_magicbrush.py).
      python tools/fetch_magicbrush.py --out eval-data/magicbrush --count "$PER_CLASS"
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
