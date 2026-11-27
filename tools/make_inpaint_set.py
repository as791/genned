#!/usr/bin/env python3
"""Make training pairs of real photos and the same photos with part of them AI-inpainted.

The app misses AI edits of part of a real photo (internal-docs/MODEL.md, scorecard:
MagicBrush). This makes training examples for that case with open-weights inpainting
models only, on real photos that are not from MS-COCO (MagicBrush's test photos are
COCO, so none can leak into training):

  1. a real photo is center-cropped and resized to --size (both outputs share this size);
  2. a random mask covers 5-35% of it: a few rectangles, ellipses and strokes;
  3. the model fills the mask from a prompt drawn from a fixed object list;
  4. for --paste-share of the pairs (default 70%) the original pixels are pasted back
     outside a feathered mask, which is what editing tools produce and the hard case: only
     the edited area differs. The rest keep the model's whole output.

Writes <out>/real/<n>.png (the cropped photo), <out>/ai/<n>.png (its edit) and
<out>/pairs.csv. Both are PNG at the same size, so format and size can't tell them apart.
Images stay on the CI runner (uploaded only as a workflow artifact for the training job).

Usage (one shard of several, e.g. in a job matrix):
    python tools/make_inpaint_set.py --sources data/src/openfake/real data/src/phone/real \\
        --out out/inpaint --count 300 --shard 0 --shards 6 --model sd15
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter

# Open-weights inpainting models (diffusers repos).
MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "kandinsky22": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}
OBJECTS = [
    "a cat", "a dog", "a bird", "a car", "a bicycle", "a person", "a child", "a man", "a woman",
    "a vase of flowers", "a lamp", "a chair", "a cup of coffee", "a laptop", "a book", "a bag",
    "a tree", "a bush", "a rock", "a wooden sign", "a window", "a painting", "a clock", "a hat",
    "sunglasses", "a plate of food", "a bottle", "a balloon", "a kite", "a boat", "a bench",
    "a street lamp", "a cloud", "a mountain", "a fence", "a gift box", "a teddy bear", "a horse",
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def square(image: Image.Image, size: int) -> Image.Image:
    """Center crop to a square, then resize to size x size (RGB)."""
    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def mask_area(mask: Image.Image) -> float:
    histogram = mask.convert("L").histogram()
    return sum(histogram[128:]) / (mask.width * mask.height)


def random_mask(rng: random.Random, size: int, min_area: float = 0.05, max_area: float = 0.35) -> Image.Image:
    """White = area to inpaint. A few random shapes, redrawn until the area is in range."""
    for _ in range(200):
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        target = rng.uniform(min_area, max_area)
        for _ in range(rng.randint(1, 3)):
            w = int(size * rng.uniform(0.15, 0.6) * (target / 0.2) ** 0.5)
            h = int(size * rng.uniform(0.15, 0.6) * (target / 0.2) ** 0.5)
            x, y = rng.randint(0, max(0, size - w)), rng.randint(0, max(0, size - h))
            shape = rng.choice(("rect", "ellipse", "stroke"))
            if shape == "rect":
                draw.rectangle((x, y, x + w, y + h), fill=255)
            elif shape == "ellipse":
                draw.ellipse((x, y, x + w, y + h), fill=255)
            else:
                points = [(x + rng.randint(0, max(1, w)), y + rng.randint(0, max(1, h))) for _ in range(4)]
                draw.line(points, fill=255, width=max(8, size // 12))
        if min_area <= mask_area(mask) <= max_area:
            return mask
    raise RuntimeError("could not draw a mask in the area range")


def paste_back(original: Image.Image, generated: Image.Image, mask: Image.Image, feather: int = 6) -> Image.Image:
    """Original pixels outside the (feathered) mask, generated pixels inside it."""
    soft = mask.convert("L").filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), soft)


def make_pair(photo: Image.Image, inpaint: Callable[[Image.Image, Image.Image, str, int], Image.Image],
              rng: random.Random, size: int, paste_share: float) -> tuple[Image.Image, Image.Image, dict]:
    """(real, edited, metadata) for one source photo."""
    real = square(photo, size)
    mask = random_mask(rng, size)
    prompt = rng.choice(OBJECTS)
    generated = inpaint(real, mask, prompt, rng.randrange(2**31)).convert("RGB").resize((size, size), Image.LANCZOS)
    pasted = rng.random() < paste_share
    edited = paste_back(real, generated, mask) if pasted else generated
    return real, edited, {"prompt": prompt, "mask_area": round(mask_area(mask), 3), "pasted": int(pasted)}


def diffusers_inpaint(model: str, steps: int, threads: int | None) -> Callable:
    import torch
    from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler

    if threads:
        torch.set_num_threads(threads)
    # Weights only from safetensors files: a pickled checkpoint can run code when loaded, so a
    # model without safetensors fails here instead of falling back to pickle.
    if model.startswith("sd"):
        # The SD 1.5 mirror has safetensors only in its fp16 variant; it is upcast to fp32.
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODELS[model], torch_dtype=torch.float32, variant="fp16", use_safetensors=True, safety_checker=None)
        # Fewer steps for the same quality; Kandinsky's decoder keeps its own scheduler.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(MODELS[model], torch_dtype=torch.float32, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)

    def run(image: Image.Image, mask: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator().manual_seed(seed)
        return pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=steps,
                    height=image.height, width=image.width, generator=generator).images[0]

    return run


def sources(folders: list[Path], shard: int, shards: int, seed: int) -> list[Path]:
    files = sorted(p for f in folders if f.is_dir() for p in f.rglob("*") if p.suffix.lower() in EXTENSIONS)
    random.Random(seed).shuffle(files)
    return files[shard::shards]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, nargs="+", required=True, help="Folders of real photos")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300, help="Pairs to make in this shard")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--model", choices=sorted(MODELS), default="sd15")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--paste-share", type=float, default=0.7)
    parser.add_argument("--max-minutes", type=float, default=None, help="Stop cleanly after this long")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started = time.time()
    rng = random.Random(args.seed * 1000 + args.shard)
    inpaint = diffusers_inpaint(args.model, args.steps, args.threads)
    (args.out / "real").mkdir(parents=True, exist_ok=True)
    (args.out / "ai").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sources(args.sources, args.shard, args.shards, args.seed):
        if len(rows) >= args.count or (args.max_minutes and time.time() - started > args.max_minutes * 60):
            break
        try:
            with Image.open(path) as photo:
                real, edited, meta = make_pair(photo, inpaint, rng, args.size, args.paste_share)
        except OSError:
            continue
        name = f"{args.model}-{args.shard:02d}-{len(rows):05d}.png"
        real.save(args.out / "real" / name)
        edited.save(args.out / "ai" / name)
        rows.append({"file": name, "model": args.model, "source": path.parent.parent.name, **meta})
        if len(rows) % 10 == 0:
            per = (time.time() - started) / len(rows)
            print(f"{len(rows)} pairs, {per:.0f} s each", flush=True)
    with (args.out / f"pairs-{args.model}-{args.shard:02d}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "model", "source", "prompt", "mask_area", "pasted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Made {len(rows)} pairs with {args.model} in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
