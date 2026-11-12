#!/usr/bin/env python3
"""Fetches training frames from a WebDataset of real and AI-generated video clips.

Written for GenVideo (sshaiton/GenVideo-50k-shard): tar shards in which every clip is a
video file plus a JSON label ({"label": "real" | "fake", "generator": ...}). Each tar is
read as a stream, so nothing but the frames is kept on disk. Frames are taken the way the
app samples a video (tools/eval_video.py sample_frame_jpegs: five positions, <= 1024 px,
JPEG q90) and written in the layout tools/adv_finetune.py reads:

    <out>/ai/<generator>/<clip>_<position>.jpg
    <out>/real/<clip>_<position>.jpg

Shards are split by number like the phone photos (number % 5: 0 -> test, 1 -> val, anything
else -> train), so training and model selection never share a clip. Clips from excluded
generators (proprietary ones; Pika by default) and from any generator not on the list of
open-weights ones are skipped, and the run fails if an excluded folder still appears. The clips are used only on the machine that runs this.

Usage:
    python tools/fetch_video_frames.py --out data/train/video --frames-per-class 6000 --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

DATASET = "sshaiton/GenVideo-50k-shard"
EXCLUDED_GENERATORS = "pika"  # proprietary: its clips are never trained on
# GenVideo's open-weights generators, as folder names (see folder_name). Anything else,
# for example a generator a later shard adds, is skipped until it has been checked.
OPEN_GENERATORS = "svd|seine|opensora|st2v|videocrafter|latte|cog|dynamicrafter|dynamiccrafter|zeroscope"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv")
MAX_CLIP_BYTES = 60_000_000
SPLIT_MOD = 5  # shard number % SPLIT_MOD: 0 -> test, 1 -> val, anything else -> train
SPLITS = ("all", "train", "val", "test")


def shard_number(name: str) -> int:
    """The last number in a shard's file name (dataset_shard_000012.tar -> 12)."""
    digits = re.findall(r"\d+", os.path.basename(name))
    if not digits:
        raise ValueError(f"no shard number in {name!r}")
    return int(digits[-1])


def in_split(name: str, split: str) -> bool:
    if split == "all":
        return True
    return {0: "test", 1: "val"}.get(shard_number(name) % SPLIT_MOD, "train") == split


def folder_name(generator: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", generator.strip().lower()).strip("_") or "unknown"


def wanted(record: dict, exclude_regex: str, allow_regex: str = "") -> tuple[str, str] | None:
    """("real", "real") or ("ai", generator folder) for a clip to keep; None for one to skip.
    With `allow_regex`, only generators whose whole folder name matches it are kept."""
    label = str(record.get("label", "")).strip().lower()
    generator = str(record.get("generator", "")).strip()
    if label == "real":
        return "real", "real"
    if label == "fake" and generator and generator.lower() != "real":
        if exclude_regex and re.search(exclude_regex, generator, re.IGNORECASE):
            return None
        folder = folder_name(generator)
        if allow_regex and not re.fullmatch(allow_regex, folder):
            return None
        return "ai", folder
    return None


def iter_clips(fileobj) -> Iterator[tuple[str, dict, bytes]]:
    """(name, label record, video bytes) for each clip of a tar read as a stream. A clip's
    label and video share a name and sit next to each other, in either order."""
    pending: dict[str, dict] = {}
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = os.path.splitext(member.name)
            ext = ext.lower()
            if ext != ".json" and (ext not in VIDEO_EXTENSIONS or member.size > MAX_CLIP_BYTES):
                continue
            data = tar.extractfile(member).read()
            slot = pending.setdefault(key, {})
            if ext == ".json":
                try:
                    record = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    record = None
                if not isinstance(record, dict):
                    pending.pop(key, None)
                    continue
                slot["record"] = record
            else:
                slot["video"] = data
            if "record" in slot and "video" in slot:
                del pending[key]
                yield key, slot["record"], slot["video"]
            while len(pending) > 4:  # members whose partner never came
                pending.pop(next(iter(pending)))


def app_frames(video: bytes) -> list[bytes]:
    """The app's sampled frames of a clip, as JPEG bytes (empty if it can't be decoded)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_video import sample_frame_jpegs

    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        handle.write(video)
        handle.flush()
        try:
            return sample_frame_jpegs(Path(handle.name))
        except Exception:  # noqa: BLE001 - an undecodable clip is simply skipped
            return []


def fetch(shards: Iterable[tuple[str, Callable]], out: Path, frames_per_class: int, max_per_generator: int,
          frames_per_clip: int, exclude_regex: str, seed: int, sampler: Callable[[bytes], list[bytes]] = app_frames,
          allow_regex: str = "") -> tuple[Counter, Counter]:
    """Reads shards (name, open function) until both classes hold `frames_per_class` frames.
    Returns (frames saved per class, frames saved per generator folder)."""
    per_class: Counter = Counter()
    per_generator: Counter = Counter()
    skipped: Counter = Counter()
    for name, opener in shards:
        if min(per_class["ai"], per_class["real"]) >= frames_per_class:
            break
        try:
            with opener() as handle:
                for key, record, video in iter_clips(handle):
                    kind = wanted(record, exclude_regex, allow_regex)
                    if kind is None:
                        skipped[f"not kept: {str(record.get('generator', '?'))[:40]}"] += 1
                        continue
                    label, generator = kind
                    if per_class[label] >= frames_per_class or (
                            label == "ai" and per_generator[generator] >= max_per_generator):
                        skipped["class or generator already full"] += 1
                        continue
                    frames = sampler(video)
                    if not frames:
                        skipped["not decodable"] += 1
                        continue
                    clip = f"{shard_number(name):06d}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
                    rng = random.Random(f"{seed}:{clip}")
                    positions = sorted(rng.sample(range(len(frames)), min(frames_per_clip, len(frames))))
                    folder = out / "real" if label == "real" else out / "ai" / generator
                    folder.mkdir(parents=True, exist_ok=True)
                    for position in positions:
                        (folder / f"{clip}_{position}.jpg").write_bytes(frames[position])
                    per_class[label] += len(positions)
                    per_generator[generator] += len(positions)
        except (OSError, tarfile.TarError, EOFError) as e:
            # A shard that breaks off keeps the frames it gave; the next one makes up the rest.
            print(f"  {name}: stopped early ({type(e).__name__}: {e})", flush=True)
        print(f"  {name}: {per_class['ai']} AI + {per_class['real']} real frames so far", flush=True)
    print(f"Saved {per_class['ai']} AI + {per_class['real']} real frames; per folder: "
          f"{dict(per_generator.most_common())}")
    print(f"Skipped clips: {dict(skipped)}")
    return per_class, per_generator


def excluded_folders(out: Path, exclude_regex: str) -> list[str]:
    """Generator folders that should not exist (an excluded generator that got through)."""
    if not exclude_regex or not (out / "ai").is_dir():
        return []
    return sorted(d.name for d in (out / "ai").iterdir() if d.is_dir() and re.search(exclude_regex, d.name, re.IGNORECASE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-class", type=int, default=6000)
    parser.add_argument("--frames-per-clip", type=int, default=3, help="Of the app's five sampled frames")
    parser.add_argument("--max-per-generator", type=int, default=0,
                        help="Frames per AI generator (0 = a share of --frames-per-class with some slack)")
    parser.add_argument("--exclude-generators", default=EXCLUDED_GENERATORS, help="Regex of generators to skip")
    parser.add_argument("--generators", default=OPEN_GENERATORS,
                        help="Regex of the generator folder names to keep (whole name; empty = all not excluded)")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--max-shards", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    tars = sorted(f for f in HfApi().list_repo_files(args.dataset, repo_type="dataset") if f.endswith(".tar"))
    chosen = [t for t in tars if in_split(t, args.split)]
    random.Random(args.seed).shuffle(chosen)
    chosen = chosen[: args.max_shards]
    print(f"{args.dataset}: {len(tars)} shards, {len(chosen)} to read for split {args.split!r}")
    # GenVideo has nine open-weights generators; a quarter of slack so a rare one doesn't stall the run.
    cap = args.max_per_generator or max(1, round(args.frames_per_class / 9 * 1.25))
    fs = HfFileSystem()
    shards = ((name, lambda name=name: fs.open(f"datasets/{args.dataset}/{name}", "rb")) for name in chosen)
    per_class, _ = fetch(shards, args.out, args.frames_per_class, cap, args.frames_per_clip,
                         args.exclude_generators, args.seed, allow_regex=args.generators)
    bad = excluded_folders(args.out, args.exclude_generators)
    if bad:
        sys.exit(f"Excluded generators found in the output: {bad}")
    if min(per_class["ai"], per_class["real"]) < min(args.frames_per_class, 50):
        sys.exit("Too few frames could be fetched.")


if __name__ == "__main__":
    main()
