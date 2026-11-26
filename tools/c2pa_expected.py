#!/usr/bin/env python3
"""c2patool's verdict on each C2PA SDK sample file, for the app's conformance test.

Runs the reference tool (c2patool, built on the same C2PA Rust SDK) on every image and
video under the given fixture folders, copies the files into the test assets and writes
expected.json: {file name: verdict}. Verdicts:

  none      no Content Credentials
  valid     a manifest whose signature and hashes check out (trusted signer or not:
            c2patool runs without trust lists, so trust isn't compared)
  invalid   a manifest that fails validation (altered file, broken signature, ...)
  remote    the manifest is stored remotely (the app never fetches it)
  error     c2patool couldn't read the file (often a deliberately malformed sample)

The same rule as the app's C2paManifestInterpreter decides valid vs invalid: the
report's validation_state when there is one, else any status code other than an
untrusted signer means invalid.

Usage:
    python tools/c2pa_expected.py --fixtures work/src/sdk/tests/fixtures --out <assets dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".heic", ".heif", ".avif", ".mp4", ".mov"}
MAX_BYTES = 5 * 1024 * 1024
NO_MANIFEST = ("no claim found", "manifestnotfound", "jumbfnotfound", "no manifest", "no c2pa", "provenancemissing")


def verdict(returncode: int, stdout: str, stderr: str) -> str:
    """Map one c2patool run to a verdict."""
    text = (stdout + "\n" + stderr).lower()
    if "remote manifest" in text or "remotemanifest" in text:
        return "remote"
    if returncode == 0:
        try:
            report = json.loads(stdout)
        except ValueError:
            return "error"
        if not isinstance(report, dict) or not report.get("active_manifest"):
            return "none"
        state = report.get("validation_state")
        if state == "Invalid":
            return "invalid"
        if state in ("Valid", "Trusted"):
            return "valid"
        codes = [s.get("code") for s in report.get("validation_status") or [] if isinstance(s, dict)]
        return "invalid" if any(c != "signingCredential.untrusted" for c in codes) else "valid"
    if any(phrase in text for phrase in NO_MANIFEST):
        return "none"
    return "error"


def candidates(folders: list[Path]) -> list[Path]:
    files = [p for folder in folders if folder.is_dir() for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix.lower() in EXTENSIONS and p.stat().st_size <= MAX_BYTES]
    return files


def flat_name(path: Path, root: Path) -> str:
    """A unique flat asset name: the path under the fixture folder with '/' as '__'."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return "__".join(relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--c2patool", default="c2patool")
    parser.add_argument("--max-files", type=int, default=150)
    parser.add_argument("--max-total-mb", type=int, default=80, help="Keep the test APK small")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    expected: dict[str, str] = {}
    budget = args.max_total_mb * 1024 * 1024
    for path in candidates(args.fixtures)[: args.max_files]:
        budget -= path.stat().st_size
        if budget < 0:
            print(f"Stopped at {len(expected)} samples: the {args.max_total_mb} MB total cap")
            break
        root = next((f for f in args.fixtures if f in path.parents), path.parent)
        name = flat_name(path, root)
        try:
            run = subprocess.run([args.c2patool, str(path)], capture_output=True, text=True, timeout=60)
            expected[name] = verdict(run.returncode, run.stdout, run.stderr)
        except subprocess.TimeoutExpired:
            expected[name] = "error"
        shutil.copyfile(path, args.out / name)
    (args.out / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True))
    counts: dict[str, int] = {}
    for v in expected.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(expected)} samples; c2patool verdicts: {counts}")


if __name__ == "__main__":
    main()
