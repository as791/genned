#!/usr/bin/env python3
"""Builds the Genned website (site/) into a folder GitHub Pages can serve.

usage: build_site.py --site site --screens <dir> --release <release.json> --out _site

- site/*.html are templates: {{VERSION}}, {{RELEASE_URL}}, {{APK_URL}}, {{APK_MB}},
  {{UNIVERSAL_URL}}, {{UNIVERSAL_MB}}, {{RELEASE_DATE}} and {{YEAR}} are filled from the
  latest GitHub release (`gh release view --json tagName,url,publishedAt,assets`).
- privacy.html gets PRIVACY.md rendered into {{PRIVACY}}, so the site never drifts from
  the policy in the repository.
- The app screens come from the "screens" artifact of the latest green Android CI run
  (ScreenshotTest), resized for the web. A missing screen fails the build rather than
  publishing a broken page.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

SCREENS = ("01_home_first_run", "02_home_with_history", "03_result_likely_ai",
           "03b_result_uncertain_high_score", "04_result_uncertain_video", "05_result_likely_not_ai",
           "06_result_verified", "07_result_likely_ai_dark", "08_history", "09_guide", "10_overlay_setup")
SCREEN_WIDTH = 720


def release_values(release: dict) -> dict[str, str]:
    """Template values from `gh release view --json tagName,url,publishedAt,assets`."""
    version = release["tagName"].lstrip("v")
    assets = {a["name"]: a for a in release.get("assets", [])}

    def asset(name: str) -> dict:
        if name not in assets:
            sys.exit(f"Release {release['tagName']} has no asset {name}; found: {', '.join(sorted(assets))}")
        return assets[name]

    arm64 = asset(f"genned-{version}-arm64-v8a.apk")
    universal = asset(f"genned-{version}.apk")
    published = dt.datetime.fromisoformat(release["publishedAt"].replace("Z", "+00:00"))
    return {
        "VERSION": version,
        "RELEASE_URL": release["url"],
        "APK_URL": arm64["url"],
        "APK_MB": str(round(arm64["size"] / 1e6)),
        "UNIVERSAL_URL": universal["url"],
        "UNIVERSAL_MB": str(round(universal["size"] / 1e6)),
        "RELEASE_DATE": f"{published.day} {published:%B %Y}",
        "YEAR": str(published.year),
    }


def fill(template: str, values: dict[str, str]) -> str:
    out = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), template)
    left = sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", out)))
    if left:
        sys.exit(f"Unfilled placeholders: {', '.join(left)}")
    return out


def render_privacy(markdown_text: str) -> str:
    import markdown  # only needed here; pip install markdown

    html = markdown.markdown(markdown_text, extensions=["tables"])
    # Links inside the policy that point back into the repository open on GitHub.
    return re.sub(r'href="(?!https?://|#|mailto:)([^"]+)"',
                  lambda m: f'href="https://github.com/as791/genned/blob/main/{m.group(1)}"', html)


def copy_screens(src: Path, dest: Path) -> None:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    for name in SCREENS:
        path = src / f"{name}.png"
        if not path.exists():
            sys.exit(f"Missing app screen {path} (the 'screens' artifact of Android CI)")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width > SCREEN_WIDTH:
                image = image.resize((SCREEN_WIDTH, round(image.height * SCREEN_WIDTH / image.width)), Image.LANCZOS)
            image.save(dest / f"{name}.jpg", quality=86, optimize=True, progressive=True)


def build(site: Path, screens: Path, release: dict, privacy_md: str, out: Path) -> None:
    values = release_values(release)
    values["PRIVACY"] = render_privacy(privacy_md)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(site, out, ignore=shutil.ignore_patterns("*.html"))
    for page in site.glob("*.html"):
        (out / page.name).write_text(fill(page.read_text(), values))
    copy_screens(screens, out / "assets" / "screens")
    (out / ".nojekyll").write_text("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=Path, default=Path("site"))
    ap.add_argument("--screens", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True, help="JSON from gh release view")
    ap.add_argument("--privacy", type=Path, default=Path("PRIVACY.md"))
    ap.add_argument("--out", type=Path, default=Path("_site"))
    args = ap.parse_args()
    build(args.site, args.screens, json.loads(args.release.read_text()), args.privacy.read_text(), args.out)
    print(f"Built {args.out} ({sum(1 for _ in args.out.rglob('*') if _.is_file())} files)")


if __name__ == "__main__":
    main()
