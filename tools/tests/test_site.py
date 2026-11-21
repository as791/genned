"""tools/build_site.py: the website build fills every placeholder from the release and
refuses to publish a page with missing downloads or app screens."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_site  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

RELEASE = {
    "tagName": "v9.8.7",
    "url": "https://github.com/as791/genned/releases/tag/v9.8.7",
    "publishedAt": "2026-11-16T07:00:00Z",
    "assets": [
        {"name": "genned-9.8.7-arm64-v8a.apk", "size": 190_512_345,
         "url": "https://github.com/as791/genned/releases/download/v9.8.7/genned-9.8.7-arm64-v8a.apk"},
        {"name": "genned-9.8.7.apk", "size": 263_200_000,
         "url": "https://github.com/as791/genned/releases/download/v9.8.7/genned-9.8.7.apk"},
        {"name": "SHA256SUMS.txt", "size": 500, "url": "https://example.invalid/SHA256SUMS.txt"},
    ],
}


def screens(folder: Path) -> Path:
    from PIL import Image

    folder.mkdir()
    for name in build_site.SCREENS:
        Image.new("RGB", (1080, 2340), (240, 240, 250)).save(folder / f"{name}.png")
    return folder


class ReleaseValuesTest(unittest.TestCase):
    def test_values_come_from_the_release(self):
        v = build_site.release_values(RELEASE)
        self.assertEqual(v["VERSION"], "9.8.7")
        self.assertTrue(v["APK_URL"].endswith("genned-9.8.7-arm64-v8a.apk"))
        self.assertEqual(v["APK_MB"], "191")
        self.assertEqual(v["UNIVERSAL_MB"], "263")
        self.assertEqual(v["RELEASE_DATE"], "16 November 2026")

    def test_a_release_without_the_arm64_apk_stops_the_build(self):
        broken = dict(RELEASE, assets=RELEASE["assets"][1:])
        with self.assertRaises(SystemExit):
            build_site.release_values(broken)

    def test_an_unknown_placeholder_stops_the_build(self):
        with self.assertRaises(SystemExit):
            build_site.fill("{{VERSION}} {{NOPE}}", {"VERSION": "1"})


class BuildTest(unittest.TestCase):
    def test_the_real_site_builds_with_nothing_left_unfilled(self):
        try:
            import markdown  # noqa: F401
        except ImportError:
            self.skipTest("markdown not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "_site"
            build_site.build(REPO / "site", screens(tmp / "screens"), RELEASE,
                             (REPO / "PRIVACY.md").read_text(), out)
            index = (out / "index.html").read_text()
            privacy = (out / "privacy.html").read_text()
            self.assertNotIn("{{", index + privacy)
            self.assertIn("genned-9.8.7-arm64-v8a.apk", index)
            self.assertIn("Version 9.8.7", index)
            self.assertIn("<h1>Genned privacy policy</h1>", privacy)
            # Every image the pages use exists in the build.
            import re
            for src in re.findall(r'src="(assets/[^"]+)"', index):
                self.assertTrue((out / src).exists(), src)
            self.assertTrue((out / ".nojekyll").exists())

    def test_a_missing_screen_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = screens(Path(tmp) / "screens")
            (folder / "06_result_verified.png").unlink()
            with self.assertRaises(SystemExit):
                build_site.copy_screens(folder, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
