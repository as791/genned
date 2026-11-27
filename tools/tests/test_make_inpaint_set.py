"""tools/make_inpaint_set.py with a stub inpainting model (no downloads)."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_inpaint_set as mi  # noqa: E402


def stub_inpaint(image, mask, prompt, seed):
    """Paints the whole picture flat magenta, so changed pixels are easy to find."""
    return Image.new("RGB", image.size, (255, 0, 255))


class MaskTest(unittest.TestCase):
    def test_area_stays_in_range(self):
        rng = random.Random(0)
        for _ in range(50):
            area = mi.mask_area(mi.random_mask(rng, 256))
            self.assertGreaterEqual(area, 0.05)
            self.assertLessEqual(area, 0.35)


class PairTest(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new("RGB", (640, 480), (20, 120, 40))

    def test_real_and_edit_share_size(self):
        real, edited, meta = mi.make_pair(self.photo, stub_inpaint, random.Random(1), 256, 0.7)
        self.assertEqual(real.size, (256, 256))
        self.assertEqual(edited.size, (256, 256))
        self.assertIn(meta["prompt"], mi.OBJECTS)

    def test_paste_back_changes_only_the_masked_area(self):
        real, edited, meta = mi.make_pair(self.photo, stub_inpaint, random.Random(2), 256, paste_share=1.0)
        self.assertEqual(meta["pasted"], 1)
        changed = sum(a != b for a, b in zip(real.getdata(), edited.getdata())) / (256 * 256)
        # The changed share is the mask area plus a feathered rim, never the whole picture.
        self.assertGreater(changed, meta["mask_area"] * 0.9)
        self.assertLess(changed, min(1.0, meta["mask_area"] + 0.25))

    def test_without_paste_back_the_whole_output_is_kept(self):
        real, edited, meta = mi.make_pair(self.photo, stub_inpaint, random.Random(3), 256, paste_share=0.0)
        self.assertEqual(meta["pasted"], 0)
        self.assertEqual(edited.getpixel((0, 0)), (255, 0, 255))

    def test_square_center_crop(self):
        tall = Image.new("RGB", (100, 300))
        tall.paste((255, 255, 255), (0, 100, 100, 200))  # white middle third
        out = mi.square(tall, 50)
        self.assertEqual(out.size, (50, 50))
        self.assertEqual(out.getpixel((25, 25)), (255, 255, 255))


class SourcesTest(unittest.TestCase):
    def test_shards_split_the_files_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, "real")
            folder.mkdir()
            for i in range(10):
                (folder / f"{i}.jpg").write_bytes(b"x")
            (folder / "notes.txt").write_bytes(b"x")
            shards = [set(mi.sources([folder], s, 3, seed=1)) for s in range(3)]
            self.assertEqual(sum(len(s) for s in shards), 10)
            self.assertFalse(shards[0] & shards[1] or shards[1] & shards[2] or shards[0] & shards[2])


class PairNameTest(unittest.TestCase):
    def test_tag_keeps_jobs_apart(self):
        self.assertEqual(mi.pair_name("sd15", "07", 3), "sd15-07-00003.png")
        self.assertNotEqual(mi.pair_name("sd15", "07", 3), mi.pair_name("sd15", "08", 3))


if __name__ == "__main__":
    unittest.main()
