"""tools/merge_inpaint_pairs.py: unique names, pairs kept together, duplicates skipped."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merge_inpaint_pairs as mp  # noqa: E402


def artifact(root: Path, name: str, pairs: dict[str, tuple[bytes, bytes]]) -> Path:
    folder = root / name
    (folder / "real").mkdir(parents=True)
    (folder / "ai").mkdir(parents=True)
    for file, (real, edit) in pairs.items():
        (folder / "real" / file).write_bytes(real)
        if edit is not None:
            (folder / "ai" / file).write_bytes(edit)
    return folder


class MergeTest(unittest.TestCase):
    def test_same_names_in_two_artifacts_are_both_kept_and_stay_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = artifact(root, "inpaint-sd15-3", {"sd15-00-00000.png": (b"real-a", b"edit-a")})
            b = artifact(root, "inpaint-sd15-4", {"sd15-00-00000.png": (b"real-b", b"edit-b")})
            kept = mp.merge([a, b], root / "out")
            self.assertEqual(kept, {"inpaint-sd15-3": 1, "inpaint-sd15-4": 1})
            out = root / "out"
            self.assertEqual((out / "real" / "inpaint-sd15-4-sd15-00-00000.png").read_bytes(), b"real-b")
            self.assertEqual((out / "ai" / "inpaint-sd15-4-sd15-00-00000.png").read_bytes(), b"edit-b")
            self.assertEqual(sorted(p.name for p in (out / "real").iterdir()),
                             sorted(p.name for p in (out / "ai").iterdir()))

    def test_duplicate_real_photos_and_half_pairs_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = artifact(root, "inpaint-k-0", {"x.png": (b"same", b"e1"), "half.png": (b"lonely", None)})
            b = artifact(root, "inpaint-k-1", {"x.png": (b"same", b"e2"), "y.png": (b"other", b"e3")})
            kept = mp.merge([a, b], root / "out")
            self.assertEqual(kept, {"inpaint-k-0": 1, "inpaint-k-1": 1})
            self.assertEqual(len(list((root / "out" / "ai").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
