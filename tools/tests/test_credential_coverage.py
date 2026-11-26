"""tools/credential_coverage.py: finding a C2PA manifest store by its JUMBF label."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import credential_coverage as cc  # noqa: E402

C2PA_UUID = bytes.fromhex("6332706100110010800000AA00389B71")  # 'c2pa' + ISO suffix


def jumbf(label: bytes) -> bytes:
    """A JUMBF superbox with a description box carrying `label`."""
    jumd_payload = C2PA_UUID + b"\x03" + label + b"\x00"
    jumd = struct.pack(">I", 8 + len(jumd_payload)) + b"jumd" + jumd_payload
    return struct.pack(">I", 8 + len(jumd)) + b"jumb" + jumd


class ManifestStoreTest(unittest.TestCase):
    def test_finds_the_c2pa_store(self):
        self.assertTrue(cc.has_manifest_store(b"\xff\xd8 header " + jumbf(b"c2pa") + b" rest"))

    def test_other_jumbf_labels_and_plain_files_are_not_counted(self):
        self.assertFalse(cc.has_manifest_store(b"\xff\xd8" + jumbf(b"c2pa.assertions")))
        self.assertFalse(cc.has_manifest_store(b"\xff\xd8 the word c2pa in text, no jumbf"))
        self.assertTrue(cc.has_manifest_store(jumbf(b"other") + jumbf(b"c2pa")))

    def test_counts_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls, contents in (("real", [b"plain", b"plain"]), ("ai", [jumbf(b"c2pa"), b"plain"])):
                (root / "set" / cls).mkdir(parents=True)
                for i, data in enumerate(contents):
                    (root / "set" / cls / f"{i}.jpg").write_bytes(data)
            counts = cc.count(root / "set")
            self.assertEqual(counts["real"], {"files": 2, "with_credentials": 0})
            self.assertEqual(counts["ai"], {"files": 2, "with_credentials": 1})
            (root / "coverage-set.json").write_text(json.dumps({"name": "set", "counts": counts}))
            table = cc.report(root)
            self.assertIn("| set | 2 | 0 (0.0%) | 2 | 1 (50.0%) |", table)


if __name__ == "__main__":
    unittest.main()
