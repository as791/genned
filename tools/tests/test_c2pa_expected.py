"""tools/c2pa_expected.py: c2patool output -> conformance verdict."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import c2pa_expected as ce  # noqa: E402


def report(**extra):
    return json.dumps({"active_manifest": "urn:uuid:1", "manifests": {"urn:uuid:1": {}}, **extra})


class VerdictTest(unittest.TestCase):
    def test_validation_state_decides_when_present(self):
        self.assertEqual(ce.verdict(0, report(validation_state="Valid"), ""), "valid")
        self.assertEqual(ce.verdict(0, report(validation_state="Trusted"), ""), "valid")
        self.assertEqual(ce.verdict(0, report(validation_state="Invalid"), ""), "invalid")

    def test_older_reports_use_the_status_codes_like_the_app(self):
        untrusted = [{"code": "signingCredential.untrusted"}]
        broken = untrusted + [{"code": "assertion.dataHash.mismatch"}]
        self.assertEqual(ce.verdict(0, report(validation_status=untrusted), ""), "valid")
        self.assertEqual(ce.verdict(0, report(validation_status=broken), ""), "invalid")
        self.assertEqual(ce.verdict(0, report(), ""), "valid")

    def test_no_manifest_remote_and_errors(self):
        self.assertEqual(ce.verdict(1, "", "Error: No claim found"), "none")
        self.assertEqual(ce.verdict(1, "", "Error: JumbfNotFound"), "none")
        self.assertEqual(ce.verdict(0, "{}", ""), "none")
        self.assertEqual(ce.verdict(1, "", "Error: Remote manifest at https://example.com"), "remote")
        self.assertEqual(ce.verdict(1, "", "Error: UnsupportedType"), "error")
        self.assertEqual(ce.verdict(0, "not json", ""), "error")


class FilesTest(unittest.TestCase):
    def test_candidates_and_flat_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "x.jpg").write_bytes(b"1")
            (root / "x.jpg").write_bytes(b"2")
            (root / "notes.txt").write_bytes(b"3")
            (root / "big.png").write_bytes(b"0" * (ce.MAX_BYTES + 1))
            found = ce.candidates([root])
            self.assertEqual([p.name for p in found], ["x.jpg", "x.jpg"])
            names = {ce.flat_name(p, root) for p in found}
            self.assertEqual(names, {"a__x.jpg", "x.jpg"})


if __name__ == "__main__":
    unittest.main()
