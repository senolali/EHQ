from pathlib import Path
import tempfile
import unittest

from ehq.artifacts import verify_run_artifacts, write_artifact_catalog, write_json


class ArtifactTests(unittest.TestCase):
    def test_catalog_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "result.json"
            write_json(target, {"score": 1})
            write_artifact_catalog(root)
            self.assertTrue(verify_run_artifacts(root)["valid"])
            target.write_text("tampered", encoding="utf-8")
            report = verify_run_artifacts(root)
            self.assertFalse(report["valid"])
            self.assertEqual(report["mismatched"][0]["path"], "result.json")
