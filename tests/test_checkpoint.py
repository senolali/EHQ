from pathlib import Path
import tempfile
import unittest

from ehq.checkpoint import Checkpoint


class CheckpointTests(unittest.TestCase):
    def test_fingerprint_prevents_stale_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            checkpoint = Checkpoint(path, fingerprint="first")
            checkpoint.append({"model": "M", "question_id": "Q"})
            self.assertEqual(len(list(checkpoint.records())), 1)
            self.assertTrue(Checkpoint(path, fingerprint="first").contains("M", "Q"))
            with self.assertRaises(ValueError):
                Checkpoint(path, fingerprint="changed")
