import os
import tempfile
import unittest
from pathlib import Path

from ehq.env import load_local_env


class EnvironmentTests(unittest.TestCase):
    def test_loads_values_without_returning_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("EHQ_TEST_SECRET='secret-value'\n", encoding="utf-8")
            try:
                result = load_local_env(path, override=True)
                self.assertEqual(os.environ["EHQ_TEST_SECRET"], "secret-value")
                self.assertEqual(result, {"EHQ_TEST_SECRET": True})
                self.assertNotIn("secret-value", repr(result))
            finally:
                os.environ.pop("EHQ_TEST_SECRET", None)

    def test_rejects_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("NOT-AN-ENTRY\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_local_env(path)
