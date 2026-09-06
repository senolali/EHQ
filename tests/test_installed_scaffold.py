import argparse
import tempfile
import unittest
from pathlib import Path

from ehq.cli import _init_project
from ehq.config import load_experiment_config, load_models


class InstalledScaffoldTests(unittest.TestCase):
    def test_init_creates_complete_minimal_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "study"
            result = _init_project(
                argparse.Namespace(directory=str(destination), force=False)
            )
            self.assertEqual(result, 0)
            self.assertTrue((destination / ".env.example").is_file())
            self.assertFalse((destination / ".env").exists())
            self.assertTrue(
                (destination / "data/examples/EHQ-20-smoke.json").is_file()
            )
            self.assertTrue(
                (destination / "data/releases/EHQ-3000.json").is_file()
            )
            smoke = load_experiment_config(destination / "config/smoke.json")
            self.assertEqual(smoke.dataset_path.name, "EHQ-20-smoke.json")
            self.assertEqual(len(load_models(smoke.models_path)), 3)

    def test_init_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "study"
            _init_project(argparse.Namespace(directory=str(destination), force=False))
            with self.assertRaises(SystemExit):
                _init_project(
                    argparse.Namespace(directory=str(destination), force=False)
                )


if __name__ == "__main__":
    unittest.main()
