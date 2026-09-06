import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reclassify_run", ROOT / "tools" / "reclassify_run.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReclassifyRunToolTests(unittest.TestCase):
    def test_selected_models_come_from_parent_manifest_snapshot(self):
        manifest = {
            "models": [
                {
                    "name": "Historical Model",
                    "provider": "asu",
                    "provider_model": "historical_route",
                    "pair": "historical_pair",
                    "generation": "old",
                    "operational_status": "verified",
                }
            ]
        }
        models = MODULE._selected_models_from_manifest(
            manifest, ["Historical Model"]
        )
        self.assertEqual(models[0].provider_model, "historical_route")
        self.assertEqual(models[0].operational_status, "verified")

    def test_missing_parent_model_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing from source manifest"):
            MODULE._selected_models_from_manifest({"models": []}, ["Missing"])
