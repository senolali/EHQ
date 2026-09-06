from pathlib import Path
import tempfile
import unittest

from ehq.constants import FRAMEWORK_VERSION
from ehq.provenance import build_manifest
from ehq.types import ModelSpec


class ProvenanceTests(unittest.TestCase):
    def test_manifest_embeds_config_snapshot_and_pair_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            dataset = root / "dataset.json"
            models = root / "models.json"
            for path in (config, dataset, models):
                path.write_text("{}", encoding="utf-8")
            snapshot = {
                "inference": {"temperature": 0.0, "max_tokens": 512},
                "weights": {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
            }
            model = ModelSpec(
                name="Test Old",
                provider="asu",
                provider_model="test_old",
                pair="test_pair",
                generation="old",
            )
            manifest = build_manifest(
                project_root=root,
                experiment_name="test",
                config_path=config,
                dataset_path=dataset,
                models_path=models,
                models=[model],
                configuration=snapshot,
            )
            self.assertEqual(manifest["config"]["snapshot"], snapshot)
            self.assertEqual(manifest["dependencies"]["ehq"], FRAMEWORK_VERSION)
            self.assertEqual(manifest["framework_version"], FRAMEWORK_VERSION)
            self.assertEqual(manifest["models"][0]["pair"], "test_pair")
            self.assertEqual(manifest["models"][0]["generation"], "old")
