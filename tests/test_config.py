import json
import tempfile
import unittest
from pathlib import Path

from ehq.config import (
    load_experiment_config,
    load_models,
    model_registry_release_issues,
)
from ehq.constants import REFERENCE_PANEL_SIZE
from ehq.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[1]


class CustomRegistryTests(unittest.TestCase):
    def test_shipped_registry_is_provider_neutral_and_loadable(self):
        models = load_models(ROOT / "config" / "models.json")
        self.assertEqual(len(models), 3)
        self.assertEqual(
            {model.provider for model in models},
            {"openai", "huggingface", "openai_compatible"},
        )
        self.assertTrue(all(model.reasoning_mode == "disabled" for model in models))

    def test_custom_registry_is_explicitly_non_reference(self):
        models = load_models(ROOT / "config" / "models.json")
        issues = model_registry_release_issues(
            ROOT / "config" / "models.json", models
        )
        self.assertGreaterEqual(len(issues), len(models))
        self.assertTrue(
            any("operational_status is not verified" in issue for issue in issues)
        )

    def test_openai_compatible_requires_base_url(self):
        entry = {
            "name": "missing-url",
            "provider": "openai_compatible",
            "provider_model": "model",
            "reasoning_mode": "disabled",
        }
        with self.assertRaises(ConfigurationError) as caught:
            self._load_custom([entry])
        self.assertIn("base_url", str(caught.exception))

    def test_local_openai_compatible_route_can_omit_api_key(self):
        entry = {
            "name": "local",
            "provider": "openai_compatible",
            "provider_model": "model",
            "base_url": "http://localhost:8000/v1",
            "requires_api_key": False,
            "reasoning_mode": "disabled",
        }
        model = self._load_custom([entry])[0]
        self.assertFalse(model.requires_api_key)

    def test_default_configs_point_to_release_and_smoke_data(self):
        full = load_experiment_config(ROOT / "config" / "experiment.json")
        self.assertEqual(full.dataset_path.name, "EHQ-3000.json")
        self.assertTrue(full.dataset_requirements.require_pcq_temporal_novelty)
        smoke = load_experiment_config(ROOT / "config" / "smoke.json")
        self.assertEqual(smoke.dataset_path.name, "EHQ-20-smoke.json")

    @staticmethod
    def _load_custom(models):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "3.0",
                        "registry_mode": "custom",
                        "models": models,
                    }
                ),
                encoding="utf-8",
            )
            return load_models(path)


class ReferenceRegistryTests(unittest.TestCase):
    def _entry(self, name, **overrides):
        entry = {
            "name": name,
            "provider": "openai",
            "provider_model": name.lower(),
            "api_key_env": "OPENAI_API_KEY",
            "operational_status": "unverified",
            "reasoning_mode": "disabled",
            "cutoff_evidence_status": "unknown",
        }
        entry.update(overrides)
        return entry

    def _valid_models(self):
        models = []
        for index in range(8):
            models.append(
                self._entry(f"Old{index}", pair=f"p{index}", generation="old")
            )
            models.append(
                self._entry(f"New{index}", pair=f"p{index}", generation="new")
            )
        while len(models) < REFERENCE_PANEL_SIZE:
            models.append(self._entry(f"Loose{len(models)}"))
        return models

    def _load(self, models):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "3.0",
                        "registry_mode": "reference",
                        "panel_constraints": {
                            "pcq_cutoff_not_after": "2026-01-31"
                        },
                        "models": models,
                    }
                ),
                encoding="utf-8",
            )
            return load_models(path)

    def test_reference_registry_retains_pinned_panel_contract(self):
        self.assertEqual(len(self._load(self._valid_models())), REFERENCE_PANEL_SIZE)

    def test_reference_registry_size_is_pinned(self):
        with self.assertRaises(ConfigurationError):
            self._load(self._valid_models()[:-1])

    def test_duplicate_pair_slot_is_rejected(self):
        models = self._valid_models()
        models[-1] = self._entry("Intruder", pair="p0", generation="new")
        with self.assertRaises(ConfigurationError) as caught:
            self._load(models)
        self.assertIn("two 'new' members", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
