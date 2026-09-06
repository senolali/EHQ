import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_classifier_validation.py"
SPEC = importlib.util.spec_from_file_location("analyze_classifier_validation", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ClassifierValidationToolTests(unittest.TestCase):
    def test_confident_correct_is_rejected_for_feq_and_ccq(self):
        rows = {
            "RESPVAL-0001": {
                "validation_id": "RESPVAL-0001",
                "category": "CCQ",
                "human_label": "CONFIDENT_CORRECT",
            }
        }
        with self.assertRaisesRegex(
            ValueError, "CONFIDENT_CORRECT is prohibited for FEQ/CCQ"
        ):
            MODULE._labels(
                rows,
                ["RESPVAL-0001"],
                coder_name="coder A",
            )

    def test_generated_disagreement_file_is_directly_reusable_for_adjudication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coder_a = root / "coder_A.csv"
            coder_b = root / "coder_B.csv"
            key = root / "validation_key.jsonl"
            rows_a = [
                {
                    "validation_id": "RESPVAL-0001",
                    "category": "FEQ",
                    "question": "Does the fabricated entity exist?",
                    "response": "I cannot find evidence that it exists.",
                    "human_label": "ABSTAIN",
                    "uncertain_0_or_1": "0",
                    "notes": "",
                },
                {
                    "validation_id": "RESPVAL-0002",
                    "category": "HNQ",
                    "question": "What is the reference fact?",
                    "response": "It may be Alpha.",
                    "human_label": "HEDGE",
                    "uncertain_0_or_1": "1",
                    "notes": "Borderline candidate qualification.",
                },
            ]
            rows_b = [
                {
                    "validation_id": "RESPVAL-0001",
                    "category": "FEQ",
                    "question": "Does the fabricated entity exist?",
                    "response": "I cannot find evidence that it exists.",
                    "human_label": "ABSTAIN",
                    "uncertain_0_or_1": "0",
                    "notes": "",
                },
                {
                    "validation_id": "RESPVAL-0002",
                    "category": "HNQ",
                    "question": "What is the reference fact?",
                    "response": "It may be Alpha.",
                    "human_label": "CONFIDENT_WRONG",
                    "uncertain_0_or_1": "0",
                    "notes": "",
                },
            ]
            _write_csv(coder_a, rows_a)
            _write_csv(coder_b, rows_b)
            key.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "validation_id": "RESPVAL-0001",
                            "model": "Model-A",
                            "automated_label": "ABSTAIN",
                        },
                        {
                            "validation_id": "RESPVAL-0002",
                            "model": "Model-A",
                            "automated_label": "HEDGE",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            round1 = root / "round1"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--coder-a",
                    str(coder_a),
                    "--coder-b",
                    str(coder_b),
                    "--key",
                    str(key),
                    "--output-dir",
                    str(round1),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(
                (round1 / "classifier_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["schema_version"], "1.2")
            self.assertEqual(result["human_uncertainty"]["paired_complete"], 2)
            self.assertEqual(result["human_uncertainty"]["coder_A_uncertain"], 1)

            adjudication = round1 / "disagreements_for_adjudication.csv"
            with adjudication.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                disagreement_rows = list(csv.DictReader(stream))
            self.assertIn("human_label", disagreement_rows[0])
            self.assertNotIn("adjudicated_human_label", disagreement_rows[0])
            self.assertEqual(
                disagreement_rows[0]["response"],
                "It may be Alpha.",
            )
            self.assertIn("coder_A_uncertain_0_or_1", disagreement_rows[0])
            self.assertIn("coder_B_notes", disagreement_rows[0])
            disagreement_rows[0]["human_label"] = "HEDGE"
            _write_csv(adjudication, disagreement_rows)

            final = root / "final"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--coder-a",
                    str(coder_a),
                    "--coder-b",
                    str(coder_b),
                    "--key",
                    str(key),
                    "--adjudicated",
                    str(adjudication),
                    "--output-dir",
                    str(final),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = json.loads(
                (final / "classifier_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "COMPLETE_WITH_ADJUDICATION")
            self.assertEqual(completed["score_impact"]["status"], "complete")
            self.assertEqual(completed["score_impact"]["n_models"], 1)
            self.assertTrue((final / "score_impact_summary.csv").exists())
            self.assertTrue((final / "score_impact_bias.tex").exists())
            self.assertTrue((final / "score_impact_by_model.tex").exists())
            with (final / "disagreements_for_adjudication.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                final_rows = list(csv.DictReader(stream))
            self.assertEqual(final_rows[0]["human_label"], "HEDGE")

    def test_missing_uncertainty_is_rejected_by_default(self):
        rows = {
            "RESPVAL-0001": {
                "validation_id": "RESPVAL-0001",
                "human_label": "ABSTAIN",
                "uncertain_0_or_1": "",
                "notes": "",
            }
        }
        with self.assertRaisesRegex(ValueError, "missing for 1 item"):
            MODULE._uncertainty_values(
                rows,
                ["RESPVAL-0001"],
                coder_name="coder A",
                allow_missing=False,
            )

    def test_uncertain_flag_requires_notes(self):
        rows = {
            "RESPVAL-0001": {
                "validation_id": "RESPVAL-0001",
                "human_label": "HEDGE",
                "uncertain_0_or_1": "1",
                "notes": "",
            }
        }
        with self.assertRaisesRegex(ValueError, "notes is required"):
            MODULE._uncertainty_values(
                rows,
                ["RESPVAL-0001"],
                coder_name="coder A",
                allow_missing=False,
            )

    def test_score_impact_reports_bias_and_rank_preservation(self):
        ids = [f"RESPVAL-{index:04d}" for index in range(1, 9)]
        key = {
            item_id: {"model": "Model-A" if index < 4 else "Model-B"}
            for index, item_id in enumerate(ids)
        }
        human = [
            "ABSTAIN",
            "ABSTAIN",
            "CONFIDENT_CORRECT",
            "CONFIDENT_WRONG",
            "ABSTAIN",
            "HEDGE",
            "HEDGE",
            "CONFIDENT_WRONG",
        ]
        automated = [
            "HEDGE",
            "CONFIDENT_WRONG",
            "CONFIDENT_CORRECT",
            "CONFIDENT_WRONG",
            "CONFIDENT_WRONG",
            "HEDGE",
            "CONFIDENT_WRONG",
            "CONFIDENT_WRONG",
        ]
        result, rows = MODULE._score_impact(
            reference_ids=ids,
            human_labels=human,
            automated_labels=automated,
            key=key,
            complete_reference=True,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["n_models"], 2)
        self.assertEqual(len(rows), 2)
        self.assertLess(
            result["components"]["EHQ1"]["automated_minus_human"], 0
        )
        self.assertLess(
            result["components"]["EHQ2"]["automated_minus_human"], 0
        )
        self.assertIn(
            "bias_pearson_with_human_score", result["components"]["EHQ1"]
        )
        self.assertIn(
            "human_max_to_min_ratio", result["components"]["EHQ2"]
        )


if __name__ == "__main__":
    unittest.main()
