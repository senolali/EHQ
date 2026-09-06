from pathlib import Path
import unittest

from ehq import publication as MODULE
from ehq.evaluation.scoring import compute_ehq_scores


ROOT = Path(__file__).resolve().parents[1]


class BuildProvisionalPilotReportToolTests(unittest.TestCase):
    def test_stratified_bootstrap_is_deterministic_and_stays_within_strata(self):
        strata = ["A", "A", "B", "B"]
        first = MODULE._stratified_bootstrap_indices(
            strata, n_resamples=20, seed=7
        )
        second = MODULE._stratified_bootstrap_indices(
            strata, n_resamples=20, seed=7
        )
        self.assertTrue((first == second).all())
        self.assertTrue(((first[:, :2] == 0) | (first[:, :2] == 1)).all())
        self.assertTrue(((first[:, 2:] == 2) | (first[:, 2:] == 3)).all())

    def test_percentile_interval_contains_center(self):
        low, high = MODULE._percentile_interval([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.3)


WEIGHTS = {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}


def _record(model, index, label, confidence, *, technical=False, no_confidence=False):
    row = {
        "model": model,
        "question_id": f"FEQ-ORG-{index:03d}",
        "category": "FEQ",
        "subcategory": f"FEQ-S{index % 4}",
        "k": 0,
        "answer_response": {"ok": not technical, "text": None if technical else "a"},
        "confidence_response": {"ok": True, "text": "50"},
        "parsed_confidence": None if (technical or no_confidence) else confidence,
        "confidence_parse_reason": "multiple_numeric_tokens" if no_confidence else None,
        "confidence_terminal": bool(no_confidence),
        "classification": None if technical else {
            "label": label,
            "is_correct": label == "CONFIDENT_CORRECT",
        },
        "valid_for_ehq12": not technical,
        "valid_for_ehq3": not (technical or no_confidence),
        "exclusion_reason": "response_format" if technical else None,
    }
    if technical:
        row["answer_response"]["error_type"] = "response_format"
    return row


def _panel():
    """Two models over the same 20 items; the second has incomplete coverage."""

    labels = ["CONFIDENT_CORRECT", "CONFIDENT_WRONG", "ABSTAIN", "HEDGE"]
    rows = []
    for index in range(20):
        label = labels[index % 4]
        confidence = 0.9 if label.startswith("CONFIDENT") else 0.1
        rows.append(_record("Complete", index, label, confidence))
        rows.append(
            _record(
                "Partial",
                index,
                label,
                confidence,
                technical=(index == 5),
                no_confidence=(index == 9),
            )
        )
    return rows


class IncompleteCoverageTests(unittest.TestCase):
    def test_coverage_tables_document_every_unscored_record(self):
        exclusions, coverage = MODULE._coverage_tables(
            _panel(), ["Complete", "Partial"]
        )
        self.assertEqual(len(exclusions), 2)
        by_reason = {row["reason"]: row for row in exclusions}
        self.assertEqual(
            by_reason["technical_failure"]["excluded_from"], "EHQ1, EHQ2, EHQ3"
        )
        self.assertEqual(by_reason["technical_failure"]["detail"], "response_format")
        self.assertEqual(
            by_reason["terminal_missing_confidence"]["excluded_from"], "EHQ3"
        )
        self.assertEqual(
            by_reason["terminal_missing_confidence"]["detail"],
            "multiple_numeric_tokens",
        )
        self.assertTrue(all(row["model"] == "Partial" for row in exclusions))

        complete, partial = coverage
        self.assertEqual(complete["n_items"], 20)
        self.assertEqual(complete["n_excluded_ehq12"], 0)
        self.assertEqual(complete["n_excluded_ehq3_only"], 0)
        self.assertEqual(partial["n_scored_ehq12"], 19)
        self.assertEqual(partial["n_excluded_ehq12"], 1)
        self.assertEqual(partial["n_excluded_ehq3_only"], 1)

    def test_bootstrap_denominators_match_the_point_estimate(self):
        rows = _panel()
        boot, _ = MODULE._bootstrap_scores(
            rows,
            ["Complete", "Partial"],
            weights=WEIGHTS,
            n_bins=10,
            n_resamples=4000,
            seed=42,
        )
        for model in ("Complete", "Partial"):
            point = compute_ehq_scores(
                [row for row in rows if row["model"] == model],
                weights=WEIGHTS,
                n_bins=10,
            )
            for field in ("EHQ1", "EHQ2"):
                self.assertAlmostEqual(
                    float(boot[model][field].mean()),
                    float(point[field]),
                    delta=0.03,
                    msg=f"{model} {field} bootstrap mean drifted from point estimate",
                )
            self.assertTrue((boot[model]["EHQ3"] <= 1.0).all())

    def test_incomplete_coverage_shifts_only_the_affected_model(self):
        rows = _panel()
        boot, _ = MODULE._bootstrap_scores(
            rows,
            ["Complete", "Partial"],
            weights=WEIGHTS,
            n_bins=10,
            n_resamples=500,
            seed=42,
        )
        # The complete model is unaffected by its neighbour's missing records.
        point = compute_ehq_scores(
            [row for row in rows if row["model"] == "Complete"],
            weights=WEIGHTS,
            n_bins=10,
        )
        self.assertAlmostEqual(
            float(boot["Complete"]["EHQ1"].mean()), float(point["EHQ1"]), delta=0.03
        )


class ReportIdentityTests(unittest.TestCase):
    def test_undefined_scores_render_as_a_dash_not_a_crash(self):
        self.assertEqual(MODULE._fmt(None), "-")
        self.assertEqual(MODULE._fmt(""), "-")
        self.assertEqual(MODULE._fmt(0.5), "0.5000")

    def test_optional_float_reads_an_empty_metric_cell(self):
        self.assertIsNone(MODULE._optional_float(""))
        self.assertIsNone(MODULE._optional_float(None))
        self.assertEqual(MODULE._optional_float("0.25"), 0.25)

    def test_every_panel_model_has_a_display_label(self):
        panel = (
            "Claude-5-Sonnet", "Claude-4.5-Haiku", "Claude-4-Sonnet",
            "Claude-3-Haiku", "DeepSeek-V4", "Gemma-4-31B",
            "LLaMA-4-Maverick", "LLaMA-3-70B", "Nova-Pro", "Nova-Micro",
            "GPT-4o", "GPT-4o-mini", "Gemini-2.5-Flash", "Gemini-2.5-Pro",
        )
        missing = [name for name in panel if name not in MODULE.MODEL_LABELS]
        self.assertEqual(missing, [], "figures would fall back to raw names")
        self.assertEqual(MODULE._label("Unregistered-Model"), "Unregistered-Model")

    def test_calibration_sources_separate_restraint_from_substantive(self):
        rows = []
        for index in range(8):
            label = "ABSTAIN" if index < 4 else "CONFIDENT_WRONG"
            # Half the abstentions follow the "report 0" instruction.
            confidence = (0.0 if index < 2 else 1.0) if index < 4 else 0.9
            rows.append(
                {
                    "model": "M",
                    "question_id": f"Q{index}",
                    "valid_for_ehq3": True,
                    "parsed_confidence": confidence,
                    "classification": {"label": label, "is_correct": False},
                }
            )
        profile = MODULE._calibration_source_rows(rows, ["M"], n_bins=10)[0]
        self.assertEqual(profile["n_abstain"], 4)
        self.assertEqual(profile["abstain_instruction_compliance"], 0.5)
        self.assertEqual(profile["n_substantive"], 4)
        # Substantive records are all wrong at 0.9 confidence.
        self.assertAlmostEqual(profile["substantive_ece"], 0.9)
        # The pooled, superseded definition mixed the two groups.
        self.assertNotAlmostEqual(
            profile["pooled_ece_superseded"], profile["substantive_ece"]
        )


class InlineReportTests(unittest.TestCase):
    """The runner builds the package before the run catalog exists."""

    def _run_dir(self, root):
        import json

        from ehq.analysis import build_analysis_report
        from ehq.artifacts import write_json
        from ehq.evaluation.scoring import compute_grouped_scores
        from ehq.reporting import write_standard_reports
        from ehq.types import ModelSpec

        models = [
            ModelSpec(name="Old-Gen", provider="asu", provider_model="o",
                      pair="p", generation="old"),
            ModelSpec(name="New-Gen", provider="asu", provider_model="n",
                      pair="p", generation="new"),
        ]
        labels = ["CONFIDENT_CORRECT", "CONFIDENT_WRONG", "ABSTAIN", "HEDGE"]
        results, ids = {}, []
        for offset, model in enumerate(models):
            rows = []
            for index in range(20):
                label = labels[(index + offset) % 4]
                rows.append(
                    _record(model.name, index, label,
                            0.9 if label.startswith("CONFIDENT") else 0.1)
                )
            results[model.name] = {
                "model": model.name,
                "scores": compute_ehq_scores(rows, weights=WEIGHTS, n_bins=10),
                "category_scores": compute_grouped_scores(
                    rows, group_field="category", weights=WEIGHTS, n_bins=10),
                "subcategory_scores": compute_grouped_scores(
                    rows, group_field="subcategory", weights=WEIGHTS, n_bins=10),
                "records": rows,
            }
            ids = [row["question_id"] for row in rows]

        run = root / "real_inline"
        run.mkdir(parents=True)
        write_json(run / "manifest.json", {
            "experiment_name": "inline",
            "framework_version": "0.3.1",
            "protocol_version": "1.1.4",
            "config": {"path": "config/experiment.json", "snapshot": {
                "seed": 42, "weights": WEIGHTS, "confidence": {"n_bins": 10}}},
            "dataset": {"sha256": "d"},
            "models": [{"name": m.name, "provider": m.provider,
                        "provider_model": m.provider_model,
                        "pair": m.pair, "generation": m.generation}
                       for m in models],
            "run": {"run_id": "inline-001", "mode": "real", "fingerprint": "fp",
                    "dataset_policy": "NON_PUBLISHABLE_CANDIDATE",
                    "selection": {"n_selected": len(ids), "question_ids": ids,
                                  "question_ids_sha256": "s"}},
        })
        for name, result in results.items():
            write_json(run / "models" / f"{name}.json", result)
        write_standard_reports(run, results)
        write_json(run / "analysis.json",
                   build_analysis_report(results, models, seed=42))
        return run

    def test_package_builds_before_the_source_catalog_exists(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run_dir(root)
            # This is the state the runner is in: no catalog yet.
            self.assertFalse((run / "artifact_catalog.json").exists())

            result = MODULE.build_report(
                run, run / "report", n_resamples=200, seed=42,
                verify_source=False, overwrite=True,
            )

            self.assertEqual(result["report_file"], "EHQ_REPORT_inline-001.md")
            self.assertTrue((run / "report" / result["report_file"]).is_file())
            self.assertTrue((run / "report" / "tables" / "model_scores.tex").is_file())
            self.assertTrue(
                (run / "report" / "figures" / "calibration-sources.png").is_file()
            )
            manifest = json.loads(
                (run / "report" / "report_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["source_artifact_catalog_sha256"])
            self.assertFalse(manifest["source_artifacts_verified"])

    def test_rebuilding_over_an_existing_package_is_allowed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            for _ in range(2):
                MODULE.build_report(
                    run, run / "report", n_resamples=100, seed=42,
                    verify_source=False, overwrite=True,
                )
            with self.assertRaises(FileExistsError):
                MODULE.build_report(
                    run, run / "report", n_resamples=100, seed=42,
                    verify_source=False,
                )

    def test_generated_tables_are_typeset_to_fit_the_text_block(self):
        """Wide tables ran into the margin because the layout ignored width."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            MODULE.build_report(
                run, run / "report", n_resamples=100, seed=42,
                verify_source=False, overwrite=True,
            )
            tables = sorted((run / "report" / "tables").glob("*.tex"))
            self.assertGreater(len(tables), 4)
            for table in tables:
                text = table.read_text(encoding="utf-8")
                with self.subTest(table=table.name):
                    if r"\begin{longtable}" in text:
                        # A longtable breaks across pages instead of scaling.
                        self.assertIn(r"\endhead", text)
                        continue
                    self.assertIn(r"\resizebox", text)
                    self.assertEqual(text.count(r"\resizebox"), 1)
                    self.assertIn(r"\small", text)

    def test_captions_describe_the_table_rather_than_naming_the_run(self):
        import re
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            MODULE.build_report(
                run, run / "report", n_resamples=100, seed=42,
                verify_source=False, overwrite=True,
            )
            for table in sorted((run / "report" / "tables").glob("*.tex")):
                text = table.read_text(encoding="utf-8")
                caption = re.search(r"\\caption\{(.*?)\}\s*\\?", text, re.S)
                self.assertIsNotNone(caption, table.name)
                body = caption.group(1)
                with self.subTest(table=table.name):
                    self.assertNotIn("inline-001", body)
                    self.assertGreater(len(body.split()), 12, body)

    def test_the_long_category_table_breaks_across_pages(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            MODULE.build_report(
                run, run / "report", n_resamples=100, seed=42,
                verify_source=False, overwrite=True,
            )
            text = (
                run / "report" / "tables" / "category_scores.tex"
            ).read_text(encoding="utf-8")
            self.assertIn(r"\begin{longtable}{llrrrr}", text)
            self.assertIn(r"\endfirsthead", text)
            self.assertIn(r"\endhead", text)
            # A longtable is not a float, so it must not be wrapped in one.
            self.assertNotIn(r"\begin{table}", text)

    def test_the_component_breakdown_figure_is_produced(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            MODULE.build_report(
                run, run / "report", n_resamples=100, seed=42,
                verify_source=False, overwrite=True,
            )
            figures = run / "report" / "figures"
            for suffix in (".png", ".pdf"):
                self.assertTrue((figures / f"component-breakdown{suffix}").is_file())

    def test_analysis_exclusion_is_removed_from_confirmatory_outputs(self):
        import csv
        import json
        import tempfile

        from ehq.artifacts import write_json

        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp))
            analysis_path = run / "analysis.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            analysis["confirmatory_panel"] = ["Old-Gen"]
            analysis["excluded_models"] = [
                {"model": "New-Gen", "reason": "route-quality failure"}
            ]
            write_json(analysis_path, analysis)
            MODULE.build_report(
                run,
                run / "report",
                n_resamples=100,
                seed=42,
                verify_source=False,
                overwrite=True,
            )
            with (run / "report" / "tables" / "model_scores.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                names = [row["model"] for row in csv.DictReader(stream)]
            self.assertEqual(names, ["Old-Gen"])
            self.assertTrue(
                (run / "report" / "tables" / "analysis_exclusions.csv").is_file()
            )
            with (run / "report" / "tables" / "coverage_by_model.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                coverage_names = [row["model"] for row in csv.DictReader(stream)]
            self.assertEqual(set(coverage_names), {"Old-Gen", "New-Gen"})
            self.assertTrue(
                (
                    run
                    / "report"
                    / "tables"
                    / "calibration_definition_sensitivity.tex"
                ).is_file()
            )
            category_tex = (
                run / "report" / "tables" / "category_scores.tex"
            ).read_text(encoding="utf-8")
            self.assertIn("CCQ values withheld from a supplied document", category_tex)


class LatexLayoutTests(unittest.TestCase):
    def test_a_column_spec_that_does_not_match_the_headers_is_rejected(self):
        """The spec is stated by hand for text columns; a silent mismatch
        shifts every cell in the table one column to the left."""

        with self.assertRaises(ValueError):
            MODULE._latex_table(
                caption="c", label="l", headers=("a", "b", "c"),
                rows=[("1", "2", "3")], column_spec="ll",
            )

    def test_text_columns_are_left_aligned(self):
        text = MODULE._latex_table(
            caption="c", label="l",
            headers=("Model", "Question", "Reason"),
            rows=[("m", "q", "r")],
            column_spec="lll",
        )
        self.assertIn(r"\begin{tabular}{lll}", text)

    def test_the_width_guard_only_shrinks(self):
        """A plain \\resizebox to \\textwidth would magnify a narrow table."""

        text = MODULE._latex_table(
            caption="c", label="l", headers=("a", "b"), rows=[("1", "2")]
        )
        self.assertIn(r"\ifdim\width>\linewidth\linewidth\else\width\fi", text)
