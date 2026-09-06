"""Command-line interface for validation and offline/online evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .analysis import (
    build_analysis_report,
    load_capability_counts,
    load_capability_scores,
)
from .artifacts import verify_run_artifacts, write_artifact_catalog, write_json
from .checkpoint import Checkpoint
from .config import (
    config_snapshot,
    load_experiment_config,
    load_models,
    model_registry_release_issues,
)
from .constants import (
    EHQ3_PROTOCOL,
    FRAMEWORK_VERSION,
    PROTOCOL_VERSION,
    RESPONSE_LABELS,
)
from .datasets import audit_ccq, load_dataset, validate_dataset
from .datasets.io import write_dataset
from .datasets.legacy import migrate_legacy_seed
from .datasets.ledger import validate_fact_ledger
from .evaluation.runner import (
    client_for_model,
    credential_env_for_model,
    run_model,
)
from .env import load_local_env
from .hashing import sha256_file, sha256_tree
from .prompts import ANSWER_PROMPT_VERSION, CONFIDENCE_PROMPT_VERSION
from .provenance import build_manifest, experiment_fingerprint
from .reporting import (
    check_reporting_dependencies,
    export_excel,
    export_figures,
    write_standard_reports,
)
from .selection import select_items, selection_manifest


_PROJECT_TEMPLATE_FILES = (
    ".env.example",
    "config/adjudication.example.json",
    "config/capability_scores.example.csv",
    "config/experiment.json",
    "config/models.json",
    "config/pilot_primary_review.json",
    "config/smoke.json",
    "data/examples/EHQ-20-smoke.json",
    "data/releases/EHQ-3000.json",
)


def _init_project(args: argparse.Namespace) -> int:
    """Create a safe, minimal working directory for an installed package."""

    destination = Path(args.directory).resolve()
    conflicts = [
        destination / relative
        for relative in _PROJECT_TEMPLATE_FILES
        if (destination / relative).exists()
    ]
    if conflicts and not args.force:
        shown = ", ".join(str(path) for path in conflicts[:5])
        suffix = " ..." if len(conflicts) > 5 else ""
        raise SystemExit(
            "Project initialization refused because template files already "
            f"exist: {shown}{suffix}. Use --force only if overwriting those "
            "specific template files is intended."
        )

    template_root = files("ehq").joinpath("resources", "project")
    for relative in _PROJECT_TEMPLATE_FILES:
        source = template_root.joinpath(*Path(relative).parts)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, target.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)

    print(f"Initialized EHQ project: {destination}")
    print("Next steps:")
    print("  1. Copy .env.example to .env and add credentials locally.")
    print("  2. Review config/models.json and select your provider/model route.")
    print(
        "  3. Run: ehq dry-run --config config/smoke.json "
        '--model "openai-example" --allow-candidate --run-id framework-smoke'
    )
    return 0


def _dataset_path(_: argparse.Namespace) -> int:
    """Print the installed, version-pinned EHQ-3000 resource path."""

    path = files("ehq").joinpath(
        "resources", "project", "data", "releases", "EHQ-3000.json"
    )
    print(path)
    return 0


def _validate(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    items = load_dataset(path)
    report = validate_dataset(
        items,
        expected_total=args.expected_total,
        expected_per_category=args.expected_per_category,
        require_source_evidence=args.require_source_evidence,
        require_pcq_temporal_novelty=args.require_pcq_temporal_novelty,
        allow_pending_human_review=args.allow_candidate,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.valid else 2


def _split_values(values: Sequence[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise SystemExit("run/model name contains no filesystem-safe characters")
    return cleaned


def _missing_runtime_dependencies(models: Sequence[Any]) -> list[str]:
    """Return mandatory packages missing for the selected real providers."""

    providers = {model.provider for model in models}
    required = []
    if providers & {"openai", "huggingface", "openai_compatible"}:
        required.append("requests")
    return [name for name in required if importlib.util.find_spec(name) is None]


def _load_adjudication(path: Path | None) -> tuple[Dict[str, str], str | None]:
    if path is None:
        return {}, None
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    if isinstance(raw, Mapping) and isinstance(raw.get("adjudications"), list):
        for row in raw["adjudications"]:
            key = f"{row['model']}::{row['question_id']}"
            mapping[key] = str(row["label"])
    elif isinstance(raw, Mapping):
        mapping = {str(key): str(value) for key, value in raw.items()}
    else:
        raise SystemExit("Adjudication must be an object or contain an adjudications list")
    invalid = sorted(set(mapping.values()) - set(RESPONSE_LABELS))
    if invalid:
        raise SystemExit(f"Invalid adjudication labels: {', '.join(invalid)}")
    return mapping, sha256_file(path)


# Progress and provider activity go to stderr so the machine-readable result on
# stdout stays clean. Human-readable is the default; --json-events restores the
# JSON-lines stream for anything that parses it.
_EVENT_FORMAT = "text"


def _count(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)


def _duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    seconds = int(round(float(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _format_event(event: str, values: Mapping[str, Any]) -> str | None:
    model = values.get("model")
    if event == "study_preflight":
        blockers = values.get("publication_blockers") or []
        lines = [
            f"Study: {values.get('study')} | "
            f"{len(values.get('models') or [])} model(s) x "
            f"{_count(values.get('items'))} item(s)",
            f"Publication status: {values.get('publication_status')}",
        ]
        if blockers:
            lines.append("Publication blockers: " + ", ".join(map(str, blockers)))
        for warning in values.get("operational_warnings") or []:
            lines.append(f"WARNING: route not operationally verified: {warning}")
        return "\n".join(lines)
    if event == "run_started":
        return (
            f"Output: {values.get('run_dir')}\n"
            f"Dataset policy: {values.get('dataset_policy')}\n"
        )
    if event == "model_started":
        return (
            f"[MODEL {values.get('index')}/{values.get('total')}] {model}"
        )
    if event == "model_progress":
        if values.get("phase") == "resume_loaded":
            return f"[{model}] resumed {_count(values.get('resumed'))} record(s)"
        rate = values.get("items_per_minute")
        parts = [
            f"[{model}] {_count(values.get('completed'))}/"
            f"{_count(values.get('total'))} ({values.get('percent')}%)",
            f"valid: {_count(values.get('valid_ehq12'))}",
            f"conf: {_count(values.get('valid_ehq3'))}",
            f"fail: {_count(values.get('technical_failures'))}",
        ]
        if rate:
            parts.append(f"{_count(rate)}/min")
        parts.append(f"ETA {_duration(values.get('eta_seconds'))}")
        return " | ".join(parts)
    if event == "model_completed":
        ehq = values.get("ehq")
        score = f"{float(ehq):.4f}" if isinstance(ehq, (int, float)) else "undefined"
        return (
            f"[DONE] {model} | EHQ: {score} | "
            f"retryable: {_count(values.get('retryable'))}"
        )
    if event == "provider_retry_scheduled":
        return (
            f"[RETRY] {model} {values.get('purpose')} ({values.get('error_type')}); "
            f"attempt {values.get('next_attempt')}/{values.get('max_retries')} "
            f"in {values.get('wait_seconds')}s"
        )
    if event == "provider_request_recovered":
        return (
            f"[RECOVERED] {model} {values.get('purpose')} after "
            f"attempt {values.get('attempt')}"
        )
    if event in ("provider_request_exhausted", "provider_request_terminal_failure"):
        return (
            f"[FAILED] {model} {values.get('purpose')}: {values.get('error_type')}"
            + (
                f" - {values.get('error_message')}"
                if values.get("error_message")
                else ""
            )
        )
    return None


def _emit_event(event: str, **values: Any) -> None:
    if _EVENT_FORMAT == "json":
        print(
            json.dumps({"event": event, **values}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
        return
    line = _format_event(event, values)
    if line is None:
        line = f"[{event}] " + " ".join(
            f"{key}={value}" for key, value in values.items() if value is not None
        )
    print(line, file=sys.stderr, flush=True)


def _make_model_progress_reporter(model_name: str):
    """Return a throttled JSON-lines progress reporter for one model."""

    state = {"last_elapsed": 0.0}

    def report(progress: Mapping[str, Any]) -> None:
        total = int(progress.get("total_items") or 0)
        completed = int(progress.get("completed_items") or 0)
        session_completed = int(progress.get("completed_this_session") or 0)
        elapsed = float(progress.get("elapsed_seconds") or 0.0)
        phase = str(progress.get("phase") or "evaluating")
        if total <= 20:
            item_interval = 5
        elif total <= 100:
            item_interval = 10
        elif total <= 500:
            item_interval = 25
        else:
            item_interval = 100
        should_emit = (
            phase == "resume_loaded"
            or session_completed == 1
            or completed == total
            or (session_completed > 0 and session_completed % item_interval == 0)
            or elapsed - state["last_elapsed"] >= 30.0
        )
        if not should_emit:
            return
        state["last_elapsed"] = elapsed
        _emit_event(
            "model_progress",
            model=model_name,
            phase=phase,
            completed=completed,
            total=total,
            percent=(round(100.0 * completed / total, 2) if total else 100.0),
            resumed=int(progress.get("resumed_records") or 0),
            valid_ehq12=int(progress.get("valid_ehq12") or 0),
            valid_ehq3=int(progress.get("valid_ehq3") or 0),
            terminal_missing_confidence=int(
                progress.get("terminal_missing_confidence") or 0
            ),
            retryable=int(progress.get("retryable_records") or 0),
            technical_failures=int(progress.get("technical_failures") or 0),
            elapsed_seconds=round(elapsed, 1),
            items_per_minute=(
                round(float(progress["items_per_minute"]), 2)
                if progress.get("items_per_minute") is not None
                else None
            ),
            eta_seconds=(
                round(float(progress["eta_seconds"]), 1)
                if progress.get("eta_seconds") is not None
                else None
            ),
        )

    return report


def _make_request_event_reporter(model_name: str):
    """Expose provider retry/backoff activity without logging prompts or secrets."""

    event_names = {
        "retry_scheduled": "provider_retry_scheduled",
        "request_recovered": "provider_request_recovered",
        "request_terminal_failure": "provider_request_terminal_failure",
        "request_exhausted": "provider_request_exhausted",
    }

    def report(values: Mapping[str, Any]) -> None:
        phase = str(values.get("phase") or "")
        event = event_names.get(phase)
        if event is None:
            return
        public = {
            key: values.get(key)
            for key in (
                "purpose",
                "attempt",
                "next_attempt",
                "max_retries",
                "wait_seconds",
                "error_type",
                "error_message",
            )
            if values.get(key) is not None
        }
        _emit_event(event, model=model_name, **public)

    return report


def _run(args: argparse.Namespace, *, offline: bool) -> int:
    try:
        check_reporting_dependencies(
            excel=args.excel,
            figures=args.figures,
            report=getattr(args, "report", False),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_experiment_config(Path(args.config))
    models = load_models(config.models_path)
    requested_models = _split_values([*args.model, *args.models])
    if requested_models:
        known = {model.name: model for model in models}
        unknown = sorted(set(requested_models) - set(known))
        if unknown:
            raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")
        models = [known[name] for name in dict.fromkeys(requested_models)]

    registry_issues = model_registry_release_issues(config.models_path, models)
    if (
        not offline
        and registry_issues
        and not args.allow_unverified_model_registry
    ):
        preview = "\n".join(f"- {issue}" for issue in registry_issues[:12])
        raise SystemExit(
            "Model registry is not release-verified. Real requests were not started.\n"
            f"{preview}\n"
            "Use --allow-unverified-model-registry only for explicitly "
            "non-publishable engineering runs."
        )
    if not offline:
        missing_dependencies = _missing_runtime_dependencies(models)
        if missing_dependencies:
            raise SystemExit(
                "Real requests were not started; missing runtime dependency/dependencies: "
                + ", ".join(missing_dependencies)
                + ". Install the project before retrying (for example: "
                "python -m pip install -e .)."
            )
        missing_credentials = sorted(
            {
                credential
                for model in models
                if (credential := credential_env_for_model(model))
                and not os.getenv(credential)
            }
        )
        if missing_credentials:
            raise SystemExit(
                "Real requests were not started; missing credential(s): "
                + ", ".join(missing_credentials)
            )
    safe_model_names = [_safe_name(model.name) for model in models]
    if len(safe_model_names) != len(set(safe_model_names)):
        raise SystemExit("Model display names collide after filename sanitization")

    project_root = config.source_path.parent.parent
    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise SystemExit(
            "Dataset file is unavailable; no evaluation request was started: "
            f"{dataset_path}\n"
            "The default release config is intentionally fail-closed until a "
            "scientifically approved EHQ-3000 release is placed there."
        )
    items = load_dataset(dataset_path)
    requirements = config.dataset_requirements
    report = validate_dataset(
        items,
        expected_total=requirements.expected_total,
        expected_per_category=requirements.expected_per_category,
        require_source_evidence=requirements.require_source_evidence,
        require_pcq_temporal_novelty=(
            requirements.require_pcq_temporal_novelty
        ),
        allow_pending_human_review=args.allow_candidate,
    )
    if not report.valid:
        errors = [issue for issue in report.issues if issue.severity == "error"]
        warnings = [issue for issue in report.issues if issue.severity == "warning"]
        first = errors[:10]
        raise SystemExit(
            "Dataset validation failed before evaluation "
            f"({len(errors)} error(s), {len(warnings)} warning(s)):\n"
            + "\n".join(
                f"- {issue.code}"
                + (f" [{issue.question_id}]" if issue.question_id else "")
                + f": {issue.message}"
                for issue in first
            )
        )

    categories = _split_values(args.category)
    selected = select_items(
        items, categories=categories or None, limit=args.limit, seed=config.seed
    )
    if not selected:
        raise SystemExit("No dataset items match the requested selection")
    selection = selection_manifest(selected)
    adjudication_path = Path(args.adjudication).resolve() if args.adjudication else None
    adjudication, adjudication_sha = _load_adjudication(adjudication_path)
    capability_path = (
        Path(args.capability_scores).resolve() if args.capability_scores else None
    )
    capability_sha = sha256_file(capability_path) if capability_path else None

    mode = "dry-run" if offline else "real"
    dataset_hash = sha256_file(dataset_path)
    provider_endpoints = {
        model.name: client_for_model(config, model, offline=offline).endpoint
        for model in models
    }
    common_fingerprint = {
        "framework_version": FRAMEWORK_VERSION,
        "framework_source_sha256": sha256_tree(project_root / "src" / "ehq"),
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "config": config_snapshot(config),
        "dataset_sha256": dataset_hash,
        "models_config_sha256": sha256_file(config.models_path),
        "selection_sha256": selection["question_ids_sha256"],
        "adjudication_sha256": adjudication_sha,
        "capability_scores_sha256": capability_sha,
        "allow_pending_human_review": args.allow_candidate,
        "allow_unverified_model_registry": args.allow_unverified_model_registry,
        "provider_endpoints": provider_endpoints,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "confidence_prompt_version": CONFIDENCE_PROMPT_VERSION,
    }
    run_fingerprint = experiment_fingerprint(
        {**common_fingerprint, "models": [asdict(model) for model in models]}
    )
    run_id = _safe_name(
        args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else config.output_dir
    run_dir = output_dir / f"{mode}_{run_id}"
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"Run directory already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest_path = run_dir / "manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        existing_fingerprint = (existing_manifest.get("run") or {}).get("fingerprint")
        if existing_fingerprint != run_fingerprint:
            raise SystemExit(
                "Refusing resume because the run fingerprint changed: "
                f"{existing_fingerprint!r} != {run_fingerprint!r}"
            )

    checkpoint_root = config.checkpoint_dir / run_fingerprint[:16]
    publication_blockers = []
    if args.allow_candidate:
        publication_blockers.append("dataset_human_gates_pending")
    if not offline and registry_issues:
        publication_blockers.append("model_registry_not_release_verified")
    dataset_policy = (
        "NON_PUBLISHABLE_CANDIDATE"
        if publication_blockers
        else "RELEASE_GATE_PASSED"
    )
    manifest = build_manifest(
        project_root=project_root,
        experiment_name=config.experiment_name,
        config_path=config.source_path,
        dataset_path=dataset_path,
        models_path=config.models_path,
        models=models,
        configuration=config_snapshot(config),
        run={
            "fingerprint": run_fingerprint,
            "framework_source_sha256": common_fingerprint[
                "framework_source_sha256"
            ],
            "run_id": run_id,
            "mode": mode,
            "dataset_policy": dataset_policy,
            "allow_pending_human_review": args.allow_candidate,
            "publication_blockers": publication_blockers,
            "model_registry_release_issues": registry_issues,
            "selection": selection,
            "categories": categories or None,
            "limit": args.limit,
            "seed": config.seed,
            "adjudication_path": str(adjudication_path) if adjudication_path else None,
            "adjudication_sha256": adjudication_sha,
            "capability_scores_path": str(capability_path) if capability_path else None,
            "capability_scores_sha256": capability_sha,
            "answer_prompt_version": ANSWER_PROMPT_VERSION,
            "confidence_prompt_version": CONFIDENCE_PROMPT_VERSION,
            "provider_endpoints": provider_endpoints,
        },
    )
    write_json(existing_manifest_path, manifest)
    _emit_event(
        "run_started",
        run_dir=str(run_dir),
        models=len(models),
        items=len(selected),
        dataset_policy=dataset_policy,
    )
    results: Dict[str, Dict[str, Any]] = {}
    for index, model in enumerate(models, 1):
        model_fingerprint = experiment_fingerprint(
            {**common_fingerprint, "model": asdict(model)}
        )
        checkpoint = Checkpoint(
            checkpoint_root / f"{_safe_name(model.name)}.jsonl",
            fingerprint=model_fingerprint,
        )
        _emit_event(
            "model_started", model=model.name, index=index, total=len(models)
        )
        result = run_model(
            selected,
            model,
            config,
            checkpoint=checkpoint,
            offline=offline,
            adjudication=adjudication,
            progress_callback=_make_model_progress_reporter(model.name),
            request_event_callback=_make_request_event_reporter(model.name),
        )
        results[model.name] = result
        write_json(run_dir / "models" / f"{_safe_name(model.name)}.json", result)
        _emit_event(
            "model_completed",
            model=model.name,
            ehq=result["scores"].get("EHQ"),
            retryable=result["n_retryable_records"],
        )
    summary = write_standard_reports(run_dir, results)
    capability_scores = load_capability_scores(capability_path) if capability_path else None
    capability_counts = (
        load_capability_counts(capability_path) if capability_path else None
    )
    excluded_models = _parse_model_exclusions(args.exclude_model)
    analysis = build_analysis_report(
        results,
        models,
        seed=config.seed,
        capability_scores=capability_scores,
        capability_counts=capability_counts,
        excluded_models=excluded_models,
    )
    write_json(run_dir / "analysis.json", analysis)
    if args.excel:
        export_excel(run_dir / "EHQ_results.xlsx", results)
    if args.figures:
        export_figures(run_dir / "figures", results)
    report_result = None
    if getattr(args, "report", False):
        from .publication import build_report

        _emit_event("report_started", run_dir=str(run_dir))
        try:
            report_result = build_report(
                run_dir,
                run_dir / "report",
                n_resamples=int(getattr(args, "report_resamples", 10_000)),
                seed=config.seed,
                verify_source=False,
                overwrite=True,
                label_prefix=getattr(args, "report_label_prefix", None),
            )
        except Exception as exc:  # noqa: BLE001 - see comment
            # The evaluation itself succeeded and its artifacts are already
            # written. A derived package is never worth discarding that work,
            # so every failure mode is caught here rather than an enumerated
            # few: the run completes, and the reason is reported.
            report_result = {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _emit_event("report_skipped", reason=report_result["reason"])
        else:
            _emit_event(
                "report_completed",
                report_dir=report_result["output_dir"],
                report_file=report_result["report_file"],
            )
    catalog = write_artifact_catalog(run_dir)
    verification = verify_run_artifacts(run_dir)
    payload = {
        "run_dir": str(run_dir),
        "fingerprint": run_fingerprint,
        "dataset_policy": dataset_policy,
        "ehq3_protocol": EHQ3_PROTOCOL,
        "summary": summary,
        "analysis_status": {
            "rq1": analysis["rq1_capability_relationship"]["status"],
            "rq2": analysis["rq2_generational_pairs"]["status"],
        },
        "artifact_count": len(catalog["artifacts"]),
        "artifacts_verified": verification["valid"],
        "report": report_result,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_render_run_summary(payload))
    return 0


def _render_run_summary(payload: Mapping[str, Any]) -> str:
    """Human-readable close-out for a completed run."""

    rows = (payload["summary"] or {}).get("models") or []
    lines = [
        "",
        "Evaluation complete",
        f"Output directory : {payload['run_dir']}",
        f"Fingerprint      : {payload['fingerprint']}",
        f"EHQ3 protocol    : {payload['ehq3_protocol']}",
        f"Dataset policy   : {payload['dataset_policy']}",
        f"Artifacts        : {payload['artifact_count']} file(s), "
        f"verified: {'yes' if payload['artifacts_verified'] else 'NO'}",
        f"Analysis         : RQ1 {payload['analysis_status']['rq1']}, "
        f"RQ2 {payload['analysis_status']['rq2']}",
    ]
    report = payload.get("report")
    if report and report.get("status") == "skipped":
        lines.append(f"Publication pkg  : SKIPPED - {report['reason']}")
    elif report:
        lines.append(
            f"Publication pkg  : {report['output_dir']} "
            f"({report['report_file']})"
        )
    lines += [
        "",
        f"{'rank':>4}  {'model':<24}{'EHQ1':>8}{'EHQ2':>8}{'EHQ3':>8}{'EHQ':>8}"
        f"{'scored':>9}{'calib':>8}{'fail':>6}",
        "-" * 84,
    ]

    def number(value: Any) -> str:
        return f"{float(value):>8.4f}" if isinstance(value, (int, float)) else f"{'-':>8}"

    incomplete = []
    for row in rows:
        rank = row.get("rank")
        lines.append(
            f"{(f'{rank:g}' if isinstance(rank, (int, float)) else '-'):>4}  "
            f"{str(row.get('model')):<24}"
            f"{number(row.get('EHQ1'))}{number(row.get('EHQ2'))}"
            f"{number(row.get('EHQ3'))}{number(row.get('EHQ'))}"
            f"{_count(row.get('n_valid_ehq12')):>9}"
            f"{_count(row.get('n_ehq3_calibration')):>8}"
            f"{_count(row.get('technical_failures')):>6}"
        )
        if row.get("technical_failures") or row.get("n_missing_confidence"):
            incomplete.append(row)

    if incomplete:
        lines.append("")
        lines.append("Incomplete coverage (excluded from the affected denominators):")
        for row in incomplete:
            lines.append(
                f"  {row.get('model')}: "
                f"{_count(row.get('technical_failures'))} technical failure(s), "
                f"{_count(row.get('n_missing_confidence'))} missing confidence"
            )
    lines.append("")
    return "\n".join(lines)


def _preset_run(args: argparse.Namespace, *, study: str) -> int:
    """Translate a safe, concise pilot/full command into the normative runner."""

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    all_models = load_models(config.models_path)
    requested = _split_values([*args.model, *args.models])
    if args.all_verified:
        if requested:
            raise SystemExit("Use either --all-verified or --model/--models, not both")
        requested = [
            model.name
            for model in all_models
            if model.operational_status == "verified"
        ]
    if not requested:
        raise SystemExit(
            f"ehq {study} requires --model, --models, or --all-verified; "
            "this prevents an accidental costly full-panel run"
        )
    known = {model.name: model for model in all_models}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")
    selected_models = [known[name] for name in dict.fromkeys(requested)]
    non_verified = [
        f"{model.name} ({model.operational_status})"
        for model in selected_models
        if model.operational_status != "verified"
    ]
    project_root = config.source_path.parent.parent
    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_path = dataset_path.resolve()
    allow_candidate = False
    if dataset_path.is_file():
        items = load_dataset(dataset_path)
        requirements = config.dataset_requirements
        strict_report = validate_dataset(
            items,
            expected_total=requirements.expected_total,
            expected_per_category=requirements.expected_per_category,
            require_source_evidence=requirements.require_source_evidence,
            require_pcq_temporal_novelty=(
                requirements.require_pcq_temporal_novelty
            ),
            allow_pending_human_review=False,
        )
        if not strict_report.valid:
            candidate_report = validate_dataset(
                items,
                expected_total=requirements.expected_total,
                expected_per_category=requirements.expected_per_category,
                require_source_evidence=requirements.require_source_evidence,
                require_pcq_temporal_novelty=(
                    requirements.require_pcq_temporal_novelty
                ),
                allow_pending_human_review=True,
            )
            allow_candidate = candidate_report.valid

    registry_issues = model_registry_release_issues(
        config.models_path, selected_models
    )
    publication_blockers = []
    if allow_candidate:
        publication_blockers.append("dataset_human_gates_pending")
    if registry_issues:
        publication_blockers.append("model_registry_not_release_verified")
    if non_verified:
        publication_blockers.append("selected_route_not_operationally_verified")
    _emit_event(
        "study_preflight",
        study=study,
        models=[model.name for model in selected_models],
        items=(args.limit if study == "pilot" else 3000),
        publication_status=(
            "NON_PUBLISHABLE_CANDIDATE"
            if publication_blockers
            else "RELEASE_GATE_PASSED"
        ),
        publication_blockers=publication_blockers,
        operational_warnings=non_verified,
    )

    forwarded = argparse.Namespace(
        config=str(config_path),
        model=[model.name for model in selected_models],
        models=[],
        dataset=str(dataset_path),
        category=list(args.category),
        limit=(args.limit if study == "pilot" else None),
        output_dir=args.output_dir,
        run_id=args.run_id,
        resume=args.resume,
        adjudication=args.adjudication,
        capability_scores=args.capability_scores,
        exclude_model=list(args.exclude_model),
        allow_candidate=allow_candidate,
        allow_unverified_model_registry=bool(registry_issues),
        excel=args.excel,
        figures=args.figures,
        report=args.report,
        report_resamples=args.report_resamples,
        report_label_prefix=args.report_label_prefix,
        json=getattr(args, "json", False),
    )
    return _run(forwarded, offline=False)


def _status(args: argparse.Namespace) -> int:
    """Report a completed or interrupted run without mutating it."""

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    expected_models = [
        str(row.get("name"))
        for row in manifest.get("models", [])
        if isinstance(row, Mapping) and row.get("name")
    ]
    completed_models = []
    for model_name in expected_models:
        if (run_dir / "models" / f"{_safe_name(model_name)}.json").is_file():
            completed_models.append(model_name)

    checkpoint_counts: Dict[str, int] = {}
    config_path = Path(str((manifest.get("config") or {}).get("path") or ""))
    fingerprint = str(run.get("fingerprint") or "")
    if config_path.is_file() and len(fingerprint) >= 16:
        try:
            checkpoint_dir = (
                load_experiment_config(config_path).checkpoint_dir
                / fingerprint[:16]
            )
            for model_name in expected_models:
                path = checkpoint_dir / f"{_safe_name(model_name)}.jsonl"
                if path.is_file():
                    checkpoint_counts[model_name] = sum(
                        1 for _ in Checkpoint(path).records()
                    )
        except (OSError, ValueError):
            checkpoint_counts = {}

    summary_path = run_dir / "summary.json"
    catalog_path = run_dir / "artifact_catalog.json"
    complete = summary_path.is_file()
    verification = (
        verify_run_artifacts(run_dir)
        if complete and catalog_path.is_file()
        else None
    )
    payload = {
        "run_dir": str(run_dir),
        "status": (
            "completed_verified"
            if verification and verification.get("valid")
            else "completed_unverified"
            if complete
            else "in_progress_or_interrupted"
        ),
        "dataset_policy": run.get("dataset_policy"),
        "expected_models": expected_models,
        "completed_models": completed_models,
        "remaining_models": [
            name for name in expected_models if name not in completed_models
        ],
        "checkpoint_terminal_records": checkpoint_counts,
        "selection_total": ((run.get("selection") or {}).get("n_selected")),
        "artifacts": verification,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not complete or verification is None or verification.get("valid") else 2


def _parse_model_exclusions(values) -> dict:
    """Parse repeated ``--exclude-model NAME=reason`` arguments.

    A reason is required. An exclusion without one is indistinguishable, later,
    from a model that was quietly dropped because its numbers were inconvenient,
    and the whole point of recording it in the analysis output is that a reader
    can tell those apart.
    """

    exclusions = {}
    for value in values or []:
        name, separator, reason = str(value).partition("=")
        name, reason = name.strip(), reason.strip()
        if not name or not separator or not reason:
            raise SystemExit(
                "--exclude-model wants NAME=reason, e.g. "
                '--exclude-model "Gemini-2.5-Pro=33% of answers truncated"; '
                f"got {value!r}"
            )
        exclusions[name] = reason
    return exclusions


def _verify_run(args: argparse.Namespace) -> int:
    report = verify_run_artifacts(Path(args.run_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


def _migrate_legacy(args: argparse.Namespace) -> int:
    source = Path(args.dataset)
    output = Path(args.output)
    audit = Path(args.audit)
    items = load_dataset(source)
    migrated, report = migrate_legacy_seed(items)
    write_dataset(output, migrated)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "audit": str(audit),
                "retained_count": report["retained_count"],
                "retained_category_counts": report["retained_category_counts"],
                "dropped_exact_duplicates": len(
                    report["dropped_exact_duplicates"]
                ),
                "excluded_for_rebuild": len(report["excluded_for_rebuild"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _validate_ledger(args: argparse.Namespace) -> int:
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    report = validate_fact_ledger(
        ledger,
        expected_total=args.expected_total,
        expected_per_subcategory=args.expected_per_subcategory,
        expected_subcategories=args.expected_subcategory,
        require_release_evidence=args.require_release_evidence,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


def _audit_ccq(args: argparse.Namespace) -> int:
    items = load_dataset(Path(args.dataset))
    attempts = []
    if args.attempts:
        for line_number, line in enumerate(
            Path(args.attempts).read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(
                    f"Invalid attempt record at {args.attempts}:{line_number}"
                )
            attempts.append(value)
    report = audit_ccq(
        items,
        attempts=attempts,
        expected_per_subcategory=args.expected_per_subcategory,
        question_similarity_threshold=args.question_similarity_threshold,
        document_similarity_threshold=args.document_similarity_threshold,
        document_ngram_threshold=args.document_ngram_threshold,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["automatic_gate_passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ehq")
    parser.add_argument(
        "--version",
        action="version",
        version=f"EHQ {FRAMEWORK_VERSION} (protocol {PROTOCOL_VERSION})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init",
        help="Create a minimal local EHQ study directory",
    )
    initialize.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Destination directory (default: current directory)",
    )
    initialize.add_argument(
        "--force",
        action="store_true",
        help="Overwrite only EHQ template files already present",
    )
    initialize.set_defaults(handler=_init_project)

    installed_data = subparsers.add_parser(
        "dataset-path",
        help="Print the installed EHQ-3000 dataset path",
    )
    installed_data.set_defaults(handler=_dataset_path)

    validate = subparsers.add_parser("validate", help="Validate an EHQ dataset")
    validate.add_argument("dataset")
    validate.add_argument("--expected-total", type=int)
    validate.add_argument("--expected-per-category", type=int)
    validate.add_argument("--require-source-evidence", action="store_true")
    validate.add_argument(
        "--require-pcq-temporal-novelty",
        action="store_true",
        help="Require human attestation that each PCQ claim itself is post-cutoff",
    )
    validate.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Allow only declared pending human gates; results remain non-publishable",
    )
    validate.set_defaults(handler=_validate)

    migrate = subparsers.add_parser(
        "migrate-legacy", help="Create an auditable FEQ/HNQ seed from EHQ-750"
    )
    migrate.add_argument("dataset")
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--audit", required=True)
    migrate.set_defaults(handler=_migrate_legacy)

    validate_ledger = subparsers.add_parser(
        "validate-ledger", help="Validate an atomic-fact source ledger"
    )
    validate_ledger.add_argument("ledger")
    validate_ledger.add_argument("--expected-total", type=int)
    validate_ledger.add_argument("--expected-per-subcategory", type=int)
    validate_ledger.add_argument("--expected-subcategory", action="append")
    validate_ledger.add_argument("--require-release-evidence", action="store_true")
    validate_ledger.set_defaults(handler=_validate_ledger)

    ccq_audit = subparsers.add_parser(
        "audit-ccq",
        help="Audit CCQ balance, provenance, cost, and near duplicates",
    )
    ccq_audit.add_argument("dataset")
    ccq_audit.add_argument("--attempts")
    ccq_audit.add_argument("--output")
    ccq_audit.add_argument("--expected-per-subcategory", type=int)
    ccq_audit.add_argument("--question-similarity-threshold", type=float, default=0.88)
    ccq_audit.add_argument("--document-similarity-threshold", type=float, default=0.82)
    ccq_audit.add_argument("--document-ngram-threshold", type=float, default=0.40)
    ccq_audit.set_defaults(handler=_audit_ccq)

    verify = subparsers.add_parser(
        "verify-run", help="Verify every run artifact against its SHA-256 catalog"
    )
    verify.add_argument("run_dir")
    verify.set_defaults(handler=_verify_run)

    status = subparsers.add_parser(
        "status",
        help="Show progress/completion and artifact integrity for a run",
    )
    status.add_argument("run_dir")
    status.set_defaults(handler=_status)

    for name, help_text, default_config, default_limit in (
        (
            "pilot",
            "Run a concise, safeguarded real-provider pilot",
            "config/pilot_primary_review.json",
            20,
        ),
        (
            "full",
            "Run a safeguarded full EHQ-3000 evaluation",
            "config/experiment.json",
            None,
        ),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default=default_config)
        command.add_argument("--dataset")
        command.add_argument("--model", action="append", default=[])
        command.add_argument(
            "--models",
            action="append",
            default=[],
            help="Comma-separated model display names",
        )
        command.add_argument(
            "--all-verified",
            action="store_true",
            help="Select every route whose operational_status is verified",
        )
        command.add_argument("--category", action="append", default=[])
        if default_limit is not None:
            command.add_argument("--limit", type=int, default=default_limit)
        command.add_argument("--output-dir")
        command.add_argument("--run-id")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--adjudication")
        command.add_argument("--capability-scores")
        command.add_argument(
            "--exclude-model",
            action="append",
            default=[],
            metavar="NAME=REASON",
            help=(
                "Withhold a model from the confirmatory analyses while keeping "
                "its records and reporting its scores and the reason. Repeatable."
            ),
        )
        command.add_argument(
            "--excel",
            action=argparse.BooleanOptionalAction,
            default=(name == "full"),
        )
        command.add_argument(
            "--figures",
            action=argparse.BooleanOptionalAction,
            default=(name == "full"),
        )
        command.add_argument(
            "--json",
            action="store_true",
            help="Print the machine-readable run result instead of the table",
        )
        command.add_argument(
            "--json-events",
            action="store_true",
            help="Emit progress as JSON lines instead of human-readable text",
        )
        command.add_argument(
            "--report",
            action=argparse.BooleanOptionalAction,
            default=(name == "full"),
            help="Build the tables/LaTeX/figures package into <run>/report",
        )
        command.add_argument(
            "--report-resamples",
            type=int,
            default=10_000,
            help="Bootstrap resamples for the publication package",
        )
        command.add_argument(
            "--report-label-prefix",
            help=(
                "Stable LaTeX label prefix for a manuscript build; defaults to "
                "the run id"
            ),
        )
        command.set_defaults(
            handler=lambda args, study=name: _preset_run(args, study=study),
            limit=default_limit,
        )

    for name, help_text, offline in (
        ("dry-run", "Run the pipeline with an offline deterministic model", True),
        ("run", "Run real configured model requests", False),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="config/experiment.json")
        command.add_argument("--model", action="append", default=[])
        command.add_argument(
            "--models",
            action="append",
            default=[],
            help="Comma-separated model display names",
        )
        command.add_argument("--dataset")
        command.add_argument("--category", action="append", default=[])
        command.add_argument("--limit", type=int)
        command.add_argument("--output-dir")
        command.add_argument("--run-id")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--adjudication")
        command.add_argument(
            "--capability-scores",
            help="Optional JSON/CSV model capability scores for RQ1",
        )
        command.add_argument(
            "--exclude-model",
            action="append",
            default=[],
            metavar="NAME=REASON",
            help=(
                "Withhold a model from the confirmatory analyses while keeping "
                "its records and reporting its scores and the reason. Repeatable."
            ),
        )
        command.add_argument("--allow-candidate", action="store_true")
        command.add_argument(
            "--allow-unverified-model-registry",
            action="store_true",
            help="Permit a stamped non-publishable real run with registry evidence gaps",
        )
        command.add_argument("--excel", action="store_true")
        command.add_argument("--figures", action="store_true")
        command.add_argument(
            "--json",
            action="store_true",
            help="Print the machine-readable run result instead of the table",
        )
        command.add_argument(
            "--json-events",
            action="store_true",
            help="Emit progress as JSON lines instead of human-readable text",
        )
        command.add_argument(
            "--report",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Build the tables/LaTeX/figures package into <run>/report",
        )
        command.add_argument(
            "--report-resamples",
            type=int,
            default=10_000,
            help="Bootstrap resamples for the publication package",
        )
        command.add_argument(
            "--report-label-prefix",
            help=(
                "Stable LaTeX label prefix for a manuscript build; defaults to "
                "the run id"
            ),
        )
        command.set_defaults(
            handler=lambda args, offline=offline: _run(args, offline=offline)
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_local_env(Path(".env"))
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "json_events", False):
        global _EVENT_FORMAT
        _EVENT_FORMAT = "json"
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
