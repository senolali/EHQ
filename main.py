"""
EHQ-3000 Evaluation Pipeline
==============================
FULLY CONFIG-DRIVEN, mirroring senolali/RQEval's architecture:
  * Models -> edit config/config_ehq_20models.yaml [models] section
    (add/remove a model without touching code; add a new PROVIDER by
    writing one models/<x>_model.py + one registry line below)
  * Dataset/categories -> --categories CLI flag or config

Run: python main.py --models all --categories all
Run specific config: python main.py --config config/config_ehq_20models.yaml
Dry run (no API calls, deterministic MockModel): python main.py --dry-run --limit 10

Environment variables (see .env.example):
  ASU_CREATEAI_TOKEN   -> for type: "asu" models (18 models)
  DEEPSEEK_API_KEY     -> for type: "deepseek" models (2 models)
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import yaml

warnings.filterwarnings("ignore", message=".*logits.*model output.*")


def _install_certifi_ssl_fallback() -> None:
    """Use Certifi only when the native Windows CA store is unreadable.

    Ported from senolali/RQEval (main.py) -- directly relevant here too,
    the project's local dev environment is Windows/conda (see
    docs/EHQ_PROJECT_CONTEXT_v2.md). Some Windows/Conda OpenSSL
    combinations raise "ASN1: NOT_ENOUGH_DATA" while loading the Windows
    certificate store. Verification remains enabled: this fallback
    changes only the CA bundle, from the failing native store to
    Certifi's Mozilla CA bundle.
    """
    import ssl

    try:
        ssl.create_default_context()
        return
    except ssl.SSLError as exc:
        if "ASN1: NOT_ENOUGH_DATA" not in str(exc):
            raise

    try:
        import certifi
    except ImportError:
        warnings.warn(
            "The Windows certificate store is unreadable and Certifi is not "
            "installed. Install it with: python -m pip install certifi"
        )
        return

    original_create_default_context = ssl.create_default_context

    def certifi_default_context(purpose=ssl.Purpose.SERVER_AUTH, *,
                                cafile=None, capath=None, cadata=None):
        if cafile is None and capath is None and cadata is None:
            cafile = certifi.where()
        return original_create_default_context(
            purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)

    ssl.create_default_context = certifi_default_context
    ssl._create_default_https_context = certifi_default_context
    warnings.warn(
        "Windows CA store could not be parsed; using Certifi's verified CA "
        "bundle for HTTPS connections."
    )


_install_certifi_ssl_fallback()

# Load .env if present (ASU_CREATEAI_TOKEN, DEEPSEEK_API_KEY -- see .env.example)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base_dir)

from utils.logger import get_logger
from utils.reproducibility import set_seed
from framework import config_scoring

logger = get_logger(__name__)


def _set_log_level(config: dict) -> None:
    import logging
    verbose = config.get("experiment", {}).get("verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    for name in list(logging.root.manager.loggerDict):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        for handler in lg.handlers:
            handler.setLevel(level)


# -------------------------------------------------------------
# Model registry: type string -> builder function
# Adding a new provider: write models/<x>_model.py (extend BaseModel),
# add one line to _get_model_registry() below.
# -------------------------------------------------------------

def _build_asu(name, p, mc, det):
    from models.asu_model import ASUCreateAIModel, DEFAULT_SYSTEM_PROMPT
    key = os.environ.get(p.get("api_key_env", "ASU_CREATEAI_TOKEN"), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env', 'ASU_CREATEAI_TOKEN')}' not set.")
    return ASUCreateAIModel(
        name=name, api_key=key,
        model_name=p["model_name"], model_provider=p["model_provider"],
        config=mc, base_url=p.get("base_url"), deterministic=det,
        temperature=p.get("temperature"),
        max_retries=p.get("max_retries", 5), timeout=p.get("timeout", 120),
        max_tokens=p.get("max_tokens", 512),
        request_delay=p.get("request_delay", 1.0),
        system_prompt=p.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
    )


def _build_deepseek(name, p, mc, det):
    from models.deepseek_model import DeepSeekModel
    key = os.environ.get(p.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not key:
        raise EnvironmentError(f"Env var '{p.get('api_key_env', 'DEEPSEEK_API_KEY')}' not set.")
    return DeepSeekModel(
        name=name, api_key=key, model_id=p["model_id"], config=mc,
        base_url=p.get("base_url"), deterministic=det,
        temperature=p.get("temperature"),
        max_retries=p.get("max_retries", 5), timeout=p.get("timeout", 60),
        max_tokens=p.get("max_tokens", 512),
        request_delay=p.get("request_delay"),
    )


def _build_mock(name, p, mc, det, seed):
    from models.mock_model import MockModel
    return MockModel(name=name, config=mc, seed=seed, deterministic=True)


def _get_model_registry(seed):
    return {
        "asu":      lambda n, p, mc, det, s: _build_asu(n, p, mc, det),
        "deepseek": lambda n, p, mc, det, s: _build_deepseek(n, p, mc, det),
        "mock":     lambda n, p, mc, det, s: _build_mock(n, p, mc, det, s),
    }


def build_models(config: dict, dry_run: bool) -> list:
    """Build every configured model. dry_run=True builds MockModel for ALL
    entries regardless of their configured type -- rest of the pipeline
    (checkpointing, classification, scoring, export) runs unmodified."""
    exp  = config.get("experiment", {})
    seed = exp.get("seed", 42)
    det  = exp.get("deterministic", False)
    registry = _get_model_registry(seed)
    models = []

    for mc in config.get("models", []):
        name  = mc["name"]
        mtype = "mock" if dry_run else mc.get("type", "mock")
        p     = mc.get("params", {})

        if mtype not in registry:
            logger.warning(f"  X Unknown model type '{mtype}' for '{name}'. "
                           f"Available: {list(registry.keys())}")
            continue
        try:
            model = registry[mtype](name, p, mc, det, seed)
            models.append(model)
            logger.info(f"  OK {name:28s} [{mtype}]")
        except EnvironmentError as e:
            logger.warning(f"  ! {name:28s} skipped -- {e}")
        except Exception as e:
            logger.error(f"  X {name:28s} failed  -- {e}")

    return models


# -------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------

def load_dataset(path: str, categories: list, limit: int = None, seed: int = 42) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if categories:
        data = [d for d in data if d["category"] in categories]
    if limit:
        data = list(data)
        random.Random(seed).shuffle(data)
        data = data[:limit]
    return data


def select_models(built_models: list, wanted: str) -> list:
    """Filter already-built BaseModel instances by --models. Filtering
    happens AFTER build_models() (not before) so a typo'd/missing model
    name is reported precisely, distinct from a model that failed to
    build due to a missing API key (see build_models()'s own warnings)."""
    if wanted == "all":
        return built_models
    names = {n.strip() for n in wanted.split(",")}
    missing = names - {m.name for m in built_models}
    if missing:
        raise ValueError(f"model name(s) not found/loaded in config: {sorted(missing)}")
    return [m for m in built_models if m.name in names]


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def print_banner():
    print("""
+==================================================================+
|   EHQ-3000: Epistemic Honesty Quotient Evaluation Pipeline      |
|   Config-driven: models via config/config_ehq_20models.yaml     |
|   Metrics: EHQ1 (Restraint) . EHQ2 (Hallucination Resistance) . |
|            EHQ3 (Calibration) . EHQ (Composite)                 |
+==================================================================+""")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(_base_dir, "config", "config_ehq_20models.yaml"))
    parser.add_argument("--dataset", default=None,
                        help="Default: config's datasets[0].params.path")
    parser.add_argument("--models", default="all",
                        help="Comma-separated list of model names, or 'all'")
    parser.add_argument("--categories", default="all",
                        help="Comma-separated list of categories (FEQ,PCQ,HNQ,CCQ), or 'all'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Caps the total number of questions (for quick tests)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: config's experiment.output_dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test the whole pipeline without real API calls (using MockModel)")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp = config.get("experiment", {})
    set_seed(exp.get("seed", 42))
    _set_log_level(config)

    categories = config_scoring.CATEGORIES if args.categories == "all" \
        else [c.strip() for c in args.categories.split(",")]
    unknown = set(categories) - set(config_scoring.CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categor(y/ies): {sorted(unknown)} "
                         f"(valid: {config_scoring.CATEGORIES})")

    dataset_path = args.dataset or config["datasets"][0]["params"]["path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(_base_dir, dataset_path)
    dataset = load_dataset(dataset_path, categories, limit=args.limit, seed=exp.get("seed", 42))
    if not dataset:
        raise ValueError(f"Dataset is empty: {dataset_path} (categories: {categories})")

    logger.info("Registering models...")
    models = build_models(config, dry_run=args.dry_run)
    models = select_models(models, args.models)
    if not models:
        logger.error(
            "\nNo models loaded/selected. Check:\n"
            "  * API key env vars are set (or a .env file exists, see .env.example)\n"
            "  * --models names match config/config_ehq_20models.yaml\n"
            "  * pip install -r requirements.txt\n"
        )
        sys.exit(1)
    logger.info(f"\n{len(models)} model(s) ready.\n")

    output_dir = args.output_dir or os.path.join(_base_dir, exp.get("output_dir", "outputs_ehq"))
    experiment_id = f"{exp.get('name', 'ehq_experiment')}_{int(time.time())}"

    logger.info(f"Config        : {args.config}")
    logger.info(f"Dataset       : {dataset_path} ({len(dataset)} items, {','.join(categories)})")
    logger.info(f"Experiment ID : {experiment_id}")
    logger.info(f"Output dir    : {output_dir}\n")
    if args.dry_run:
        logger.info("DRY RUN: MockModel in use, no real API calls will be made.\n")

    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(config=config, categories=categories,
                          output_dir=output_dir, experiment_id=experiment_id)
    results = evaluator.evaluate_all(models=models, dataset=dataset)

    print("\n" + "=" * 88)
    print(f" {'Model':<28} {'N':>6} {'EHQ1':>7} {'EHQ2':>7} {'EHQ3':>7} {'EHQ':>7}")
    print("-" * 88)
    for r in sorted(results, key=lambda x: x.get("EHQ", 0), reverse=True):
        print(f" {r['model']:<28} {r['n_questions']:>6} "
              f"{r['EHQ1']:>7.3f} {r['EHQ2']:>7.3f} {r['EHQ3']:>7.3f} {r['EHQ']:>7.3f}")
    print("=" * 88)
    print(f"  Results -> {evaluator.tracker.exp_dir}\n")

    return results


if __name__ == "__main__":
    main()
