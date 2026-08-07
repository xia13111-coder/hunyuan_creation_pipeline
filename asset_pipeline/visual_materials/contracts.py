"""Small cross-runtime constants with no material-model dependencies."""

RESULT_SCHEMA_VERSION = "asset-pipeline-visual-material-result/v2"
RESTORED_HISTORICAL_BASELINE = "RESTORED_HISTORICAL_BASELINE"
SEALED_BASELINE_EVIDENCE_SCHEMA = "qwen-bundled-project-evidence/v1"
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
VISUAL_INFERENCE_MODES = frozenset({"live", "auto", "bundled"})

ISOLATED_ENV_REMOVE = (
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
)


__all__ = [
    "ISOLATED_ENV_REMOVE",
    "RESTORED_HISTORICAL_BASELINE",
    "RESULT_SCHEMA_VERSION",
    "SEALED_BASELINE_EVIDENCE_SCHEMA",
    "USD_SUFFIXES",
    "VISUAL_INFERENCE_MODES",
]
