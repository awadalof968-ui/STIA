# -*- coding: utf-8 -*-
"""Stable Python API re-export."""
from .method_diagnostics_v1 import method_recipes, diagnose_method, diagnose_saved_method, summarize_embeddings
from .baselines import BASELINE_SURFACE, baseline_cells, baseline_comparisons, framework_quality_metrics
from .contracts import BASELINES, BASELINE_IDS, PRIMARY_METRIC_IDS
from .engine import AuditRun, audit
from .regression import (REGRESSION_CORPUS, corpus_records, execute_regression_case,
                          execute_regression_corpus, validate_corpus,
                          validate_regression_record)
from .io import validate_canonical_bundle
from .splits import SplitPlan, grouped_holdout_split, grouped_nested_cv_split, write_split_plan
from .perturbations import PERTURBATION_AXES, PERTURBATION_SEMANTICS, REQUIRED_FIXED_FIELDS, perturbation_contract
from .ablation import (
    ABLATION_METRIC_IDS,
    ABLATION_PROTOCOL_VERSION,
    ABLATION_RESULT_VERSION,
    PROTOCOL_SCHEMA_VERSION as ABLATION_PROTOCOL_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION as ABLATION_RESULT_SCHEMA_VERSION,
    ablation_variants,
    run_targeted_ablations,
    score_targeted_ablations,
    validate_ablation_protocol,
    validate_protocol,
)
from .cross_ai import (
    CROSS_AI_BUNDLE_FILES,
    CROSS_AI_PROTOCOL_VERSION,
    CrossAIValidationError,
    agreement_summary,
    compute_agreement,
    compute_agreement_metrics,
    disagreement_inventory,
    gold_scoring,
    ingest_answer_manifest,
    ingest_cross_ai,
    run_cross_ai,
    score_cross_ai,
    score_cross_ai_answers,
    score,
    summarize_agreement,
    validate,
    validate_cross_ai,
    validate_cross_ai_input,
    write_cross_ai_bundle,
)

__all__ = [
    "method_recipes", "diagnose_method", "diagnose_saved_method", "summarize_embeddings",
    "AuditRun", "audit", "BASELINE_SURFACE", "BASELINES", "BASELINE_IDS", "PRIMARY_METRIC_IDS", "baseline_cells", "baseline_comparisons",
    "framework_quality_metrics", "REGRESSION_CORPUS", "corpus_records", "execute_regression_case",
    "execute_regression_corpus", "validate_corpus",
    "validate_regression_record", "validate_canonical_bundle", "SplitPlan", "grouped_holdout_split",
    "grouped_nested_cv_split", "write_split_plan",
    "PERTURBATION_AXES", "PERTURBATION_SEMANTICS", "REQUIRED_FIXED_FIELDS", "perturbation_contract",
    "ABLATION_PROTOCOL_SCHEMA_VERSION", "ABLATION_RESULT_SCHEMA_VERSION", "ablation_variants",
    "ABLATION_PROTOCOL_VERSION", "ABLATION_RESULT_VERSION", "ABLATION_METRIC_IDS",
    "validate_ablation_protocol", "validate_protocol", "score_targeted_ablations", "run_targeted_ablations",
    "CROSS_AI_PROTOCOL_VERSION", "CROSS_AI_BUNDLE_FILES", "CrossAIValidationError",
    "validate_cross_ai_input", "ingest_cross_ai", "agreement_summary", "disagreement_inventory",
    "compute_agreement", "compute_agreement_metrics", "summarize_agreement", "validate_cross_ai", "validate", "gold_scoring", "ingest_answer_manifest", "run_cross_ai", "score_cross_ai", "score_cross_ai_answers", "score", "write_cross_ai_bundle",
]
