# -*- coding: utf-8 -*-
"""STIA — Spatial Transcriptomics Integration Audit.

This staged package extends the existing ``invest_st`` checks behind one
versioned, read-only M1-M9 interface.
"""
from .contracts import BASELINES, BASELINE_IDS, MODULES, PRIMARY_METRIC_IDS, SCHEMA_VERSION, TOOL_VERSION
from .baselines import BASELINE_SURFACE, baseline_cells, baseline_comparisons, framework_quality_metrics
from .engine import AuditRun, audit
from .diagnostics import diagnose
from .regression import (REGRESSION_CORPUS, corpus_records, execute_regression_case,
                          execute_regression_corpus, validate_corpus,
                          validate_regression_record)
from .io import validate_canonical_bundle
from .splits import SplitPlan, grouped_holdout_split, grouped_nested_cv_split, write_split_plan
from .perturbations import PERTURBATION_AXES, PERTURBATION_SEMANTICS, REQUIRED_FIXED_FIELDS, perturbation_contract
from .order_repair import ORDER_REPAIR_AXES, ORDER_REPAIR_VERSION, CanonicalOrderPlan, build_canonical_order_plan
from .staligner_guard import fit_group_order_guard
from .fault_injection import FAULT_INJECTION_VERSION, inject_row_identity_mismatch, interpolate_order_drift, shrink_representation_variance
from .ablation_v2 import complete_ablation_variants, evaluate_complete_ablations
from .ablation_cases import build_executable_ablation_cases, run_executable_ablation_benchmark
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

__version__ = TOOL_VERSION
__all__ = [
    "diagnose",
    "fit_group_order_guard",
    "AuditRun", "MODULES", "SCHEMA_VERSION", "TOOL_VERSION", "BASELINE_SURFACE", "BASELINES", "BASELINE_IDS", "PRIMARY_METRIC_IDS",
    "baseline_cells", "baseline_comparisons", "framework_quality_metrics",
    "REGRESSION_CORPUS", "corpus_records", "execute_regression_case", "execute_regression_corpus",
    "validate_corpus", "validate_regression_record", "validate_canonical_bundle", "audit",
    "SplitPlan", "grouped_holdout_split", "grouped_nested_cv_split", "write_split_plan",
    "PERTURBATION_AXES", "PERTURBATION_SEMANTICS", "REQUIRED_FIXED_FIELDS", "perturbation_contract",
    "ORDER_REPAIR_AXES", "ORDER_REPAIR_VERSION", "CanonicalOrderPlan", "build_canonical_order_plan",
    "FAULT_INJECTION_VERSION", "inject_row_identity_mismatch", "interpolate_order_drift", "shrink_representation_variance",
    "complete_ablation_variants", "evaluate_complete_ablations",
    "build_executable_ablation_cases", "run_executable_ablation_benchmark",
    "ABLATION_PROTOCOL_SCHEMA_VERSION", "ABLATION_RESULT_SCHEMA_VERSION", "ablation_variants",
    "ABLATION_PROTOCOL_VERSION", "ABLATION_RESULT_VERSION", "ABLATION_METRIC_IDS",
    "validate_ablation_protocol", "validate_protocol", "score_targeted_ablations", "run_targeted_ablations",
    "CROSS_AI_PROTOCOL_VERSION", "CROSS_AI_BUNDLE_FILES", "CrossAIValidationError",
    "validate_cross_ai_input", "ingest_cross_ai", "agreement_summary", "disagreement_inventory",
    "compute_agreement", "compute_agreement_metrics", "summarize_agreement", "validate_cross_ai", "validate", "gold_scoring", "ingest_answer_manifest", "run_cross_ai", "score_cross_ai", "score_cross_ai_answers", "score", "write_cross_ai_bundle",
]
