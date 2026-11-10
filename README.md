# Spatial Transcriptomics Integration Audit

STIA evaluates biological signal retention, batch correction, spatial structure, and sensitivity to input order and repeated execution. It also supports targeted source checks, downstream re-evaluation and fault diagnosis. This repository contains the frozen STIA 1.3.8 implementation, an installable wheel, a synthetic demonstration and the records needed to verify the paper's principal quantitative results. The Python distribution is named `invest-st`, and the main command is `stia`.

## Install and run

The commands below use Python 3.12 and run from the repository root in a fresh environment.

```sh
python -m pip install -r requirements-review-py312.txt
python -m pip install --no-deps dist/invest_st-1.3.8-py3-none-any.whl
python -m stia --version
python -m stia audit --case-root examples/synthetic_audit --config examples/synthetic_audit/config.json --output demo_report
python -m stia diagnose-method --help
```

The synthetic audit example contains 48 spots from 12 simulated donors. A relative audit output path resolves inside the case directory, and the example writes its decision to `examples/synthetic_audit/demo_report/decision.json`. To generate a fresh case, use `python -m stia demo --output demo_case`. Its expected decision is ABSTAIN (exit code 1) because the demo provides insufficient evidence for a required biological-signal check. Use fresh output directories for repeated runs. To install from source, run `python -m pip install ./source_snapshot`. The command help and [software documentation](source_snapshot/README_SUBMISSION.md) describe settings and report fields.

## Audit your own inputs

Start with [the synthetic configuration](examples/synthetic_audit/config.json).
The audit configuration points to a metadata table, a numeric representation
and the corresponding vector of stable spot identifiers. Metadata specifies
the spot, sample and donor keys, the condition label and applicable batch
labels. Add spatial coordinates, tissue annotations, repeat arms and order arms
as required by the comparison. Reported performance axes depend on the
available design and evidence. Keep the original representations and input
receipts when preparing an audit.

The `diagnose-method` command accepts either an `--input` NPZ with a dense
matrix `X`, unique `spot_ids` and contiguous `group_ids`, or a
`--results` manifest for existing outputs. With a fresh input, the default
run produces a plan. The `--execute` flag runs the configured backend.
The built-in PCA backend supports `fixed_settings.n_components`. Other
fitting backends require a method adapter. Use `--config` to supply
technical seeds, input orders, adapter settings or a control recipe, and
`--output` to select a new result directory. See `python -m stia --help`
for the available commands.

## Verify the paper results

```sh
python -I verify_submission.py
python -I numerical_stress_768/verify_stress.py
python -I verify_latest.py
```

The first two verifiers use the Python standard library. The third uses NumPy and SciPy from the pinned environment.

| Record | Scope |
| --- | --- |
| `case_records/` | Expected and observed decisions and fault sets for 600 controlled and 208 held-out Xenium cases |
| `numerical_stress_768/` | Prespecified labels and workflow predictions for 768 numerical stress cases |
| `stia138_880_replay/` | Decision and ablation records for 880 cases from 160 parent templates |
| `verified_20261106/` | Frozen slide-level and donor-level evaluation records for the corrected GSE307403 analysis and paired inference |
| `figure_inputs/` | Source values for selected performance and intervention figures |
| `simulation_sidecars/` | Generation parameters and input fingerprints for the 90 simulated datasets |

[PUBLIC_DATA_INDEX.csv](PUBLIC_DATA_INDEX.csv) identifies the original public datasets and accessions. Re-fitting third-party integration methods requires retrieving the original inputs and setting up their upstream environments. The verifiers above recompute results from the included records. Simulation and method configuration indices identify the corresponding designs. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) gives hashes for the released wheel and its modules, and [MANIFEST.json](MANIFEST.json) gives hashes for the public files.

The STIA software is distributed under the [MIT license](LICENSE). Original public datasets are governed by their source repositories and publications.
