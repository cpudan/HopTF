# HopTF Handoff

Date: 2026-05-04
Branch: `peter-dan`

## Current State

- Full real-key smoke pipeline passes: `14 / 14`.
- Test suite passes: `14 passed`.
- Latest status report: `tmp/hopfield_fitting_yolo_status_report.md`.
- Latest full runner report: `tmp/hopfield_fitting_yolo_full_rerun/overnight_summary.md`.
- Latest calibrated panel report: `tmp/hopfield_fitting_leave_one_panel_3seed_calibrated/otcfm_leave_one_panel.md`.
- Recovery archive: `tmp/hopfield_fitting_yolo_full_rerun/recovery_latest.tgz`.

## Commits To Review

- `5c0e576` Add Hopfield CFM smoke pipeline
- `9b99e47` Use standardized endpoint metric for isoform holdouts
- `811728b` Add mixed-response OT-CFM panel
- `98f5335` Add seed-aware leave-one panel criteria
- `d0f0181` Add calibrated HopTF execution reporting

## Key Results

- Real AlphaGenome gene-pooled key path is wired into the runner.
- Hopfield projection smoke tests pass on synthetic and real-key overfit checks.
- TorchCFM synthetic, real overfit, HNF4A leave-one, and TP53 leave-one checks pass.
- Calibrated mixed-response panel has `30 / 42` seed-level label-aware passes and `8 / 14` stable isoforms.
- Controlled baselines show artifact plus Hopfield-query features improve over artifact-only in matched contexts.
- Mutant sequence coordinate checks verify HNF4A and GATA2 candidates, while TP53 hotspot mapping is skipped due to isoform-coordinate mismatch.
- Mutant endpoint predictions are scaffolded and produce a structured blocker report when mutant ESM-C embeddings are absent.

## Workstreams

### 1. Model Calibration

Goal: improve the calibrated mixed-response panel beyond `8 / 14` stable isoforms.

Start with:

- `scripts/run_otcfm_leave_one_panel.py`
- `scripts/train_otcfm_sequence_conditioned.py`
- `tmp/hopfield_fitting_leave_one_panel_3seed_calibrated/otcfm_leave_one_panel.md`

Focus cases:

- `IKZF3-5`: hard responder endpoint failure.
- `SOX5-1`: unsupported nonresponder holdout with no same-gene nonresponder sibling support.
- `TP73-2`: seed-sensitive nonresponder after calibration.
- `ZNF195-3`: seed-sensitive under-transported responder.

### 2. Mutant ESM-C Embeddings

Goal: generate mutant embeddings and run frozen endpoint predictions.

Needed files:

- `tmp/hopfield_fitting_yolo_full_rerun/mutant_esmc_embeddings.npy`
- `tmp/hopfield_fitting_yolo_full_rerun/mutant_esmc_vocab.json`

Then run:

```bash
uv run python scripts/evaluate_mutant_endpoint_predictions.py \
  --outdir tmp/hopfield_fitting_yolo_full_rerun \
  --mutant-metadata tmp/hopfield_fitting_yolo_full_rerun/mutant_sequences_metadata.csv \
  --mutant-embedding-matrix tmp/hopfield_fitting_yolo_full_rerun/mutant_esmc_embeddings.npy \
  --mutant-vocab tmp/hopfield_fitting_yolo_full_rerun/mutant_esmc_vocab.json \
  --latents tmp/hopfield_fitting_yolo_full_rerun/perturbation_latents.npz \
  --checkpoint tmp/hopfield_fitting_yolo_full_rerun/otcfm_real_overfit.pt
```

### 3. Endpoint Expansion

Goal: move beyond perturbation-level PCA centroids.

Start with:

- `scripts/build_perturbation_latents.py`
- `scripts/add_scarf_embeddings.py`

Target additions:

- SCARF endpoint extraction.
- Cell-level mini-batch training/evaluation.
- DE/signature-agreement metrics where pseudobulk counts are available.

## Reproduction Commands

```bash
uv run pytest -q
uv run python scripts/run_hopfield_overnight.py --outdir tmp/hopfield_fitting_yolo_full_rerun --key-source auto --holdout-gene HNF4A --holdout-gene TP53 --max-holdouts-per-gene 1
uv run python scripts/run_otcfm_leave_one_panel.py --outdir tmp/hopfield_fitting_leave_one_panel_3seed_calibrated --latents tmp/hopfield_fitting_metric_refactor_full/perturbation_latents.npz --steps 700 --max-train-rows 512 --min-cells 5 --n-seeds 3 --aggregate-pass-rate 0.67 --nonresponder-response-ratio-threshold 1.5 --endpoint-loss-weight 0.1 --response-amplitude-loss-weight 0.01 --endpoint-loss-steps 8 --endpoint-loss-interval 10
uv run python scripts/compile_hoptf_status_report.py --overnight-dir tmp/hopfield_fitting_yolo_full_rerun --panel-dir tmp/hopfield_fitting_leave_one_panel_3seed_calibrated --out tmp/hopfield_fitting_yolo_status_report.md
```
