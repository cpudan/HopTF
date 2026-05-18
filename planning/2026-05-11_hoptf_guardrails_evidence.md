# HopTF Guardrails And Baseline-Control Evidence

Source bundle: `tmp/hoptf_guardrails_may18/`

Additional remote result: `laplace` mutant-transport bundle
`tmp/laplace_returns/hoptf_mutant_transport_laplace_20260511_140909.tgz`
and extracted files under
`tmp/laplace_returns/hoptf_mutant_transport_laplace_20260511_140909/`.

Detailed natural-isoform failure audit:
`planning/2026-05-11_natural_isoform_failure_audit.md`.

This note summarizes the guardrail evidence received from the other machine for the Monday, May 18 report. The transferred bundle contains report-friendly artifacts and manifests, not large arrays/checkpoints. Exact original artifact paths are preserved for traceability.

## Baseline And Control Table

| Check | Evidence path | Key result | Interpretation for report | Caveat |
| --- | --- | --- | --- | --- |
| Artifact-only endpoint baseline | `tmp/hopfield_fitting_yolo_full_rerun/controlled_evaluation_report.md`; `controlled_sequence_baselines.csv`; `controlled_sequence_baselines_summary.json` | Artifact-only grouped MSE `6.594611`; length-matched MSE `6.565562`; cell-count-matched MSE `6.614219`; length+cell matched MSE `6.578882`. | This is the main negative-control baseline for endpoint prediction using protein length, ORF length, and recovered-cell-count artifacts. | Ridge model over perturbation-level PCA centroid endpoints; not cell-level or SCARF. |
| Artifact + HopTF-query endpoint baseline | `controlled_evaluation_report.md`; `controlled_sequence_baselines.csv` | `artifacts_plus_hopfield_query` MSE `6.272790`, improving over artifact-only by `-0.321821` / `-4.88%`; improvement persists in length-matched `-4.32%`, cell-count-matched `-2.73%`, and length+cell matched `-1.69%`. | Supports the bounded claim that HopTF-query features add endpoint-prediction signal beyond simple artifact covariates in this benchmark. | The effect is modest and still endpoint-benchmark evidence, not final biology. |
| Artifact + raw ESM-C endpoint baseline | `controlled_evaluation_report.md`; `controlled_sequence_baselines.csv` | `artifacts_plus_esm_c` MSE `6.889832`, worse than artifact-only by `+4.48%`; also worse in length-matched `+13.37%`, cell-count-matched `+13.77%`, and length+cell matched `+19.48%`. | Supports the claim that the Hopfield-query projection is doing something nontrivial relative to simply adding raw ESM-C vectors to artifact covariates. | A high-dimensional raw ESM-C ridge baseline may be underfit; do not claim raw ESM-C has no biological information. |
| Sequence endpoint baselines | `sequence_endpoint_baselines.md`; `sequence_endpoint_baselines.csv`; `sequence_endpoint_baselines_summary.json` | Grouped 5-fold endpoint MSE: `length_cell_orf` `6.594611`; `length_plus_cell_count` `6.596982`; `length_only` `6.632317`; `hopfield_query` `6.868612`; `cell_count_only` `7.252584`; `esm_c` `7.275968`. | Artifact covariates are strong endpoint baselines and must be shown next to sequence models. | Same PCA-centroid endpoint and ridge setup. |
| Natural same-gene isoform panel | `tmp/hopfield_fitting_leave_one_panel_3seed_calibrated/otcfm_leave_one_panel.md`; aggregate/metrics/runs/summary CSV+JSON files; detailed audit in `planning/2026-05-11_natural_isoform_failure_audit.md` | Calibrated panel: endpoint criterion `24 / 42` seed-level passes; label-aware criterion `30 / 42`; stable label-aware isoforms `8 / 14`. Failure categories: `8` stable pass, `4` seed-sensitive, `1` responder endpoint failure, `1` unsupported nonresponder sibling. | Shows there is a real same-gene isoform benchmark with named failure modes; useful context for interpreting mutant predictions. | `IKZF3-5` is the clearest fair responder failure. `SOX5-1` is not a clean nonresponder failure because it lacks same-gene nonresponder sibling support. `TP73-1`/`TP73-2` are borderline two-of-three pass cases near the configured stability threshold. |
| Response artifact QC | `tf_length_response_qc_outputs/tf_length_response_summary.csv`; PDF report; downsampled/null/model-coefficient outputs | Spearman correlations: protein length vs response score `0.6501`; protein length vs `n_cells` `-0.7402`; response score vs `n_cells` `-0.9022`; corrected length-response rho `0.2411`; downsampled rho `0.0668`; conclusion: “Partially coverage-driven; residual length effect remains.” | Strong evidence that length and recovered-cell-count artifacts are real and need explicit controls. | Generated local analysis; use as artifact-control support, with provenance noted if needed. |
| Endpoint representation | `perturbation_latents_report.json`; metadata CSV; latent NPZ listed in manifest | PCA endpoint report: latent shape `[3266, 50]`; `3266` matched perturbations; control mean/sd shape `[50]`; source h5ad `GSE217460_210322_TFAtlas_subsample_raw_csr.h5ad`. | Documents exactly what endpoint target current smoke/baseline results use. | Current endpoint is perturbation-level PCA centroid only. No completed SCARF/cell-level endpoint report found. |
| Mutant endpoint predictions | `laplace:tmp/hopfield_fitting_yolo_full_rerun/mutant_endpoint_predictions_report.json`; local return bundle `tmp/laplace_returns/hoptf_mutant_transport_laplace_20260511_140909.tgz`; extracted local `mutant_endpoint_predictions.csv` | Strict evaluator status `ok`; `6 / 6` verified mutants predicted. HNF4A predicted response deltas: `R85W` `-165.66`, `S87N` `-8.49`, `R89W` `-22.76`, `T139I` `+21.30`. GATA2 predicted response deltas: `R396Q` `-33.31`, `R362Q` `-140.98`. | Supports the bounded claim that the frozen HopTF endpoint model is sequence-sensitive to curated mutant substitutions, with HNF4A DNA-binding mutations moving in the expected weaker-response direction while the HNF4A weak/benign comparison does not. | Current endpoint is PCA-centroid transport through a frozen smoke-model checkpoint. GATA2 `R362Q` should be interpreted cautiously. TP53 remains excluded because local isoform coordinates do not match canonical mutation coordinates. |
| AlphaGenome key provenance | `alphagenome_keys_report.json`; metadata CSV; planning notes | Prepared key matrix shape `[54901, 3072]` from `gene_pooled_npz`; metadata includes gene IDs/symbols/types, chroms, starts, ends, strands, and sequence lengths. | Supports use of real gene-pooled AlphaGenome-style keys rather than synthetic keys. | Genome build is not verified by the local report. Do not claim GRCh38/hg38 unless provenance is confirmed. |
| Full smoke runner | `overnight_summary.md`; `overnight_run_summary.json` | Full smoke runner completed `14 / 14` steps with key source `gene_pooled_npz`. | Supports that the end-to-end pipeline produced the evidence artifacts. | Smoke success includes a structured blocked mutant endpoint step, not successful mutant predictions. |
| Linear-probe background | `mvp/linear_probe_report/RESULTS.md`; summary JSON files | ESM-C is above shuffled-label null; protein length is a strong baseline; ESM-C remains above chance within protein-length quartiles. | Useful background that sequence contains response-label signal while length artifacts remain important. | Classification-label benchmark, not endpoint transport. Keep as secondary context. |
| Response DE/signature artifacts | response-extreme DE/GO, limma-voom, and TF signature score outputs | Exploratory response-extreme and signature analyses exist; batch-1 response/control limma-voom reports no FDR-significant terms in high/low/control contrasts. | Optional biological-context support if needed. | Exploratory; not part of the core HopTF guardrail pipeline. Use cautiously. |

## Safe Report Claims

- In the controlled PCA-endpoint benchmark, HopTF-query features improve modestly over artifact-only covariates, including in length-matched and recovered-cell-count-matched contexts.
- Raw ESM-C added to artifact covariates does not improve this same controlled endpoint benchmark, so the Hopfield-query representation appears more useful than a naive raw-embedding baseline in this setup.
- Protein length, recovered cell count, and response score are strongly correlated; these artifacts must be reported and controlled rather than treated as nuisance details.
- The natural same-gene isoform panel is useful but not fully stable: `8 / 14` isoforms are stable label-aware passes, with one clear fair responder failure (`IKZF3-5`), one unsupported nonresponder case (`SOX5-1`), and several threshold/seed-sensitive cases.
- Current endpoint evidence is based on perturbation-level PCA centroids. Strong biological endpoint claims require either careful caveating or a stronger endpoint representation.
- Mutant endpoint prediction is now unblocked for six verified HNF4A/GATA2 variants. In this frozen PCA-endpoint model, the HNF4A DNA-binding-disrupting mutations reduce predicted response relative to WT, while the HNF4A weak/benign comparison `T139I` slightly increases predicted response.
- AlphaGenome key shape/source is documented, but genome-build provenance is not verified.

## Claims To Avoid

- Do not claim HopTF is validated as a general predictive biological model.
- Do not claim mutant transport is biologically validated from this result alone; report it as a frozen-model sequence-sensitivity test over PCA centroid endpoints.
- Do not claim attention weights are binding maps unless motif enrichment supports that interpretation.
- Do not claim GRCh38/hg38 AlphaGenome provenance from the current local key report.
- Do not treat recovered cell count as a simple confounder that can be regressed out everywhere; it may be downstream of both biology and technical recovery.

## Missing Or Still Needed

- Completed beta-sensitivity diagnostics.
- Completed motif-enrichment diagnostics.
- Completed SCARF or cell-level endpoint comparison, if we decide to use the extra time for endpoint upgrade.
- Provenance check for AlphaGenome genome build.
