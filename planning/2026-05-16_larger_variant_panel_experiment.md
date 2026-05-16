# Larger HopTF Variant Panel Experiment

Date: 2026-05-16

## Purpose

The first ClinVar mutation-validation run produced a clean but underpowered result:

- 24 benign / likely benign variants
- 24 pathogenic / likely pathogenic variants
- all coordinate-checked against local responder isoforms
- endpoint evaluator status: `ok`, `48 / 48`
- median percent delta:
  - benign: `-0.38%`
  - deleterious: `-1.82%`
- AUROC for deleterious using `-percent_delta`: `0.578`

The result is directionally consistent but not strong enough to call statistically or biologically convincing. The next experiment should test whether the weak signal is mostly a power problem, a panel-curation problem, or an endpoint/model limitation.

## Main Question

Do high-confidence deleterious missense variants in TF responder isoforms reduce predicted HopTF transport response more than high-confidence benign variants, after controlling for gene, domain context, residue position, and ESM-C sequence-distance magnitude?

## Design Overview

Run three nested panels, not one broad panel.

### Panel A: Expanded All-ClinVar Panel

Goal: increase power while keeping the same curation logic as the 48-variant run.

Paper role:

Panel A is the first, broadest test for the paper's sequence-sensitivity claim. It fills the gap between the clean but underpowered 48-variant pilot and a reportable statement about whether HopTF endpoint predictions respond systematically to TF missense variation. It should support the mutation Results subsection in `paper/HopTF_full_draft.tex`, provisionally titled:

> Sequence variants perturb retrieval and endpoint predictions

If only Panel A endpoint outputs are available, use narrower wording:

> Expanded ClinVar variant panel tests endpoint sequence sensitivity

Question answered:

Do deleterious ClinVar TF missense variants produce more negative predicted response shifts than benign ClinVar TF missense variants under the current frozen PCA-centroid HopTF endpoint model?

Gap filled:

The pilot panel had only 24 benign and 24 deleterious variants, with directional but weak separation (`AUROC = 0.578`). Panel A tests whether that weak signal is mostly a power/yield problem or whether broad ClinVar pathogenicity is too heterogeneous for this endpoint target.

Target:

- 100-300 benign / likely benign variants
- 100-300 pathogenic / likely pathogenic variants
- minimum useful yield: at least 100 variants per class, or a clear documented reason why strict coordinate validation yields fewer
- all variants coordinate-checked against local responder isoforms
- cap at 3-5 variants per gene per class for the primary balanced analysis
- retain an uncapped locked table for sensitivity analysis and provenance

Primary analysis set:

- all valid ClinVar missense variants passing the inclusion/rejection criteria below
- local HopTF isoform must be a responder with `n_cells >= 5`
- protein coordinate must match the local isoform residue exactly
- one row per variant/isoform pair after deduplication
- if multiple ClinVar records collapse to the same gene, residue, alternate amino acid, and isoform, keep the strongest-review record and record merged accessions/IDs in a provenance field

Required row-level metadata:

- stable variant ID, ClinVar variation/accession IDs when available, `gene_symbol`, `isoform_embedding_id`, `mutant_embedding_id`
- clinical class normalized to `benign` or `deleterious`
- original ClinVar clinical significance, review status, submitter count if available, and source assembly
- protein mutation string, WT residue, 1-based position, mutant residue, protein position fraction
- local sequence length, local WT residue check result, coordinate-check status and rejection reason if rejected
- `n_cells`, `label_status`, WT predicted response magnitude after endpoint prediction
- optional but desired: TF family, broad domain label, DNA-binding-domain indicator, and domain-source provenance
- ESM-C WT-mutant distance fields once embeddings are available

Retrieval-delta hooks:

Panel A remains endpoint-centered, but its locked tables must be joinable to later retrieval-delta analysis. Preserve `isoform_embedding_id`, `mutant_embedding_id`, variant ID, gene symbol, protein coordinate, and clinical class in every downstream output. These fields are required to join Panel A endpoint results to future `delta_q`, `delta_attention`, and `delta_mu_p` tables.

Required Panel A artifacts:

- `variant_panel_locked_all_clinvar.csv`: primary locked Panel A table after coordinate validation and balancing flags
- `variant_panel_locked_all_clinvar_uncapped.csv`: optional uncapped valid set before per-gene caps, if enough records exist
- `variant_panel_curation_report.json`: candidate counts, rejection counts by reason, final counts by class, genes, TF families, review status, and coordinate-check outcomes
- `variant_panel_endpoint_predictions_annotated.csv`: endpoint predictions joined to all Panel A metadata
- `variant_panel_validation_summary.json`: primary statistics, bootstrap confidence intervals, calibration checks, and failure-mode flags
- `variant_panel_validation_report.md`: report-ready summary with paper-safe interpretation

Primary endpoint:

```text
percent_delta = 100 * mutant_minus_wt_predicted_response_l2 / wt_predicted_response_l2
```

Negative values mean the mutant has weaker predicted response than WT.

Primary statistics:

- Mann-Whitney U test comparing `percent_delta` for benign vs deleterious variants
- rank-biserial effect size or Cliff's delta
- AUROC for deleterious classification using `-percent_delta`
- bootstrap confidence intervals for benign median, deleterious median, and median class difference
- cluster bootstrap by `gene_symbol` if variants cluster heavily in a few genes

Calibration and robustness checks:

- fraction of benign variants with `abs(percent_delta) <= 5%`
- fraction of deleterious variants with `percent_delta < 0`
- class separation after applying the 3-5 variants per gene per class cap
- class separation in the uncapped valid set, if produced
- `percent_delta` versus ESM-C WT-mutant distance, to check whether the signal is just embedding-distance magnitude
- class separation stratified by review strength, WT response magnitude, `n_cells`, protein position fraction, and optional domain indicator

Paper outputs:

- Main mutation table: Panel A counts, median `percent_delta` by class, class difference, AUROC, p-value, and bootstrap confidence interval.
- Main mutation figure: violin/box/strip plot of `percent_delta` by clinical class, with points colored or faceted by gene or TF family if readable.
- Secondary figure or supplement: ESM-C WT-mutant distance versus `percent_delta`, plus per-gene class deltas.
- Results paragraph: state whether the expanded panel strengthens, weakens, or fails to improve the 48-variant pilot trend.

Expected limitation:

ClinVar pathogenicity may reflect disease mechanisms unrelated to TF DNA binding or transcriptional response amplitude.

Interpretation rules:

- If Panel A succeeds, claim only that the current frozen endpoint model is sequence-sensitive to broad ClinVar missense variation.
- Do not claim biological mutation-effect validation from Panel A alone.
- If Panel A fails but Panel B or Panel C succeeds, report that broad ClinVar pathogenicity is too heterogeneous and that functional/domain-matched curation is required.
- If Panel A fails and later panels also fail, interpret this as evidence that the current endpoint checkpoint or PCA-centroid target may be insufficiently mutation-sensitive.

### Panel B: DNA-Binding-Domain-Enriched Panel

Goal: increase biological specificity.

Target:

- variants located in DNA-binding domains or immediately adjacent functional residues
- benign controls from the same gene/domain whenever possible
- deleterious variants prioritized if literature or ClinVar notes support altered DNA binding, transcriptional activity, or TF syndrome mechanism

Potential domain sources:

- UniProt feature annotations
- Pfam/InterPro domains
- local sequence-domain annotation if already available
- curated TF domain definitions for common families: zinc fingers, bHLH, bZIP, homeobox, forkhead, ETS, nuclear receptor DBD, GATA zinc finger, HMG box

Use this panel to ask whether the model is sensitive to functional TF-domain perturbations rather than generic clinical pathogenicity.

### Panel C: Same-Gene Matched Panel

Goal: reduce gene-level calibration noise.

For each gene, select matched sets where possible:

- one or more deleterious variants
- one or more benign variants
- same local responder isoform if possible
- similar protein region/domain if possible
- similar ESM-C L2 distance from WT if possible

Use this panel for paired or mixed-effects statistics.

## Variant Inclusion Criteria

Primary source:

`https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`

Required:

- `Assembly == GRCh38`
- `Type == single nucleotide variant`
- parseable missense protein substitution such as `p.Arg123Trp`
- clinical class is one of:
  - deleterious: `Pathogenic`, `Likely pathogenic`, `Pathogenic/Likely pathogenic`
  - benign: `Benign`, `Likely benign`, `Benign/Likely benign`
- local HopTF isoform has:
  - `label_status == responder`
  - `n_cells >= 5`
  - sequence residue at 1-based protein coordinate matches the ClinVar WT residue

Reject:

- VUS
- conflicting interpretations
- no assertion / no classification / not provided
- risk alleles
- pharmacogenomic labels
- protective labels
- association-only labels
- truncations, frameshifts, splice variants, synonymous variants
- coordinate mismatches
- variants where the ClinVar protein coordinate cannot be interpreted unambiguously

Prioritize:

- reviewed by expert panel
- practice guideline
- multiple submitters, no conflicts
- higher submitter count
- stronger local responder support by `n_cells`
- variants in TF functional domains for Panel B
- same-gene benign/deleterious matches for Panel C

## Generated Artifacts

Use a new output directory:

```bash
OUT=tmp/hoptf_variant_validation_expanded_20260516
```

Required curation outputs:

- `variant_panel_candidates.csv`
- `variant_panel_mapped_all.csv`
- `variant_panel_locked_all_clinvar.csv`
- `variant_panel_locked_dbd_enriched.csv`
- `variant_panel_locked_same_gene_matched.csv`
- `variant_panel_mutant_sequences_metadata.csv`
- `variant_panel_mutant_sequences.fasta`
- `variant_panel_mutant_esmc_vocab_input.json`
- `variant_panel_mutant_esmc_sequences.json`
- `variant_panel_curation_report.json`

Required prediction outputs:

- `variant_panel_mutant_esmc_embeddings.npy`
- `variant_panel_mutant_esmc_embeddings.meta.json`
- `variant_panel_mutant_esmc_vocab.json`
- `mutant_endpoint_predictions.csv`
- `mutant_endpoint_predictions_report.json`
- `variant_panel_endpoint_predictions_annotated.csv`
- `variant_panel_validation_summary.json`
- `variant_panel_validation_report.md`

Required plots:

- percent delta by class
- percent delta by panel type
- per-gene variant deltas
- matched benign/deleterious same-gene paired deltas
- ESM-C L2 distance from WT vs percent delta
- domain vs non-domain subgroup plot

## Embedding And Prediction

Generate ESM-C 600M `mean_non_special` embeddings using the local standalone script:

```bash
.venv/bin/python mvp/make_esmc_embeddings.py \
  --model-size 600m \
  --pool mean_non_special \
  --device cuda \
  --vocab "$OUT/variant_panel_mutant_esmc_vocab_input.json" \
  --sequences "$OUT/variant_panel_mutant_esmc_sequences.json" \
  --snapshot-dir data/raw/protein/esmc/esmc-600m-2024-12 \
  --out "$OUT/variant_panel_mutant_esmc_embeddings.npy" \
  --mask-out "$OUT/variant_panel_mutant_esmc_embeddings_mask.npy" \
  --meta-out "$OUT/variant_panel_mutant_esmc_embeddings.meta.json" \
  --batch-size 1
```

Run the existing frozen endpoint evaluator:

```bash
.venv/bin/python scripts/evaluate_mutant_endpoint_predictions.py \
  --strict \
  --outdir "$OUT" \
  --mutant-metadata "$OUT/variant_panel_mutant_sequences_metadata.csv" \
  --mutant-embedding-matrix "$OUT/variant_panel_mutant_esmc_embeddings.npy" \
  --mutant-vocab "$OUT/variant_panel_mutant_esmc_vocab.json" \
  --latents tmp/hopfield_fitting_yolo_full_rerun/perturbation_latents.npz \
  --checkpoint tmp/hopfield_fitting_yolo_full_rerun/otcfm_real_overfit.pt \
  --device cuda
```

Primary endpoint:

```text
percent_delta = 100 * mutant_minus_wt_predicted_response_l2 / wt_predicted_response_l2
```

Negative values mean the mutant has weaker predicted response than WT.

## Statistical Analysis

Primary broad-panel tests:

- Mann-Whitney U test comparing percent delta for benign vs deleterious variants
- Cliff's delta or rank-biserial effect size
- AUROC for deleterious classification using `-percent_delta`
- bootstrap confidence intervals for median percent delta by class and class difference

Same-gene/domain tests:

- paired signed-rank tests for matched benign/deleterious variants within gene
- mixed-effects model:

```text
percent_delta ~ label_class + esmc_l2_distance_from_wt + protein_position_fraction + domain_indicator + (1 | gene_symbol)
```

If mixed-effects tooling is inconvenient, use a cluster bootstrap by gene.

Calibration checks:

- compare benign variants with `abs(percent_delta) <= 5%`
- compare deleterious variants with `percent_delta < 0`
- stratify by:
  - review status strength
  - DNA-binding-domain membership
  - same-gene matched availability
  - WT predicted response magnitude
  - ESM-C L2 distance from WT
  - local `n_cells`

Multiple testing:

- Treat Panel A as broad discovery.
- Treat Panel B and Panel C as more biologically interpretable validation subsets.
- Report unadjusted p-values plus clearly label subgroup analyses.

## Success Criteria

Minimum useful result:

- at least 100 variants per class in Panel A, or a clear documented reason why yield is lower
- strict endpoint evaluator `status == ok`
- no coordinate mismatches in locked panels
- all plots and summary files generated

Statistical success:

- deleterious median percent delta is more negative than benign
- AUROC using `-percent_delta` is meaningfully above 0.5, target `>= 0.65`
- Mann-Whitney or bootstrap class difference excludes zero
- same-gene matched panel shows the same direction

Biological success:

- DNA-binding-domain-enriched subset has stronger separation than the broad ClinVar panel
- benign variants mostly stay near WT, e.g. high fraction with `abs(percent_delta) <= 5%`
- deleterious variants show consistent weak-response direction without being driven by one gene family

Failure modes to report explicitly:

- larger Panel A remains near AUROC 0.5
- DBD-enriched panel does not improve over all-ClinVar panel
- signal is dominated by ESM-C distance from WT rather than clinical class
- signal is dominated by one gene or TF family
- endpoint checkpoint appears insensitive to mutation effects

## Interpretation Guardrails

Keep claims bounded.

- Current endpoint predictions use the PCA-centroid OT-CFM smoke checkpoint.
- ClinVar pathogenicity is not equivalent to loss of TF DNA binding.
- Many pathogenic variants may act through protein stability, interactions, localization, expression, or developmental dosage, not acute transcriptional response in this assay.
- Benign variants are not guaranteed to be neutral in ESM-C space.
- A statistically significant broad-panel result may still be biologically weak if the effect size is tiny.

Recommended report language if successful:

> In an expanded ClinVar-curated mutation panel, pathogenic/likely pathogenic missense variants produced a more negative predicted response shift than benign/likely benign variants, with stronger separation in DNA-binding-domain-enriched and same-gene matched subsets. This supports the model's sensitivity to sequence perturbations, but the current analysis remains tied to the PCA-centroid endpoint checkpoint.

Recommended report language if weak:

> Expanding the ClinVar mutation panel increased power but did not produce strong separation between benign and pathogenic classes under the current PCA-centroid checkpoint. The weak trend suggests either the effect is small, the endpoint is not mutation-sensitive enough, or broad ClinVar pathogenicity is too heterogeneous for this validation target. The next test should focus on DNA-binding-domain variants with direct functional evidence.

## Next Implementation Step

Promote cleaned versions of the temporary scripts from the 48-variant run into source-controlled scripts:

- previous curation script:
  `tmp/hoptf_variant_validation_20260511/curate_variant_panel.py`
- previous summary script:
  `tmp/hoptf_variant_validation_20260511/summarize_variant_panel.py`

Suggested tracked scripts:

- `scripts/curate_clinvar_variant_panel.py`
- `scripts/summarize_variant_panel_predictions.py`

Do not commit the downloaded ClinVar table, ESM-C embeddings, model weights, `.npy` arrays, or output bundles.
