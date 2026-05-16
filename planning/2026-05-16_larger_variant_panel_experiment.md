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

Target:

- 100-300 benign / likely benign variants
- 100-300 pathogenic / likely pathogenic variants
- all coordinate-checked against local responder isoforms
- cap at 3-5 variants per gene per class

Use this panel to ask whether the broad ClinVar signal becomes statistically stable.

Expected limitation:

ClinVar pathogenicity may reflect disease mechanisms unrelated to TF DNA binding or transcriptional response amplitude.

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
