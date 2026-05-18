# HopTF Experiment Plan: May 17 Follow-Up Experiments

## Executive Summary

The current HopTF mutation and endpoint evidence uses a PCA-derived perturbation response representation. That representation has been useful, but it is not the only possible way to measure TF-induced cell-state response. The active open question is whether the same conclusions hold when response is measured using SCARF or another cell-level representation.

This plan starts with one planned experiment and adds follow-up experiments motivated by the 2026-05-17 result bundle from `tmp/hoptf_followup_full_20260517/`:

1. **SCARF response representation check**: test whether the variant and sequence-feature conclusions from the PCA-derived response representation persist, weaken, or improve when the response target is computed in SCARF latent space.
2. **Benign/control variant curation setup**: build an evidence-tiered benign/control panel instead of treating every non-pathogenic ClinVar label as a reliable negative control.
3. **Narrow missense variant sensitivity redo**: redo the variant experiment around high-confidence pathogenic single-residue substitutions versus matched random substitutions, with evidence-tiered benign/control variants reported as secondary stratified analyses.
4. **Variant retrieval example panels**: make two concrete genomic retrieval examples showing how WT, pathogenic mutant, and matched control differ across AlphaGenome windows.
5. **Beta retrieval-regime finalization**: finish the beta sensitivity analysis needed to support the associative-memory framing.
6. **Biological validation redesign around resolution limits**: redo the motif/binding validation with clearer definitions, better plots, and an explicit caveat about motif-size versus AlphaGenome-window-size mismatch.
7. **Isoform domain annotation audit**: check whether the negative domain-aware isoform result reflects weak biology or weak annotation retrieval.
8. **Sequence encoder appendix diagnostic**: keep ESM-C versus ESM-DBP as an appendix diagnostic with no further follow-up unless the paper needs extra reviewer-facing detail.

This is not a broad model-development task. It is a bounded set of validation and diagnostic experiments. The first and highest-priority experiment is a response-representation validation. Its purpose is to answer:

> Are HopTF's sequence-conditioned conclusions stable when the perturbed-cell response is measured in SCARF latent space, after controlling for protein length, ORF length, and recovered cell count?

This control is required. Earlier HopTF analyses established that protein length, ORF length, recovered cell count, and response strength are strongly related in the TF Atlas data. Any SCARF response analysis that does not control for protein length and recovered cell count would repeat the main weakness of an uncontrolled PCA-response analysis.

## Tie-In To Existing Plans And Paper Draft

This plan extends:

- [planning/2026-05-16_experiment_plan.md](/home/dmeyer/courses/clmm/HopTF/planning/2026-05-16_experiment_plan.md), especially Experiment 1 and Experiment 2.
- [planning/2026-05-16_experimental_plan_followup.md](/home/dmeyer/courses/clmm/HopTF/planning/2026-05-16_experimental_plan_followup.md), especially the matched variant validation design.
- `paper/HopTF_full_draft.tex`, especially the sections describing SCARF as the cell encoder and the current caveats around PCA-derived response targets.

The paper currently uses PCA-derived results as the concrete evidence base. The SCARF experiment should decide whether the final report can say:

- the PCA-derived result is robust to a cell-representation change;
- the SCARF representation gives a cleaner response signal;
- or the mutation/sequence result depends on the PCA-derived response representation and should be framed more cautiously.

## Claim Ladder For The Paper

The experiments should support a clear sequence of claims, with stronger claims allowed only if the corresponding controls pass.

1. **Predictive claim**: TF sequence features, after projection into AlphaGenome key space, add modest information for predicting the state of the perturbed cell beyond protein length, ORF length, and recovered cell count. This claim is already supported in the PCA-derived benchmark and will become stronger if it persists in SCARF space.
2. **Sequence-sensitivity claim**: missense TF variants can change predicted perturbed-cell state activity more than matched random substitutions. This claim is currently suggestive for matched random controls but not clean enough for a main conclusion without the redesigned signed/absolute tests and ESM-C-distance diagnostics.
3. **Retrieval-movement claim**: variants that change predicted activity can also move the AlphaGenome-key retrieval distribution. This should be presented as a diagnostic of the model's intermediate representation, not as evidence of changed binding unless independent locus evidence is added.
4. **Biological-validation claim**: high-retrieval AlphaGenome windows should recover independent motif, binding, or accessibility support above matched backgrounds. Current exact-symbol motif and ReMap recall results are weak, so the paper should frame this as an unresolved validation challenge unless the redesigned family-level and accessibility-supported analyses improve.
5. **Associative-memory behavior claim**: beta changes retrieval concentration in a predictable way, exposing diffuse, intermediate, and concentrated retrieval regimes. This claim is needed for the Hopfield/associative-memory framing and should be tied to prediction quality and biological validation rather than shown as entropy alone.
6. **Interpretability boundary**: high-retrieval loci are model-prioritized regulatory-context hypotheses. The paper should not call them binding sites, causal regulatory elements, or validated targets without independent evidence.

## Completed Evidence Versus Pending Evidence

Completed evidence that can be used now:

- PCA-derived held-out-gene benchmark: protein length, ORF length, and recovered cell count are strong predictors; ESM-C projected into AlphaGenome key space gives a modest improvement over those inputs; raw ESM-C and ESM-DBP do not support the story as strongly.
- ESM-C versus ESM-DBP recheck: ESM-C outperforms ESM-DBP across overall, length-matched, cell-count-matched, and joint matched settings.
- May 17 matched variant run: pathogenic variants exceed matched random substitutions under the previous absolute-change framing, but the gene-bootstrap interval crosses zero and the benign comparison fails.
- May 17 retrieval-change run: pathogenic variants move retrieval more than matched random substitutions, but the result should be treated as a follow-up diagnostic.
- May 17 motif and binding recall: exact same-symbol motif and exact same-TF ReMap recall are weak relative to random-rank expectation, making biological validation a main open issue.
- May 17 isoform diagnostic: current UniProt-based domain annotation coverage is too poor for a main isoform claim.

Pending evidence needed before stronger paper claims:

- SCARF response reconstruction and SCARF-space reanalysis.
- Evidence-tiered benign/control curation.
- Redesigned signed and absolute narrow missense tests.
- Variant retrieval example panels.
- Beta sensitivity finalization linking prediction quality, retrieval concentration, and motif/binding support.
- Family-level and accessibility-supported motif/binding validation.
- Isoform annotation audit across UniProt, InterPro, Pfam, Ensembl/GENCODE, and TF-specific resources.

## Key Definitions

**PCA-derived response representation**: the current perturbation-level response target derived from PCA-space summaries of control and TF-perturbed cells. This is the representation used in the completed endpoint, variant, and matched random-substitution results.

**SCARF response representation**: a response target computed from SCARF latent embeddings of control and TF-perturbed cells. The exact construction must be recorded: SCARF model version, input data, latent dimension, whether the model is frozen, how cells are grouped by perturbation, and how a perturbation-level response vector or score is computed.

**Recovered cell count**: the number of cells recovered for a TF perturbation or isoform in the local TF Atlas subset, usually stored as `n_cells`. This is a required covariate and matching variable because it is strongly related to measured response and likely reflects both technical recovery and biological response.

**Protein length and ORF length**: sequence metadata used as required covariates. Longer ORFs and proteins can be harder to package or recover in overexpression screens. They are not biological response measures, but they are strong predictors of observed response behavior in the current data.

**Response-representation agreement**: whether the PCA-derived and SCARF-derived analyses give the same direction and similar relative ranking of variants, TFs, or sequence-feature models. Agreement does not require identical numeric values because the latent spaces have different scales.

**Mutation edit size**: the literal sequence edit being tested, such as a single-nucleotide variant, one amino-acid substitution, or a larger insertion/deletion. This is not the same as ESM-C distance. For the narrow variant experiment, the intended edit-size scope is missense SNVs that create exactly one amino-acid substitution.

**ESM-C distance from wild type**: the distance between the wild-type and mutant protein embeddings in ESM-C representation space. This is a model-space perturbation magnitude, not the number of edited base pairs and not a direct biological severity score. A single-nucleotide missense variant can still have a large ESM-C distance if the amino-acid substitution looks unusual to the encoder. Use ESM-C distance for matching, caliper checks, and outlier diagnostics, but do not describe high ESM-C-distance variants as "large mutations" or "large edits."

## Experiment List

1. SCARF response representation check.
2. Benign/control variant curation setup.
3. Narrow missense variant sensitivity redo.
4. Variant retrieval example panels.
5. Beta retrieval-regime finalization.
6. Biological validation redesign around resolution limits.
7. Isoform domain annotation audit.
8. Sequence encoder appendix diagnostic.

Any additional experiments should be added only if they strengthen one of the claim-ladder steps above or resolve a specific blocker for the paper.

## 1. SCARF Response Representation Check

### Question

Does the HopTF sequence signal persist when predicted perturbed-cell state activity is measured in SCARF latent space instead of the current PCA-derived response representation?

The practical subquestions are:

- Does ESM-C projected into AlphaGenome key space still improve prediction over protein length, ORF length, and recovered cell count?
- Do pathogenic variants produce weaker predicted TF responses, or larger response disruptions, than matched random substitutions?
- Do evidence-tiered benign/control variants behave differently from pathogenic variants, and do high-ESM-C-distance controls behave like a distinct stress-test subset rather than ordinary negatives?
- Is the SCARF-derived response less tied to protein length and recovered cell count than the PCA-derived response, or does it share the same confounding structure?

### Paper Role

This experiment is a response-representation robustness check. It should not replace the variant or retrieval experiments. It decides how much confidence to place in the PCA-derived response results.

If SCARF agrees with the PCA-derived results, the paper can say the sequence-sensitivity findings are not specific to the PCA response target.

If SCARF weakens the result, the paper should say the current evidence depends on response representation and should frame mutation validation as exploratory.

If SCARF strengthens the result after the same controls, the SCARF version should become the preferred response result for the final report, with the PCA version retained as a supporting analysis.

Paper edit implied:

- Add a Results subsection titled "SCARF response representation robustness" after the current PCA-derived prediction and ablation results.
- If SCARF agrees with PCA, report a compact side-by-side table and use the Discussion to say that the sequence signal is not tied to one latent representation.
- If SCARF disagrees with PCA, keep PCA as a diagnostic benchmark and state that response-representation choice is a limitation.
- In all cases, keep the protein length, ORF length, and recovered-cell-count controls in the main text, not only the appendix.

### Inputs

- TF Atlas cell-level expression data used by the current HopTF pipeline.
- SCARF embeddings for control and TF-perturbed cells.
- Existing perturbation metadata with `gene_symbol`, `isoform_id`, `isoform_embedding_id`, `label_status`, `n_cells`, `protein_aa_length`, and ORF length if available.
- Existing PCA-derived perturbation response artifacts for side-by-side comparison.
- ESM-C embeddings and HopTF projected query features used in the sequence ablation.
- Matched variant/control panels from the May 16 follow-up design, including ESM-C distance from wild type.
- Existing matched random-substitution outputs if the same locked panel can be reused.

### SCARF Inventory Status

SCARF cell-level embeddings are available and should be used before generating any new embeddings.

Cluster path:

- `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/data/processed/TFAtlas_subsample_raw_csr_scarf.h5ad`

Local mount path:

- `~/knowles_lab/dmeyer/hoptf/data/processed/TFAtlas_subsample_raw_csr_scarf.h5ad`

Provenance and integrity:

- file size: `8,308,791,592` bytes.
- SHA256: `f57649b8e6b520de72697848fefd5f9a6d523eca5a9e6a802777ead03345e75b`.
- matches Hugging Face LFS object: `hf://datasets/pvd232/HopTF/processed/TFAtlas_subsample_raw_csr_scarf.h5ad`.
- inventory JSON: `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/data/processed/TFAtlas_subsample_raw_csr_scarf.inventory.json`.

Coverage:

- h5ad shape: `671,453` cells by `37,528` genes.
- `obsm["X_scarf"]`: `671,453 x 512`.
- `layers["counts"]`: present.
- `obs["TF"]`: present.
- unique TF/perturbation labels: `3,266`.
- GFP control cells: `1,000`.
- mCherry control cells: `1,000`.
- h5ad `obs["TF"]` labels exactly match HopTF perturbation metadata: `3,266 / 3,266`.

Controlled-analysis metadata:

- metadata file: `data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv`.
- required fields are present for all `3,266` perturbations: `n_cells`, `protein_aa_length`, `orf_nt_length`, `gene_symbol`, `isoform_id`, `isoform_embedding_id`, `label_status`.

Variant-panel join:

- previously checked variant panel unique WT `isoform_embedding_id`: `252`.
- matched current HopTF perturbation metadata: `252 / 252`.
- since SCARF h5ad TF labels match the same perturbation set, WT isoforms in the variant panel have SCARF-backed cell groups.

SCARF provenance from `uns["scarf_embedding"]`:

- modality: RNA-only.
- embedding key: `X_scarf`.
- latent dimension: `512`.
- species: hg38.
- counts layer: `counts`.
- generation device: cuda.
- checkpoint path used during generation: `/gpfs/commons/home/dmeyer/HopTF/data/reference/scarf/weights`.
- prior data path: `/gpfs/commons/home/dmeyer/HopTF/data/reference/scarf/prior_data`.
- median source: `derived_from_input_counts`.
- model genes used: `17,854`.

Still missing:

- perturbation-level SCARF response table. We need to compute mean SCARF latent per TF perturbation minus mean SCARF latent for the chosen control set.

### Required Construction Details

The SCARF response target must be documented before running statistical comparisons:

- SCARF source h5ad path and SHA256.
- input modality: RNA-only, unless inventory inspection contradicts this.
- whether any additional fine-tuning is performed for this experiment; default should be no additional fine-tuning.
- latent dimension: `512`.
- Cell filtering rules.
- Control-cell set used as the reference.
- Perturbed-cell grouping rule by TF isoform or gene.
- Minimum recovered-cell count threshold.
- Definition of the SCARF response vector.
- Definition of any scalar SCARF response score.

Recommended first-pass SCARF response vector:

> For each perturbation, compute the mean SCARF latent of perturbed cells minus the mean SCARF latent of control cells. Use GFP and mCherry controls as the initial pooled control set unless metadata indicate that one control is more appropriate for a specific perturbation batch. If batch/library metadata support matched controls, run that as a sensitivity analysis rather than blocking the first pass.

Recommended first-pass SCARF response magnitude:

> L2 norm of the perturbation response vector in SCARF latent space, with optional control-standardized scaling by the control-cell latent standard deviation.

This mirrors the PCA-derived setup closely enough to make the comparison interpretable.

### Required Confound Controls

Every SCARF analysis must include the same core controls used for the PCA-derived response analysis:

- protein length;
- ORF length, if available;
- recovered cell count, preferably `log1p(n_cells)` in regression models;
- grouped or matched evaluation within protein-length bins;
- grouped or matched evaluation within recovered-cell-count bins;
- grouped or matched evaluation within joint protein-length and recovered-cell-count bins.

Do not treat recovered cell count as a harmless nuisance variable. It may partly reflect technical recovery, biological response strength, and measurement uncertainty. The plan should therefore report both controlled models and diagnostic correlations rather than simply "regressing it away" and acting as if the problem is solved.

### Implementation Slices

#### 1A. Build SCARF Perturbation Response Artifacts

Question:
Can we construct a clean SCARF response target for the same TF perturbations used in the PCA-derived analysis?

Outputs:

- `scarf_response_latents.npz`: perturbation-level SCARF response vectors.
- `scarf_response_metadata.csv`: one row per perturbation with identifiers, label status, `n_cells`, protein length, ORF length, and response-vector row index.
- `scarf_response_report.json`: SCARF source, dimensions, filtering rules, number of control cells, number of perturbed cells, number of perturbations, and missingness.

Diagnostic plots:

1. **SCARF response magnitude by recovered cell count**
   - x-axis: `log1p(n_cells)`.
   - y-axis: SCARF response magnitude.
   - Concern diagnosed: SCARF response may still be strongly driven by recovered-cell count or measurement noise.

2. **SCARF response magnitude by protein length**
   - x-axis: protein amino-acid length.
   - y-axis: SCARF response magnitude.
   - Concern diagnosed: SCARF response may preserve the protein-length artifact seen in the PCA-derived response.

3. **PCA response magnitude versus SCARF response magnitude**
   - x-axis: PCA-derived response magnitude.
   - y-axis: SCARF response magnitude.
   - Concern diagnosed: the two response representations may measure different aspects of perturbation response; disagreement would limit direct comparability.

Statistical checks:

- Spearman correlation between SCARF response magnitude and `log1p(n_cells)`.
- Spearman correlation between SCARF response magnitude and protein length.
- Spearman correlation between SCARF and PCA-derived response magnitudes.

Why these tests:
Spearman correlation checks whether one quantity tends to increase or decrease with another without assuming a linear relationship. These are diagnostics, not causal tests.

How to interpret:
Strong correlations with recovered cell count or protein length mean SCARF does not remove the confounding problem and must be controlled just as carefully as PCA. Strong correlation with PCA means the two response targets are broadly aligned. Weak correlation with PCA means SCARF may be measuring a different response geometry.

#### 1B. Repeat Controlled Sequence-Feature Benchmark In SCARF Space

Question:
Does TF sequence help predict SCARF-derived perturbed-cell response after the model already has protein length, ORF length, and recovered cell count?

Minimum models:

- protein length + ORF length + recovered cell count;
- raw ESM-C;
- protein length + ORF length + recovered cell count + raw ESM-C;
- ESM-C projected into AlphaGenome key space;
- protein length + ORF length + recovered cell count + ESM-C projected into AlphaGenome key space;
- shuffled ESM-C control if time allows.

Primary metric:

- grouped held-out-gene MSE for SCARF response vectors.

Secondary metrics:

- cosine similarity between predicted and observed SCARF response vectors;
- MSE within protein-length bins;
- MSE within recovered-cell-count bins;
- MSE within joint protein-length and recovered-cell-count bins.

Required plots:

1. **SCARF sequence-feature benchmark**
   - x-axis: model feature set.
   - y-axis: grouped held-out-gene MSE in SCARF response space.
   - Concern diagnosed: sequence features may not improve beyond length and recovered-cell controls.

2. **SCARF percent improvement over controlled baseline**
   - x-axis: held-out setting: all genes, within protein-length bins, within recovered-cell-count bins, within joint bins.
   - y-axis: feature set.
   - color: percent MSE improvement over protein length + ORF length + recovered cell count, where positive means better than that baseline.
   - Concern diagnosed: an apparent sequence gain may disappear after controlling for protein length and recovered cell count.

Statistical tests:

- Paired bootstrap over held-out genes for the MSE difference between the controlled baseline and the controlled sequence-feature model.
- Optional paired sign test over held-out genes for whether the sequence-feature model improves more genes than it worsens.

Why these tests:
The same held-out genes are evaluated by both models, so paired differences are the relevant quantity. Bootstrapping over genes gives an uncertainty interval without assuming the per-gene MSE differences are normally distributed. The sign test asks a simpler robustness question: does the sequence model improve a majority of held-out genes?

How to interpret:
If the confidence interval for MSE improvement is above zero, the sequence feature adds SCARF-response signal beyond length and recovered-cell controls. If the confidence interval crosses zero, treat any gain as inconclusive. If the sequence model only improves the uncontrolled setting but not the length/cell-count-matched settings, do not claim robust sequence signal.

Success criteria:

- ESM-C projected into AlphaGenome key space improves over protein length + ORF length + recovered cell count in SCARF response space.
- Improvement remains positive within protein-length bins, recovered-cell-count bins, and joint bins.
- Shuffled ESM-C does not improve over the controlled baseline.

#### 1C. Re-score Variant Effects In SCARF Space

Question:
Do the May 16 matched variant conclusions persist when WT-vs-mutant changes are measured using SCARF-derived response?

Use the same locked variant/control panel as the matched PCA-derived analysis wherever possible. Do not redesign the variant panel for SCARF unless the existing panel cannot be scored.

Required outputs:

- `scarf_variant_activity_results.csv`: WT, mutant, random-control, and optional evidence-tiered benign/control predictions or response scores in SCARF space.
- `scarf_variant_activity_differences.csv`: mutant-minus-WT and control-minus-WT changes in SCARF space.
- `scarf_pca_variant_comparison.csv`: same variants joined across PCA-derived and SCARF-derived metrics.
- `scarf_variant_validation_summary.md`: matched tests, ESM-C distance diagnostics, and comparison to PCA-derived conclusions.

Required plots:

1. **SCARF signed response-change plot**
   - x-axis: pathogenic real, matched random, optional evidence-tiered benign/control groups.
   - y-axis: signed mutant-minus-WT SCARF response change, with the sign convention defined so negative means weaker predicted response than WT.
   - Concern diagnosed: pathogenic variants may impair TF function and produce a weaker response rather than simply a larger absolute change.

2. **SCARF absolute activity-change plot**
   - x-axis: pathogenic real, matched random, optional evidence-tiered benign/control groups.
   - y-axis: absolute WT-vs-mutant change in SCARF response space.
   - Concern diagnosed: pathogenic variants may be more disruptive in either direction even when the signed weakening effect is mixed.

3. **SCARF activity change versus ESM-C distance**
   - x-axis: ESM-C distance from wild type.
   - y-axis: signed and absolute WT-vs-mutant change in SCARF response space, shown as separate panels.
   - color: pathogenic, evidence-tiered benign/control, random.
   - Concern diagnosed: SCARF variant effects may still mostly reflect the size of the ESM-C perturbation.

4. **PCA-derived versus SCARF variant effect**
   - x-axis: WT-vs-mutant change under the PCA-derived response representation.
   - y-axis: WT-vs-mutant change under the SCARF response representation.
   - color: pathogenic, evidence-tiered benign/control, random.
   - Make separate panels for signed change and absolute change.
   - Concern diagnosed: the variant conclusion may be response-representation-specific.

5. **SCARF real-minus-control difference plot**
   - x-axis: comparison type: pathogenic minus matched random, with pathogenic minus evidence-tiered benign/control shown only as a descriptive sensitivity analysis unless matching diagnostics support a formal secondary comparison.
   - y-axis: paired difference in signed SCARF response change and paired difference in absolute SCARF response change, shown separately.
   - Concern diagnosed: the matched-control advantage may depend on group averages rather than consistent within-pair differences.

Statistical tests:

- Primary biological direction test: one-sided paired Wilcoxon signed-rank test comparing signed pathogenic real variant change against signed matched random-control change in SCARF response space. Define the sign so negative means the mutant evokes a weaker predicted response than WT; the one-sided alternative is then that real pathogenic variants are more negative than matched random controls.
- Secondary disruption test: one-sided paired Wilcoxon signed-rank test comparing absolute pathogenic real variant change against absolute matched random-control change in SCARF response space. This asks whether pathogenic variants change the response more than matched controls regardless of direction.
- Optional secondary signed and absolute comparisons against evidence-tiered benign/control variants where matched controls are available. Report core, matched, and high-ESM-C-distance stress-test controls separately.
- Spearman correlation between signed and absolute SCARF variant effect size and ESM-C distance.
- Spearman correlation between signed and absolute PCA-derived variant effect size and SCARF variant effect size.
- Gene-level bootstrap confidence interval for the median real-minus-control difference.

Why these tests:
The variant design is matched, so paired tests are appropriate. The signed weakening test is biologically important because a pathogenic TF variant may impair TF function and therefore evoke a weaker predicted response rather than a larger response. The absolute disruption test is still useful because some pathogenic variants may alter the response in either direction. The Wilcoxon signed-rank test does not assume normally distributed paired differences. Spearman correlations diagnose whether SCARF effects are driven by ESM-C distance and whether SCARF agrees with PCA. Gene-level bootstrapping avoids treating multiple variants from one TF as fully independent.

How to interpret:
If pathogenic variants are more negative than matched random controls in the signed test, that supports a loss-of-function or weakened-response interpretation. If pathogenic variants exceed matched random controls in the absolute test, that supports a broader disruption interpretation. If the signed and absolute tests disagree, report both rather than forcing one story. If the result is weak in SCARF but strong in PCA, the report should say the variant signal depends on the response representation. If SCARF effects correlate more strongly with ESM-C distance than with clinical label or matched-control status, the result should remain cautious.

Success criteria:

- Pathogenic real variants are more negative than matched random controls in the signed weakening test, or larger than matched random controls in the absolute disruption test, with the supported interpretation stated explicitly.
- Evidence-tiered benign/control comparisons do not contradict the interpretation, but they are not required for the primary claim unless the control curation and matching diagnostics are strong.
- The signal is not explained entirely by ESM-C representation-space distance.
- PCA-derived and SCARF-derived effect sizes are directionally aligned.

### Expected Runtime

SCARF cell embeddings are available, so this should be moderate rather than a new embedding-generation job. The main work is computing perturbation-level response vectors from the 8.3GB h5ad, joining metadata, and adapting the current response-scoring scripts.

The controlled sequence-feature benchmark and variant scoring should be smaller than the full motif/binding validation once SCARF response artifacts are available.

### Blockers

- No perturbation-level SCARF response table exists yet.
- Control-cell matching may require a choice between pooled GFP/mCherry controls and batch/library-matched controls if batch/library metadata are available.
- SCARF latent scale may need control-standardization before comparing response magnitudes.
- Existing WT-vs-mutant evaluator may assume PCA-derived targets and require adaptation.

### QC Checklist

- Confirm the SCARF embedding rows align to cell metadata.
- Confirm all perturbation identifiers match the HopTF metadata.
- Report number of control cells and perturbed cells used for each response vector.
- Report missing perturbations and why they are missing.
- Plot SCARF response magnitude versus `log1p(n_cells)`.
- Plot SCARF response magnitude versus protein length.
- Include ORF length whenever available.
- Compare PCA-derived and SCARF-derived response magnitudes before running variant interpretation.
- Use the same locked variant panel as the PCA-derived follow-up whenever possible.
- Use the same ESM-C-distance diagnostics as the PCA-derived variant analysis.

### Clear Do-Not-Claim-Yet Caveats

- Do not claim SCARF is automatically more biological than PCA. It is a different representation and must be checked.
- Do not claim sequence signal from SCARF unless it improves over protein length, ORF length, and recovered-cell controls.
- Do not claim variant validation from SCARF unless matched controls and ESM-C distance diagnostics support it.
- Do not compare raw SCARF and PCA magnitudes directly without noting that the latent spaces have different scales.
- Do not treat recovered cell count as a purely technical nuisance variable; it can reflect both recovery artifacts and biological response.

## SCARF Data And Script Checklist

Data likely needed:

- TF Atlas cell-level expression data.
- Cell metadata with perturbation and isoform identifiers.
- existing SCARF h5ad at `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/data/processed/TFAtlas_subsample_raw_csr_scarf.h5ad`.
- Existing PCA-derived response artifacts.
- Existing ESM-C embeddings and projected HopTF query features.
- Existing matched variant/control panel.
- Protein length, ORF length, and recovered-cell metadata.

Scripts likely needed:

- SCARF h5ad loader for `obsm["X_scarf"]` and `obs["TF"]`.
- Perturbation-level SCARF response builder.
- Controlled SCARF sequence-feature benchmark.
- SCARF variant response scorer.
- PCA-versus-SCARF comparison plots and summaries.

## SCARF Implementation Unknowns To Resolve Before Asking User

Resolve these from files, metadata, or cluster context before asking for user input:

- whether batch/library metadata support matched control cells or whether pooled GFP/mCherry controls are the clean first pass;
- whether cell metadata support matched control cells by batch/library.

Resolved from laplace inventory:

- SCARF embeddings exist and should not be regenerated.
- canonical file for this report is `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/data/processed/TFAtlas_subsample_raw_csr_scarf.h5ad`.
- ORF length is present in the metadata for all `3,266` perturbations.
- variant-panel WT isoforms join to the SCARF-backed perturbation metadata.

User input is only needed if control-cell matching requires a scientific choice that cannot be inferred from metadata.

## 2. Benign/Control Variant Curation Setup

### Result Trigger

The May 17 benign-variant guide in [planning/2026-05-17_benign_guide.md](/home/dmeyer/courses/clmm/HopTF/planning/2026-05-17_benign_guide.md) changes how the negative controls should be handled. The previous plan treated benign ClinVar labels as exploratory because they were too heterogeneous for the primary comparison. That remains true, but the guide makes a stronger point: a useful benign/control panel can be built if each variant has explicit evidence provenance, strict mapping, hard exclusions, and clear separation between clinical benignity, population tolerance, and functional neutrality.

This setup step should happen before any new benign-versus-pathogenic claim. It does not replace the matched random-substitution control. Instead, it creates a better secondary control panel and a stress-test subset for asking whether HopTF overreacts to variants that have independent benign or tolerated evidence.

### Question

Can we construct a high-precision benign/control variant panel for TF missense variants that is strong enough to use as a secondary comparison, without defining the controls using ESM-C or HopTF outputs?

### Paper Role

This is a setup experiment, but it belongs in the Methods or Appendix because it determines whether any benign-versus-pathogenic comparison is interpretable. The paper should not present "benign variants" as a single label. It should describe clinical benignity, population tolerance, functional neutrality, and synthetic ortholog-supported controls as separate evidence categories.

Paper edit implied:

- Add a Methods paragraph or appendix subsection titled "Variant label curation and control tiers."
- In the Results variant section, report random controls as the primary comparison and evidence-tiered benign/control variants as secondary analyses.
- If the curated benign/control set is too small or unbalanced, explicitly say that benign controls were not used for the main hypothesis test.

### Definition To Use Going Forward

**Benign/control variant**: a missense variant used as a negative or tolerated comparison because it has independent evidence from clinical curation, population frequency, or functional assays. This is not a claim that the variant has no molecular effect in every TF context.

**Evidence tier**: a recorded source category that explains why a variant is allowed into the benign/control panel. The tier is part of the analysis and should be plotted or reported, not hidden.

Recommended tiers:

- **Tier 1, clinical benign**: ClinVar `Benign`, `Likely benign`, or `Benign/Likely benign`, preferably with two-star-or-better review status, no conflicts, exact transcript/protein match, missense consequence only, no splice-risk concern, and no somatic-only interpretation if germline tolerance is the intended evidence.
- **Tier 2, population tolerated**: gnomAD PASS missense variants with adequate allele count and allele frequency, no ClinVar pathogenic, likely pathogenic, VUS, or conflict annotation, exact transcript/protein match, and no obvious TF-feature or splice-risk concern. Prefer group maximum or overall allele frequency at least `0.1%`, with `1%` and many carriers treated as stronger evidence.
- **Tier 3, functional neutral**: variants with WT-like results in endpoint-relevant MAVE, DMS, PBM, reporter, or similar assays. Assay endpoint must be recorded because molecular neutrality in one assay is not universal benignity.
- **Tier 4, ortholog-supported or family-supported synthetic control**: synthetic substitutions supported by ortholog or family evidence. These should be reported separately from natural human benign controls and should not be pooled into the main benign set unless natural controls are too sparse.

### Inputs

- ClinVar missense variants with clinical significance, review status, conflict status, VCV identifier, and transcript/protein notation.
- gnomAD PASS missense variants with version, allele count, allele number, allele frequency, group maximum allele frequency, population label, and homozygote count when relevant.
- MaveDB or published DMS, MAVE, PBM, and reporter data when available for TFs in the panel.
- Local HopTF isoform and perturbation metadata.
- InterPro and Pfam domain annotations.
- TF-feature annotations for DNA-binding domains, dimerization regions, activation/repression domains, NLS/NES, PTM sites, degrons, short linear motifs, and known interfaces where available.
- Conservation and alignment-derived features, including MSA entropy, wild-type residue frequency, mutant residue frequency, and whether the mutant residue is observed in orthologs.
- Structure-derived annotations where reliable: pLDDT, solvent exposure, buriedness, interface proximity, and optional stability or inverse-folding scores.
- ESM-derived annotations for stratification and diagnostics, including ESM log-likelihood ratio, site-normalized ESM delta, local-window delta, global embedding delta, and ESM-C distance from WT.

### Hard Exclusions

Exclude candidate benign/control variants if any of these apply:

- not missense in the chosen HopTF isoform;
- reference amino acid does not match the local isoform sequence;
- ambiguous transcript, coordinate, or protein mapping;
- ClinVar pathogenic, likely pathogenic, VUS, or conflicting interpretation, unless the row is being retained only for a separate diagnostic table;
- splice-risk concern;
- low-quality population call;
- very small population allele count;
- known disease or cancer hotspot;
- residue is a canonical DNA-contact or specificity residue;
- residue is a zinc-finger Cys/His ligand or other metal/cofactor ligand;
- residue is in a core dimerization interface, known cofactor interface, NLS/NES, degron, PTM cluster, or other TF feature that makes neutrality implausible for this endpoint;
- buried high-confidence structured core residue with strong structural-risk evidence.

High ESM-C distance from WT is not a hard exclusion by itself. It should trigger stratification, manual review, and stronger independent evidence requirements. Do not define benignity using ESM-C or another model feature that HopTF may also use.

### Soft Downgrade Flags

Record these as flags rather than automatic exclusions:

- high ESM-C distance from WT;
- high ESM log-likelihood penalty;
- high conservation or low MSA entropy;
- mutant residue rarely or never observed in orthologs;
- high regional constraint;
- charge reversal, Pro/Gly introduction into structured secondary structure, or Cys gain/loss;
- near a domain boundary;
- in a low-complexity activation domain, disordered motif, or regulatory region where molecular effects may be real even if clinical consequence is weak.

### Panel Structure

Create several control subsets rather than one pooled "benign" label:

- **Core benign/control set**: strong ClinVar, gnomAD, or endpoint-relevant functional evidence; low or moderate ESM-C distance; no major biological red flags.
- **Matched benign/control set**: selected to match pathogenic positives by TF family, same gene where possible, domain class, DNA-binding-domain status, structured/disordered status, conservation bin, solvent exposure, protein length, recovered cell count, and ESM-C-distance bin.
- **High-ESM-C-distance stress-test set**: strong independent benign, population-tolerated, or functional-neutral evidence, high ESM-C distance, and no obvious domain, motif, conservation, or structure red flag after manual review.

The stress-test set answers a specific diagnostic question: does HopTF systematically call high-ESM-C-distance but independently tolerated variants disruptive? A failure there is not automatically a false positive, because population or clinical tolerance does not prove molecular neutrality for TF perturbation response.

### Outputs

- `benign_control_candidate_variants.csv`: all candidate controls before filtering.
- `benign_control_evidence_tiers.csv`: included variants with evidence tier, source label, and final include/exclude decision.
- `benign_control_exclusion_table.csv`: excluded variants with one primary exclusion reason and optional secondary flags.
- `benign_control_balance_report.md`: comparison of pathogenic positives, random controls, and benign/control subsets across gene, TF family, domain class, DNA-binding-domain status, protein length, recovered cell count, conservation, and ESM-C distance.
- `benign_control_stress_test_set.csv`: high-ESM-C-distance controls retained only for the stress-test analysis.
- `benign_control_provenance_schema.md`: exact columns and definitions needed to reproduce the panel.

### Data Staging Checklist

Data to stage or verify in `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf` before running the benign/control analysis:

- ClinVar variant summary or VCV records with GRCh38 coordinates, clinical significance, review status, conflict status, condition, transcript notation, and protein notation.
- gnomAD coding missense variants for GRCh38, with PASS status, allele count, allele number, allele frequency, group maximum allele frequency, population label, and homozygote count.
- MaveDB or published functional assay tables for TFs in the HopTF panel where available.
- UniProt, InterPro, Pfam, and TF-family/domain annotations needed to identify DNA-binding domains and other TF features.
- AlphaFold DB or PDB-derived structural annotations when available and reliable enough for residue-level review.
- MSA or conservation resources used to compute entropy, residue frequencies, and ortholog support.
- Existing HopTF isoform sequence table and perturbation metadata, used as the reference coordinate system.

Script changes likely needed:

- variant normalizer that maps ClinVar, gnomAD, and functional-assay rows onto the exact local HopTF protein isoform;
- evidence-tier assignment script with deterministic inclusion and exclusion reasons;
- TF-feature annotation joiner for domains, DNA-binding regions, interfaces, motifs, and regulatory features;
- ESM annotation script that records ESM-derived features for stratification without using them as label definitions;
- balance-report script comparing pathogenic positives, random controls, and each benign/control subset.

### Required Plots

1. **Benign/control source and tier counts**
   - x-axis: evidence tier.
   - y-axis: number of variants retained.
   - fill: source type, such as ClinVar, gnomAD, functional assay, or synthetic ortholog-supported.
   - Concern diagnosed: the benign/control set may be too small or dominated by one weak evidence source.

2. **Domain balance of pathogenic and benign/control variants**
   - x-axis: variant group: pathogenic, core benign/control, matched benign/control, high-ESM-C-distance stress test, matched random.
   - y-axis: fraction of variants.
   - fill: domain class, with DNA-binding-domain status shown explicitly.
   - Concern diagnosed: benign controls may be mostly easy non-domain or disordered-region variants while pathogenic variants are mostly DNA-binding-domain variants.

3. **ESM-C distance distribution by evidence tier**
   - x-axis: ESM-C distance from WT.
   - y-axis: density or count.
   - facets or colors: evidence tier and pathogenic/random/control group.
   - Concern diagnosed: model effects may be driven by ESM-C geometry rather than independent label evidence.

4. **Control matching feasibility plot**
   - x-axis: pathogenic variant.
   - y-axis: number of available controls after each matching requirement.
   - color or facets: random controls, core benign/control, matched benign/control.
   - Concern diagnosed: the benign/control comparison may be underpowered or biased because some pathogenic variants cannot be matched without relaxing key criteria.

5. **High-ESM-C-distance stress-test review plot**
   - x-axis: ESM-C distance bin.
   - y-axis: count or fraction of retained control candidates.
   - fill: manual review outcome: retained core, retained stress-test, excluded biological red flag, excluded weak evidence.
   - Concern diagnosed: high-ESM-C-distance benign candidates may be a mixture of true tolerated variants, molecularly ambiguous variants, and poor controls.

### Statistical Checks

- Fisher's exact test or chi-square test for whether domain-class distributions differ between pathogenic variants and each benign/control subset.
- Kolmogorov-Smirnov test or rank-sum test for whether ESM-C-distance distributions differ between pathogenic variants and each benign/control subset.
- Spearman correlation between ESM-C distance and HopTF variant effect within each benign/control subset once model scores are available.
- Gene-level bootstrap for any pathogenic-versus-benign/control model-effect comparison.

Why these tests:
The first two checks are balance diagnostics, not biological proof. They ask whether the control set differs from the pathogenic set in obvious ways that could make the comparison easy or misleading. Spearman correlation asks whether model effects rise with ESM-C distance even among controls. Gene-level bootstrapping is needed because variants from the same TF are not independent.

How to interpret:
If the benign/control set has very different domain or ESM-C-distance distributions from the pathogenic set, report it as a limited descriptive comparison rather than a clean hypothesis test. If high-ESM-C-distance controls have high HopTF effects, inspect whether they are plausible molecular effects, label mismatches, or model over-sensitivity before calling them false positives.

### Success Criteria

- At least one core benign/control subset can be constructed with clear independent evidence and exact isoform mapping.
- A matched benign/control subset is feasible for enough pathogenic variants to be useful as a secondary analysis.
- The high-ESM-C-distance stress-test subset is explicitly separated from ordinary benign controls.
- Control-set diagnostics make it clear whether benign/control comparisons are balanced enough for interpretation.

### Expected Runtime

This is mostly data integration and annotation rather than model inference. Runtime depends on whether gnomAD and MAVE/DMS/PBM annotations are already staged. If not, this may require a separate cluster download and preprocessing job before the variant rerun.

### Blockers

- gnomAD missense data may not yet be staged in the shared HopTF area.
- MAVE/DMS/PBM coverage for the exact HopTF TF panel may be sparse.
- Isoform mapping must be exact; ambiguous mappings should be excluded rather than manually forced.
- TF-feature annotations may be incomplete, especially for non-DNA-binding regulatory regions.

### Clear Do-Not-Claim-Yet Caveats

- Do not claim a ClinVar or population-tolerated variant is molecularly neutral for TF response unless assay evidence supports that specific claim.
- Do not pool all benign/control tiers into one negative label without showing tier-specific behavior.
- Do not remove high-ESM-C-distance controls silently. Either exclude them for a recorded biological or evidence reason, or keep them in a named stress-test subset.
- Do not use ESM-C, AlphaMissense, conservation predictors, or HopTF itself to define benign ground truth. They can be annotations, filters, or stratification variables, but not the label source.

## 3. Narrow Missense Variant Sensitivity Redo

### Result Trigger

The 2026-05-17 matched variant run answered part of the May 16 follow-up plan, but not cleanly enough for a main biological claim.

Observed result:

- Pathogenic versus matched benign, using the previous absolute-change framing: `158` matched sets, median paired difference `-0.1654`, fraction pathogenic greater `0.481`, one-sided Wilcoxon `p = 0.7314`, gene-bootstrap 95% CI `[-0.608, 0.305]`.
- Pathogenic versus matched random, using the previous absolute-change framing: `189` matched sets, median paired difference `0.2928`, fraction pathogenic greater `0.5397`, one-sided Wilcoxon `p = 0.017`, gene-bootstrap 95% CI `[-0.184, 0.585]`.

Interpretation:

The matched benign comparison is not usable as primary evidence right now. This could be because ClinVar benign labels are heterogeneous for our purpose, because some "benign" variants are only benign in a clinical context that does not map onto TF-overexpression response, or because our matching is still insufficient. The benign/control guide now makes this more precise: clinical benignity, population tolerance, and assay-level molecular neutrality are different evidence types and should be tiered rather than pooled. The matched random comparison is still the more coherent primary control under the absolute-change framing, but the effect is modest and the bootstrap interval crosses zero. The next run should also test signed weakening because a pathogenic TF variant may impair function and reduce the predicted response.

### Definition To Use Going Forward

**Predicted perturbed-cell state activity**: the model-predicted TF-induced response in the current perturbed-cell response representation. In the current PCA-derived variant runs, this is computed from the frozen PCA-centroid transport checkpoint. Variant effects should be reported in two forms: signed mutant-minus-WT change, where the sign convention must state which direction means weaker response, and absolute WT-vs-mutant change, which measures disruption regardless of direction. This is a model output, not an experimental measurement of TF activity, binding, viability, or clinical effect.

### Question

Among high-confidence pathogenic or likely pathogenic single-residue TF missense substitutions, does HopTF predict a weaker TF-induced response or a larger response disruption than matched random single-residue substitutions with comparable ESM-C representation-space distance?

This is intentionally narrower than the previous benign-versus-pathogenic question. The goal is to test model sensitivity to small, interpretable TF sequence changes, not to claim clinical pathogenicity prediction. Evidence-tiered benign/control variants should be used as a secondary analysis after the curation setup in Experiment 2, not as the primary negative control.

The primary edit type should be held fixed as missense SNVs causing one amino-acid substitution. Matching should then use ESM-C representation-space distance as a separate model-space control, not as a proxy for nucleotide edit size.

### Paper Role

This is the main sequence-sensitivity experiment for the paper if it passes. It tests whether changing the TF amino-acid sequence changes the predicted TF-induced response in a way that is stronger than matched random substitutions. It should be written as a model-sensitivity and perturbation-response experiment, not as a clinical variant pathogenicity classifier.

Paper edit implied:

- Add a Results subsection titled "Missense variants perturb predicted TF response."
- Use two panels or tables: signed weakening and absolute disruption.
- Report matched random controls in the main text; report evidence-tiered benign/control subsets only if curation and matching diagnostics support them.
- Include a short paragraph explaining ESM-C distance as an embedding-space matching variable, not mutation size.

### Variant Inclusion Rules

Primary panel:

- ClinVar `Pathogenic` or `Likely pathogenic`.
- Missense SNVs only.
- One amino-acid substitution only.
- Exact reference amino-acid match to the local HopTF isoform.
- Local perturbation has enough recovered cells for the response representation being used.
- Exclude out-of-caliper variants whose ESM-C distance from wild type cannot be matched well. This is an embedding-space exclusion, not an edit-size exclusion.

Secondary benign/control panel:

- Use the evidence-tiered panel from Experiment 2.
- Keep clinical benign, population-tolerated, functional-neutral, and ortholog-supported synthetic controls separated by evidence tier.
- Use core and matched benign/control subsets for secondary comparisons only after exact isoform mapping and balance diagnostics pass.
- Keep high-ESM-C-distance benign/control variants in a named stress-test subset rather than silently discarding them or pooling them with ordinary negatives.
- Do not use benign/control variants as the primary negative control unless the matched benign/control subset is balanced against the pathogenic set by gene or TF family, domain class, DNA-binding-domain status, structured/disordered status, conservation, protein length, recovered cell count, and ESM-C-distance bin.

Important clarification:

Restricting to a few base pairs is not enough by itself, because the current ClinVar missense SNV panel is already mostly single-nucleotide substitutions that create one amino-acid change. The problem is not only nucleotide edit size. The problem is that a single amino-acid change can still be large in ESM-C representation space or biologically disruptive. The cleaner filter is: missense SNV, one amino-acid substitution, ESM-C-distance caliper, and domain/residue-class matching.

Updated interpretation of ESM-C distance:

A point mutation that evokes a large change in ESM-C space is not automatically a bad control, because large ESM-C distance can reflect departure from the protein language model's learned sequence manifold rather than a literal large edit or proven structural damage. It is also not automatically reassuring. For benign/control variants, high ESM-C distance should trigger stronger independent evidence requirements, manual biological review, and separate stress-test reporting. For random controls, ESM-C distance remains a matching variable so the pathogenic comparison is not driven by generic embedding-space movement.

### Controls

Primary control:

- Many matched random single-residue substitutions per pathogenic variant.

Matching variables:

- same TF isoform;
- same broad sequence region or protein-position bin;
- same domain class if domain annotations are available;
- similar ESM-C distance from wild type;
- similar mutant-residue class where possible.

Optional descriptive control:

- Evidence-tiered benign/control variants from Experiment 2, shown by tier and subset. Do not collapse clinical benign, population-tolerated, functional-neutral, and high-ESM-C-distance stress-test controls into a single negative label.

### Outputs

- `narrow_pathogenic_variant_panel.csv`: primary pathogenic panel with exclusion reasons.
- `narrow_random_controls.csv`: many matched random substitutions per real variant.
- `narrow_variant_activity_results.csv`: WT and mutant predicted activity values.
- `narrow_variant_activity_differences.csv`: WT-vs-mutant activity changes.
- `narrow_variant_validation_summary.md`: matched tests, caliper diagnostics, benign/control tier diagnostics, and excluded-variant table.

### Plots

1. **Signed variant effect violin plot**
   - x-axis: group: pathogenic real, matched random controls, evidence-tiered benign/control subsets if available.
   - y-axis: signed mutant-minus-WT predicted perturbed-cell state activity change, with negative values defined as weaker predicted response than WT.
   - Use violin plots with overlaid medians and points for real pathogenic variants.
   - Reason: this directly tests the biologically plausible loss-of-function pattern where pathogenic variants weaken TF response.

2. **Absolute variant effect violin plot**
   - x-axis: group: pathogenic real, matched random controls, evidence-tiered benign/control subsets if available.
   - y-axis: absolute WT-vs-mutant predicted perturbed-cell state activity change.
   - Use violin plots with overlaid medians and points for real pathogenic variants.
   - Reason: this tests broader disruption regardless of whether the variant weakens or strengthens the predicted response. It also fixes the readability issues in the previous paired-difference plot.

3. **Per-variant real versus random quantile plot**
   - x-axis: pathogenic variant, ordered by the real variant's percentile among matched random controls.
   - y-axis: percentile of the real variant effect within its matched random-control distribution.
   - Make separate panels for signed weakening and absolute disruption.
   - Add horizontal lines at 0.5, 0.9, and 0.95.
   - Concern diagnosed: if most real variants sit near the middle of their matched random distributions, the model is not detecting variant-specific effects beyond generic substitution effects.

4. **Random-control p-value diagnostic**
   - x-axis: empirical p-value for the relevant matched-control alternative.
   - y-axis: number of variants.
   - Make separate panels for the signed weakening alternative and the absolute disruption alternative.
   - Also plot p-values against ESM-C distance.
   - Concern diagnosed: the previous p-value distribution looked troubling. This plot asks whether the per-variant test is underpowered, poorly calibrated, or dominated by ESM-C representation-space distance.

5. **ESM-C distance caliper plot**
   - x-axis: ESM-C distance from wild type.
   - y-axis: signed predicted activity change and absolute predicted activity change in separate panels.
   - color: pathogenic real, matched random, evidence-tiered benign/control subset.
   - Mark the primary inclusion caliper.
   - Concern diagnosed: high ESM-C-distance variants may dominate the apparent effect. These are not necessarily large sequence edits; they are single-residue substitutions that look large to ESM-C and should be excluded or separately analyzed if they cannot be matched.

### Statistical Tests

Primary biological direction test:

- One-sided paired Wilcoxon signed-rank test comparing each pathogenic variant's signed mutant-minus-WT effect to the median signed effect of its matched random-control distribution. Define the sign so negative means weaker predicted response than WT, and use the one-sided alternative that real pathogenic variants are more negative than matched random controls.

Why this test:
Each pathogenic variant has its own control distribution, so the comparison must be within matched sets. The signed weakening test matches the biology of many pathogenic TF variants: impaired DNA binding or impaired regulatory function may reduce the predicted response. The Wilcoxon test does not assume normally distributed paired differences.

Secondary disruption test:

- One-sided paired Wilcoxon signed-rank test comparing each pathogenic variant's absolute WT-vs-mutant effect to the median absolute effect of its matched random-control distribution.

Why this test:
Some pathogenic variants may alter the response without simply weakening it. The absolute test asks whether real pathogenic variants are more disruptive than matched random substitutions regardless of direction.

Per-variant diagnostic:

- Empirical per-variant p-values against each variant's matched random-control distribution, reported as descriptive evidence rather than as the main result.

Why this test:
This asks whether a specific variant is unusually weakening or unusually disruptive compared with random substitutions in the same protein. It is sensitive to the number and quality of random controls, so it should not be overinterpreted if the distribution is poorly calibrated.

Uncertainty estimate:

- Bootstrap confidence interval resampling genes.

Why this test:
Multiple variants from the same TF are not independent. Gene-level bootstrapping gives a more honest uncertainty estimate.

### Success Criteria

- Median real-minus-random signed effect is negative under the weaker-response sign convention, or median real-minus-random absolute effect is positive for the broader disruption test.
- Gene-bootstrap confidence interval supports the relevant direction or mostly supports it with a clearly stated caveat.
- Real pathogenic variants are enriched in the lower tail of the matched random signed-effect distribution for weakening, or above the 90th/95th percentile of the matched random absolute-effect distribution for disruption.
- The signal persists after excluding unmatched high ESM-C-distance variants.

### Decisions From User Feedback

- Primary inference should use pathogenic or likely pathogenic variants versus matched random controls.
- Benign/control variants should be evidence-tiered, not defined as every ClinVar variant lacking a pathogenic or VUS label.
- Clinical benign, population-tolerated, functional-neutral, and ortholog-supported synthetic controls should be reported separately before any pooled analysis.
- High-ESM-C-distance benign/control variants should be stress-test controls with stronger independent evidence requirements, not automatic exclusions and not ordinary negatives.
- Choose a sensible fixed ESM-C-distance caliper after a quick distribution sanity check. If no obvious fixed threshold is stable across the panel, use matched-control feasibility and report the resulting included/excluded variants.

## 4. Variant Retrieval Example Panels

### Result Trigger

The matched retrieval-change run was stronger for pathogenic versus matched random than for pathogenic versus matched benign:

- Pathogenic versus matched benign: `158` matched sets, median paired difference `4.34e-05`, one-sided Wilcoxon `p = 0.2478`.
- Pathogenic versus matched random: `189` matched sets, median paired difference `1.08e-04`, one-sided Wilcoxon `p = 2.62e-04`, with a gene-bootstrap CI that nearly touches zero.

This is useful as a retrieval diagnostic, and the last plot from the run was visually useful. The next step should make this concrete with examples.

### Question

For a few pathogenic variants with large retrieval shifts, what genomic regions gain or lose attention relative to wild type and matched controls?

### Paper Role

This is a visualization and interpretability bridge between the variant-sensitivity result and the biological-validation result. It should not be used as statistical proof. Its job is to show what the retrieval distribution is doing when a sequence variant changes the predicted response.

Paper edit implied:

- Add this as a figure in the retrieval-results section, after aggregate retrieval-change statistics.
- Caption should say "AlphaGenome-window attention" or "locus-window retrieval", not binding-site activity.
- The text should connect the figure to the claim that HopTF exposes an auditable intermediate representation.

### Proposed Example Design

Select two examples where:

- pathogenic mutant differs strongly from wild type;
- matched random or evidence-tiered benign/control sequence does not show the same shift;
- ESM-C distance is matchable and not an unmatched high-distance outlier;
- TF has interpretable motif or regulatory context if possible.

Selection must be deterministic. Rank all candidate examples before choosing figures, record the ranking table, and choose the top examples that pass QC rather than hand-picking visually dramatic cases after plotting. This prevents the figure from becoming anecdotal cherry-picking.

For each example, create three matched genome-track panels:

1. wild type TF sequence;
2. pathogenic mutant TF sequence;
3. matched random or evidence-tiered benign/control sequence.

### Plot

**Chromosome-binned AlphaGenome attention heatmap**

- Use an ideogram-like layout: draw each chromosome as a separate vertical bar, ordered by chromosome, similar to a genome-wide chromosome heatmap.
- Split each chromosome into bins that line up with the AlphaGenome locus bins used by HopTF. If multiple AlphaGenome loci map into a display bin, aggregate by summed attention mass or maximum attention, and state which aggregation was used.
- y-axis: genomic coordinate in Mb, shared across chromosomes where possible.
- x-axis: chromosome labels.
- color: attention weight for the given sequence. Use a perceptually ordered scale, preferably with a transformed value such as `log10(attention + epsilon)` or attention-rank percentile so tiny raw attention values remain visible.
- panels: wild type, pathogenic mutant, matched control. The three panels must use the same chromosome layout and the same color scale.
- Add a fourth optional panel for `pathogenic mutant - WT` attention change if the absolute attention maps are too visually similar.
- Mark the highest-changing bins with small ticks or outlines, but do not label too many loci directly on the heatmap.

Reason:
The aggregate retrieval metrics show that the retrieval distribution moves, but they do not show what "moves" means biologically. A chromosome-binned AlphaGenome attention heatmap can show whether attention changes are diffuse, concentrated, chromosome-specific, or driven by a few AlphaGenome locus bins. Matching the display bins to the AlphaGenome locus bins is important because the figure should show the resolution at which HopTF actually retrieves, not imply motif-scale precision.

Pitfalls:

- AlphaGenome-key windows are large relative to motifs, so this should be described as locus-level retrieval movement, not binding-site movement.
- Attention values may be extremely small and need log scaling or rank-based coloring.
- Chromosome lengths differ, so avoid a layout that makes shorter chromosomes look more important only because their bars are shorter or visually denser.
- If a color scale saturates on a few high-attention bins, include a capped scale or separate change panel so the rest of the genome remains interpretable.
- A visually striking example is not proof of global validity.

### Outputs

- `variant_retrieval_example_candidates.csv`: ranked candidate examples with selection metrics.
- `variant_retrieval_example_tracks.csv`: per-window attention values for selected examples.
- `variant_retrieval_example_chromosome_diagrams.png`: final example figure.
- `variant_retrieval_example_notes.md`: why each example was selected and what not to claim.

Selection metrics:

- pathogenic mutant versus WT retrieval Jensen-Shannon divergence;
- matched control versus WT retrieval Jensen-Shannon divergence;
- ratio or difference between those two divergences;
- ESM-C-distance match quality between pathogenic and control variants;
- predicted perturbed-cell state activity change for the same variants;
- whether the TF has motif, family, or external binding annotations available for interpretation.

### Minimal Question For User

4. Should the example control be the matched random substitution, the best available evidence-tiered benign/control variant, or both if available?

## 5. Beta Retrieval-Regime Finalization

### Result Trigger

The paper uses the Hopfield/associative-memory framing as a central conceptual contribution. That framing requires more than showing that a softmax layer exists. We need to show that the retrieval inverse-temperature parameter \(\beta\) controls the retrieval distribution in a predictable way and that the useful regime is not an arbitrary implementation detail.

The May 16 plan already identified beta sensitivity as one of the core experiments. The May 17 plan must retain it because `paper/HopTF_full_draft.tex` has a dedicated Results subsection, "Beta controls retrieval regime."

### Question

As \(\beta\) changes, does HopTF move from diffuse genome-wide averaging to concentrated AlphaGenome-window retrieval in a way that can be related to predicted perturbed-cell state performance and biological support?

### Paper Role

This experiment supports the associative-memory behavior claim. It should show that beta is an interpretable inverse-temperature parameter controlling retrieval concentration, not a hidden hyperparameter tuned only for performance.

Paper edit implied:

- Fill the "Beta controls retrieval regime" Results subsection.
- Use a three- or four-panel figure showing prediction quality, retrieval entropy or effective number of loci, top-k mass, and motif/binding support versus \(\beta\).
- Phrase beta as inverse temperature. Do not claim it is quantitatively equivalent to TF concentration unless dose-calibrated data are available.

### Inputs

- Fixed TF sequence embeddings and projected queries.
- Fixed AlphaGenome-key memory bank.
- The same perturbed-cell response representation used for the primary prediction result, with SCARF added later if available.
- Motif/binding support annotations from the redesigned biological-validation setup where available.
- A predetermined beta grid, such as low, intermediate, and high values spanning diffuse to concentrated retrieval.

### Controls

- Keep model checkpoint, TF panel, locus set, and response representation fixed across beta values.
- Include shuffled TF-query or random-memory controls for at least a subset of beta values if feasible.
- Use the same top-k thresholds and support definitions across beta values.
- Do not choose beta after looking only at motif enrichment; prediction quality and retrieval stability must also be reported.

### Outputs

- `beta_sensitivity_metrics.csv`: one row per beta value with prediction metric, retrieval entropy, effective number of loci, top-k mass, and motif/binding support summaries.
- `beta_sensitivity_per_tf.csv`: per-TF retrieval concentration and support metrics.
- `beta_sensitivity_summary.md`: recommended beta regime and claim boundary.
- `beta_sweep_figure.png`: final multi-panel paper figure.

### Required Plots

1. **Prediction quality versus beta**
   - x-axis: \(\beta\).
   - y-axis: primary perturbed-cell state prediction metric, such as grouped held-out-gene MSE.
   - Concern diagnosed: the retrieval regime that looks interpretable may hurt response prediction.

2. **Retrieval concentration versus beta**
   - x-axis: \(\beta\).
   - y-axis: retrieval entropy or effective number of loci.
   - Concern diagnosed: beta may not actually control retrieval concentration if scores are poorly scaled.

3. **Top-k mass versus beta**
   - x-axis: \(\beta\).
   - y-axis: fraction of retrieval mass in top-k AlphaGenome windows.
   - Concern diagnosed: high beta may collapse onto too few loci, while low beta may average over too much of the genome.

4. **Biological support versus beta**
   - x-axis: \(\beta\).
   - y-axis: precision@k or matched-background lift for same-TF, TF-family, or accessibility-supported motif/binding windows.
   - Concern diagnosed: sharper retrieval may not become more biologically plausible.

### Statistical Tests

- Spearman correlation between beta and retrieval concentration metrics.
- Paired bootstrap over TFs for prediction-metric differences between candidate beta values.
- Paired bootstrap over TFs for motif/binding support differences between candidate beta values.
- Stability analysis of top-k retrieved windows between adjacent beta values, measured by Jaccard overlap or rank correlation.

Why these tests:
The beta sweep is ordered, so Spearman correlation is a direct diagnostic of monotonic concentration. Prediction and biological-support differences should be bootstrapped over TFs because TFs are the relevant repeated units. Stability analysis checks whether nearby beta values produce a coherent retrieval regime rather than erratic top-k changes.

How to interpret:
The ideal result is an intermediate beta regime where retrieval is concentrated enough to be interpretable, stable across nearby beta values, not collapsed onto a tiny number of windows, and not worse for perturbed-cell state prediction. If prediction quality and biological support favor different beta values, report the tradeoff explicitly rather than choosing whichever result looks best.

### Success Criteria

- Retrieval entropy or effective number of loci decreases as beta increases.
- Top-k mass increases as beta increases.
- There is at least one intermediate beta regime with stable top-k retrieval and acceptable prediction quality.
- Biological support does not collapse to random in the beta regime used for paper figures.

### Clear Do-Not-Claim-Yet Caveats

- Do not claim beta is measured TF concentration.
- Do not present beta sensitivity as biological validation by itself.
- Do not select beta solely from motif or binding support after seeing the results.
- Do not hide a prediction-versus-interpretability tradeoff if one appears.

### Minimal Question For User

5. If SCARF results are not ready in time, should the beta figure use the PCA-derived response representation as the main metric and mark SCARF beta sensitivity as future work?

## 6. Biological Validation Redesign Around Resolution Limits

### Result Trigger

The motif and binding follow-up results are not strong positive biological validation.

Top-ranked support checks:

- Exact same-symbol JASPAR motif support at top 1000: median `5.9%` in top loci versus `5.3%` matched background.
- Exact same-TF ReMap peak overlap among top-locus examples: `11.5%`.
- Broad cCRE overlap among top-locus examples: `80.0%`.

Full recall checks at top 1000:

- Exact same-symbol JASPAR motif median recall: `1.77%`, random expectation `1.82%`, lift `0.97x`.
- Exact same-TF ReMap peak median recall: `1.13%`, random expectation `1.82%`, lift `0.62x`.
- JASPAR motif or ReMap peak median recall: `1.47%`, random expectation `1.82%`, lift `0.81x`.

Interpretation:

This is ambiguous at best. It does not show that HopTF recovers a high fraction of known supported regulatory sites. The safest useful claim is that exact motif enrichment is slightly above matched background in some precision-style checks, but full recall over known supported sites is weak. This suggests the current AlphaGenome-key retrieval setup is not as locus-specific as we want.

### Resolution Caveat To State Clearly

TF motifs are short DNA patterns, often roughly 6-20 base pairs. The current AlphaGenome-key windows are much larger genomic units; in the current motif runs these are TSS-centered windows derived from the AlphaGenome-key metadata rather than motif-scale windows. A large window can contain many sequence features and regulatory annotations, so exact motif recovery is a blunt test. This mismatch can dilute motif signal even if the model is retrieving broadly relevant regulatory neighborhoods.

This caveat should not be used to excuse a failed validation. It should define the next experiment: make the biological validation match the resolution of the retrieval object.

### Question

At the AlphaGenome-window scale used by HopTF, does retrieval prioritize broad regulatory neighborhoods for the queried TF or TF family better than matched random rankings?

### Paper Role

This is the biological plausibility test for the retrieval distribution. The paper should treat current results as weak and diagnostic unless redesigned family-level, accessibility-supported, or binding-supported analyses improve. The strongest honest story right now is that retrieval is auditable but not yet strongly biologically validated.

Paper edit implied:

- Replace old motif-enrichment placeholders with a subsection that reports the current weak exact-symbol recall results and explains why this motivates family-level and accessibility-supported validation.
- Avoid the word "cognate" unless explicitly defined. Prefer "same TF motif", "TF-family motif", or "motif/binding-supported AlphaGenome window."
- Add a limitation that AlphaGenome-key windows are much larger than 6-20 bp TF motifs, so motif recall is a resolution-mismatched validation.

### Redesign

Replace confusing top-locus overlap plots with clearer precision/recall panels:

1. precision at top-k for same-TF motif, TF-family motif, same-TF ReMap, TF-family ReMap, and cCRE-supported motif;
2. recall at top-k using the same support definitions;
3. rank-percentile distribution of supported loci;
4. background-matched enrichment with a clear denominator.

### Definitions

**Motif/binding-supported AlphaGenome window**: an AlphaGenome-key window counted as independently supported for a queried TF if the window overlaps at least one of the following external evidence types after mapping all coordinates to the same genome build:

- a sequence motif match for the same TF;
- a sequence motif match for the TF's annotated family;
- a ReMap ChIP-seq peak assigned to the same TF;
- a ReMap ChIP-seq peak assigned to a TF in the same family;
- an accessibility-supported motif, meaning a same-TF or TF-family motif match inside a window that also overlaps an accessible regulatory annotation such as cCRE or the chosen accessibility mask.

This definition is window-level support, not base-pair-level binding. A supported window is evidence that the retrieved genomic neighborhood contains plausible TF regulatory evidence. It is not evidence that the exact retrieved position is bound in the TF Atlas cell state.

### Plot To Replace

The plot `top_loci_remap_overlap.png` is currently confusing. The labels "Top-k source" and "TF ReMap peak" do not explain the test.

Replace with:

**Independent binding support among top-ranked loci**

- x-axis: support definition: exact same-TF ReMap, TF-family ReMap, same-TF motif, TF-family motif, cCRE-supported motif.
- y-axis: fraction of top-k AlphaGenome windows with support.
- facets or color: top-k cutoff.
- baseline: matched background fraction.

Definitions:

- ReMap peak: an externally curated ChIP-seq peak from ReMap assigned to a TF. It is independent binding evidence aggregated across cell types, not binding in the TF Atlas perturbation cell state.
- Top-k source: remove this phrase from the plot. Use "HopTF top-k windows" and "matched background windows" instead.

### Follow-Up Data Or Annotation Improvements

- Add TF-family motif mappings instead of exact TF-symbol matching only.
- Add TF aliases and motif aliases.
- Add TF-family ReMap peak support if exact TF support is too sparse.
- Add cell-type labels for ReMap peaks so exact overlap is not treated as cell-type matched evidence.
- Consider smaller sequence windows around the highest-scoring positions inside each AlphaGenome-key window if the sequence data support it.

### Success Criteria

- Precision at top-k exceeds matched background for TF-family or accessibility-supported motif targets.
- Recall at top-k is above random-rank expectation for at least one biologically coherent support definition.
- Supported-locus rank distributions are shifted toward high HopTF ranks.

### Statistical Tests

- Hypergeometric or permutation enrichment test comparing supported-window counts in HopTF top-k windows against matched background windows.
- Paired bootstrap over TFs for the difference in precision@k between HopTF rankings and matched background rankings.
- Wilcoxon signed-rank test over TFs for whether supported windows have better HopTF rank percentiles than matched random windows.
- Multiple-testing correction across support definitions and top-k thresholds.

Why these tests:
The enrichment test asks whether the number of supported windows in the top-k set is higher than expected under the chosen background. The paired bootstrap and Wilcoxon tests treat TFs as the unit of replication rather than treating thousands of windows as independent observations. Multiple-testing correction is required because the analysis evaluates several support definitions and top-k values.

How to interpret:
If exact same-TF support remains weak but TF-family or accessibility-supported windows improve over matched background, the paper can claim broad biological plausibility at the AlphaGenome-window scale. If all support definitions are near random, the paper should state that retrieval is interpretable in form but not yet biologically validated.

### Minimal Question For User

6. Should the paper frame the current motif/binding result as weak-but-directional support, or mainly as evidence that the current AlphaGenome-window resolution is too coarse for strong motif recovery?

## 7. Isoform Domain Annotation Audit

### Result Trigger

The domain-aware isoform diagnostic found:

- `246` pairs with annotation unavailable.
- `9` pairs with no annotated domain presence difference.

This is not enough to conclude that natural isoform domain differences do not matter. It may mostly show that the current UniProt feature-mapping workflow is too conservative or pointed at the wrong annotation source.

### Question

Can we obtain enough reliable domain annotations for TF isoforms to make a domain-aware natural isoform test meaningful?

### Paper Role

This should remain a diagnostic unless the annotation audit substantially improves coverage. The current negative isoform result mostly says that conservative UniProt feature mapping did not produce enough interpretable domain-altered pairs.

Paper edit implied:

- Put current isoform results in the appendix or limitations, not the main empirical story.
- If the audit improves coverage, add a short Results subsection on natural isoform retrieval only if it yields a clear domain-aware comparison.
- If coverage remains poor, use it to motivate better TF isoform annotation rather than claiming that isoform domain differences do not matter.

### Proposed Follow-Up

Audit annotation sources before rerunning the isoform experiment:

- UniProt reviewed entries, checking whether isoform-specific features are being retrieved correctly.
- InterPro domain assignments.
- Pfam domain hits.
- Ensembl/GENCODE protein feature annotations if available.
- DNA-binding-domain annotations from TF-specific resources such as Lambert TF families or AnimalTFDB, if available.

Also audit the mapping code:

- Are local HopTF isoform IDs mapped to UniProt accessions correctly?
- Are canonical UniProt coordinates being incorrectly applied to noncanonical isoforms?
- Are isoform-specific feature tables being dropped?
- Are feature names being filtered too aggressively?

### Outputs

- `isoform_annotation_source_audit.md`: source availability and mapping failure reasons.
- `isoform_domain_annotation_coverage.csv`: coverage by source and by TF family.
- `domain_aware_isoform_redesign_decision.md`: whether to rerun, drop, or keep as diagnostic.

### Success Criteria

- A meaningful number of same-gene isoform pairs can be classified as DNA-binding-domain altered, activation/repression-domain altered, or domain-preserving.
- Mapping failures are explainable rather than silently treated as missing biology.

### Minimal Question For User

7. Should this remain a diagnostic only unless annotation coverage improves, or do we need a domain-aware isoform result for the main report?

## 8. Sequence Encoder Appendix Diagnostic

### Result Trigger

The ESM-C versus ESM-DBP recheck is clean enough and does not need more follow-up.

Result:

- ESM-C projected into AlphaGenome key space beats ESM-DBP projected into AlphaGenome key space across overall, length-matched, cell-count-matched, and length-plus-cell-count matched settings.
- The bootstrap confidence intervals for ESM-DBP-minus-ESM-C MSE are positive across the reported settings, meaning ESM-C has lower error.

Interpretation:

This answers a likely reviewer question: a DBP-specialized encoder did not improve this HopTF objective. It does not help or hurt the main argument enough to be a central result.

### Plan

No follow-up experiment unless the writing needs a cleaner appendix figure.

Outputs to use:

- `tmp/hoptf_followup_full_20260517/sequence_representation_recheck/sequence_representation_recheck_summary.md`
- `tmp/hoptf_followup_full_20260517/sequence_representation_recheck/sequence_representation_recheck_table.csv`
- `tmp/hoptf_followup_full_20260517/sequence_representation_recheck/sequence_representation_gene_mse_differences.png`

### Paper Placement

Appendix or short ablation note. Do not make it part of the main empirical story.

Paper edit implied:

- Keep the ESM-C versus ESM-DBP table in the ablation section or appendix.
- Use the result to answer a likely reviewer question: the DBP-specialized encoder does not outperform ESM-C for this objective.
- Do not let this distract from the main retrieval and validation story.
