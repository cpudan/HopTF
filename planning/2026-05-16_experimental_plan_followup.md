# HopTF Follow-Up Experimental Plan After 2026-05-16 Results

## Executive Summary

The current results support a narrower and more defensible next run than the original experiment plan. The strongest result is not the broad ClinVar benign-versus-pathogenic comparison by itself. The strongest result is the matched random-substitution control: pathogenic and likely pathogenic variants changed predicted perturbed-cell state activity and AlphaGenome-key retrieval more than matched random substitutions, while benign and likely benign variants did not.

The next run should therefore be built around matched comparisons. The primary question is:

> Do high-confidence pathogenic TF missense variants change predicted perturbed-cell state activity more than carefully matched benign variants and matched random substitutions, after controlling for the size of the sequence change in ESM-C embedding space?

The biological validation question should also be sharpened. The previous motif analysis asked whether top-ranked loci were enriched for exact TF-symbol JASPAR motif matches. That is useful but weak. The next analysis should ask:

> Among loci with known TF motif or binding evidence, does HopTF rank a high fraction of that evidence near the top, and does it do so more than matched background rankings?

The natural isoform analysis should not be part of the main claim in its current broad form. The negative result means that arbitrary same-gene isoform sequence differences do not reliably produce retrieval differences that track observed response-score differences. That is a useful diagnostic, but it is not strong evidence against the variant result because the isoform analysis mixes many biological cases. Keep it as a diagnostic or redesign it around domain-altering isoforms only.

## Key Definitions

**Predicted perturbed-cell state activity**: the scalar or vector summary produced by the current HopTF evaluator for the predicted TF-perturbed cell state. In the current variant outputs this is summarized as mutant-minus-wild-type percent change. Larger absolute values mean the model predicts that the sequence change changes the perturbed-cell state more strongly.

**ESM-C distance from wild type**: the L2 distance between the ESM-C embedding of the wild-type TF sequence and the ESM-C embedding of the variant TF sequence. This is not a biological mutation severity score. It is a model-space measure of how large the protein sequence change looks to the encoder. It must be controlled because the current results show that larger ESM-C distance is strongly associated with larger predicted activity and retrieval changes.

**Motif-supported locus**: a HopTF candidate locus whose hg38 sequence window contains a statistically called motif match for the same TF or a TF-family member, using a named motif database, motif-to-TF mapping, score threshold, strand policy, and sequence window definition. The locus is motif-supported only with respect to the database and threshold used in that run.

**Binding-supported locus**: a HopTF candidate locus that overlaps an external ChIP-seq peak or curated binding peak assigned to the same TF or a TF-family member, after lifting or confirming coordinates to hg38 and recording the source cell type. Binding-supported does not mean the TF is bound in the exact perturbation cell state unless the external binding data are from that cell type or a close context.

**Motif/binding-supported locus**: a locus that is motif-supported, binding-supported, or both. Reports must keep the support type visible as separate columns: `motif_supported`, `binding_supported`, and `motif_or_binding_supported`. Do not collapse these into one label without showing which evidence type produced the support.

## Experiment List, Ordered By Priority

1. Variant validation with matched benign and matched random-substitution controls.
2. Retrieval-change analysis on the matched variant design.
3. Motif and binding evidence capture for retrieved loci.
4. Natural isoform diagnostic or domain-aware redesign.
5. Sequence representation ablation recheck for ESM-C versus ESM-DBP.

## 1. Variant Validation With Matched Controls

### Question

Do high-confidence pathogenic TF missense variants change predicted perturbed-cell state activity more than matched benign variants and matched random substitutions?

This experiment should answer whether TF sequence helps predict perturbed-cell state in a way that is aligned with credible variant biology, not merely whether the model reacts more to larger ESM-C embedding changes.

### Inputs

- ClinVar variant table, frozen with download date and ClinVar release date.
- Local wild-type TF isoform sequences used by HopTF.
- Coordinate-checked missense substitutions only.
- ESM-C embeddings for wild-type, real variant, and random-substitution sequences.
- Domain annotations from UniProt, Pfam, or InterPro, with DNA-binding domains flagged separately.
- Existing HopTF evaluator for predicted perturbed-cell state activity.

### Inclusion And Exclusion Rules

Use only variants that can be mapped unambiguously to the exact HopTF isoform sequence. Exclude variants when the reference amino acid does not match the local sequence, when the variant affects more than one residue, or when the variant creates a stop codon, frameshift, splice change, insertion, deletion, or complex replacement. Those variants may be biologically important, but they answer a different question from single-residue missense sensitivity.

For pathogenic labels, include only `Pathogenic` or `Likely pathogenic`. Prefer variants with at least one review star, and report a stricter sensitivity subset using only variants with at least two review stars or practice guideline/reviewed-by-expert status.

For benign labels, include only `Benign` or `Likely benign`. Exclude variants with conflicting interpretations, uncertain significance, risk factor only, association only, drug response only, protective only, or other non-disease-causality categories.

Exclude out-of-caliper variants from the primary matched analysis. A variant is out of caliper if its ESM-C distance from wild type cannot be matched to controls within the pre-specified distance tolerance. The NOTCH2 Q1656R issue falls here: if a benign variant is very large in ESM-C space, then a paired "benign version" is not a meaningful control for ordinary benign missense variation. Such points can be listed in an outlier appendix, but they should not anchor the primary benign-versus-pathogenic claim.

### Controls

For each pathogenic variant, build a matched control set:

- Matched benign variants from the same gene if available.
- Matched benign variants from the same domain class if same-gene matches are unavailable.
- Matched random substitutions on the same wild-type sequence.

Matching variables should be applied in this order:

1. Same gene when possible.
2. Same broad domain class: DNA-binding domain, other annotated domain, outside annotated domain.
3. Similar ESM-C distance from wild type, using a fixed caliper defined before testing.
4. Similar residue class change, such as charged to charged, hydrophobic to hydrophobic, or polar to polar.
5. Similar protein length and recovered-cell count when comparing across genes.

The random substitutions are not meant to represent benign biology. They answer a narrower question: would a random substitution of similar size in the same protein produce a similar model change?

### Exact Outputs

- `matched_variant_panel.csv`: one row per real variant with clinical label, review status, gene, isoform, protein position, domain class, wild-type amino acid, mutant amino acid, ESM-C distance, and match status.
- `matched_variant_controls.csv`: one row per matched control, linked to the pathogenic variant it controls.
- `variant_activity_results.csv`: predicted perturbed-cell state activity for wild type, real variant, and controls.
- `variant_activity_differences.csv`: real-minus-wild-type and control-minus-wild-type changes.
- `variant_validation_summary.md`: sample sizes, exclusions, primary tests, sensitivity tests, and outlier table.

### Plots And Axes

#### 2026-05-17 Plot Update

For the matched-control plot variants requested after reviewing Experiment 1, the paired difference should be treated as an absolute value. The useful distinction here is "little activity-change difference" versus "large activity-change difference"; the sign of `pathogenic - matched control` is not the main question for these views.

Use:

`absolute_activity_change_delta_pct = abs(pathogenic_abs_activity_change_pct - control_median_abs_activity_change_pct)`

Updated plot outputs are in:

`tmp/hoptf_followup_full_20260517/results/matched_variant_retrieval/absolute_delta_plots/`

The generated files are:

- `matched_activity_change_lines_colored_by_abs_delta.png`
- `absolute_activity_delta_distributions_with_thresholds.png`
- `matched_activity_change_by_abs_delta_mode_4panel.png`
- `absolute_delta_mode_thresholds.json`
- `variant_activity_differences_with_absolute_delta.csv`
- `absolute_delta_plot_update.md`

Automatic thresholding was rerun on the absolute paired differences. The called thresholds are 2.86% for matched benign controls and 2.55% for matched random controls. These split the matched benign comparison into 113 little-difference rows and 45 large-difference rows, and the matched random comparison into 147 little-difference rows and 42 large-difference rows.

1. **Matched activity-change plot**
   - x-axis: variant/control group: pathogenic real, matched benign, matched random.
   - y-axis: absolute predicted perturbed-cell state activity change from wild type.
   - Show paired lines from each pathogenic variant to its matched controls.
   - Diagnostic concern: this checks whether pathogenic variants still look larger after matching, rather than only in an unmatched class comparison.

2. **ESM-C distance versus activity-change plot**
   - x-axis: ESM-C distance from wild type.
   - y-axis: absolute predicted perturbed-cell state activity change from wild type.
   - Color: pathogenic, benign, random.
   - Add vertical lines for the primary matching caliper and any exclusion threshold.
   - Diagnostic concern: this checks whether the apparent biological signal is mostly explained by the size of the sequence change in ESM-C space.

3. **Matched real-minus-control difference plot**
   - x-axis: matched comparison type: pathogenic minus benign, pathogenic minus random.
   - y-axis: paired difference in absolute predicted activity change.
   - Add a horizontal line at zero.
   - Diagnostic concern: this shows whether the conclusion depends on group medians or whether most matched pathogenic variants exceed their own controls.

4. **Out-of-caliper variant plot**
   - x-axis: ESM-C distance percentile within all candidate variants.
   - y-axis: absolute predicted activity change.
   - Label excluded high-distance benign variants and other influential points.
   - Diagnostic concern: this identifies variants like NOTCH2 Q1656R that are too large in model-space distance to serve as ordinary benign controls.

### Statistical Tests

Primary test: paired Wilcoxon signed-rank test comparing each pathogenic variant's absolute activity change against the median absolute activity change of its matched controls.

Why this test: each pathogenic variant has its own matched control set, so the comparison should be within matched sets. The Wilcoxon signed-rank test asks whether the paired differences tend to be greater than zero without assuming the differences are normally distributed.

How to interpret it: a small one-sided p-value means pathogenic variants usually change predicted perturbed-cell state activity more than their matched controls. It does not prove clinical pathogenicity; it supports the narrower claim that high-confidence pathogenic substitutions produce larger model changes than matched alternatives.

Secondary test: empirical p-value for each pathogenic variant against its matched random substitutions.

Why this test: it asks whether a specific real variant is more disruptive than random substitutions generated under the same matching rules.

How to interpret it: if a pathogenic variant is above the 95th percentile of its matched random controls, it is unusually disruptive under that control design. The fraction of pathogenic variants passing this criterion is an interpretable summary.

Uncertainty estimate: bootstrap confidence intervals resampling genes, not individual variants.

Why this test: variants from the same gene are not independent. Resampling by gene avoids overstating precision when many variants come from the same TF.

How to interpret it: if the bootstrap confidence interval for the median paired difference is above zero, the result is more stable than a single p-value.

Sensitivity analysis: regression of activity change on clinical label, ESM-C distance, domain class, protein length, and recovered-cell count, with gene-clustered standard errors if possible.

Why this test: matching is the primary design, but regression checks whether the label association remains after accounting for measured variables.

How to interpret it: if the pathogenic-label coefficient remains positive after ESM-C distance adjustment, that supports a biological-label signal beyond perturbation size. If it disappears, the paper should say the effect is largely explained by ESM-C distance.

Descriptive metric: AUROC for separating pathogenic from benign variants using activity change.

Why this metric: AUROC summarizes class separation, but it ignores matching and should not be the primary evidence.

How to interpret it: AUROC above 0.5 means pathogenic variants tend to have larger activity changes than benign variants. It is not sufficient if the matched tests fail.

### Success Criteria

- Primary matched pathogenic-versus-control paired difference is positive with a bootstrap confidence interval above zero.
- One-sided paired Wilcoxon p-value remains small after the pre-specified multiple-testing correction.
- The result remains directionally similar in the stricter ClinVar review-status subset.
- The effect is not eliminated after restricting to the ESM-C distance-matched subset.

### Expected Runtime

Moderate. Most runtime is embedding variant and random-substitution sequences with ESM-C and running the existing HopTF evaluator. If embeddings are cached, statistics and plotting should be short.

### Blockers

- Too few same-gene benign matches.
- Too few high-confidence pathogenic variants after review-status filtering.
- ESM-C embedding generation for many matched random substitutions.
- Domain annotation mismatch between isoform coordinates and UniProt/Pfam/InterPro coordinates.

### Do Not Claim Yet

Do not claim that HopTF predicts clinical pathogenicity. The intended claim is narrower: under matched controls, high-confidence pathogenic missense variants produce larger HopTF-predicted perturbation changes than matched alternatives.

## 2. Retrieval-Change Analysis On The Matched Variant Design

### Question

When a variant changes predicted perturbed-cell state activity, does it also change the AlphaGenome-key retrieval distribution more than matched controls?

This experiment is a follow-up to Experiment 1. It should not be interpreted independently until ESM-C distance is controlled.

### Inputs

- The matched variant and control tables from Experiment 1.
- Wild-type, real variant, benign control, and random-substitution ESM-C embeddings.
- HopTF query projections.
- AlphaGenome-key retrieval weights and retrieved vectors for each sequence.

### Controls

Use exactly the same matched sets as Experiment 1. Do not introduce a separate unmatched retrieval comparison as primary evidence.

### Exact Outputs

- `matched_retrieval_changes.csv`: one row per sequence pair with query L2, attention Jensen-Shannon divergence, retrieved-vector L2, top-100 overlap, top-500 overlap, attention entropy change, ESM-C distance, and predicted activity change.
- `retrieval_activity_joined.csv`: matched retrieval metrics joined to predicted activity metrics.
- `retrieval_change_summary.md`: primary matched tests, ESM-C distance diagnostics, and metric interpretation.

### Metrics To Keep

- Query L2: change after projecting the TF sequence embedding into the HopTF query space.
- Attention Jensen-Shannon divergence: change in the full retrieval-weight distribution.
- Retrieved-vector L2: change in the AlphaGenome-key-weighted retrieved representation.
- Top-100 and top-500 overlap: how many highly ranked loci remain highly ranked.
- Attention entropy change: whether the variant makes retrieval more or less concentrated.

### Plots And Axes

1. **Matched retrieval-change plot**
   - x-axis: variant/control group: pathogenic real, matched benign, matched random.
   - y-axis: attention Jensen-Shannon divergence from wild type.
   - Show paired lines for matched sets.
   - Diagnostic concern: this checks whether retrieval changes are larger for pathogenic variants after the same matching used for predicted activity.

2. **Retrieval change versus ESM-C distance plot**
   - x-axis: ESM-C distance from wild type.
   - y-axis: attention Jensen-Shannon divergence.
   - Color: pathogenic, benign, random.
   - Diagnostic concern: the previous run showed retrieval change was strongly correlated with ESM-C distance. This plot diagnoses whether retrieval metrics still mostly measure sequence perturbation size.

3. **Retrieval change versus activity change plot**
   - x-axis: attention Jensen-Shannon divergence.
   - y-axis: absolute predicted perturbed-cell state activity change.
   - Color: clinical label or control type.
   - Diagnostic concern: this asks whether retrieval changes are connected to predicted perturbed-cell state activity, rather than being isolated attention movement.

4. **Top-k overlap plot**
   - x-axis: variant/control group.
   - y-axis: top-100 or top-500 overlap with wild-type retrieval.
   - Diagnostic concern: this makes the retrieval result easier to read biologically: lower overlap means the variant changes which loci are prioritized.

### Statistical Tests

Primary test: paired Wilcoxon signed-rank test on each retrieval metric, comparing pathogenic real variants against their matched controls.

Why this test: retrieval metrics are skewed and matched by design. The test asks whether retrieval changes are consistently larger for real pathogenic variants than for matched controls.

How to interpret it: a significant positive result supports the claim that sequence changes propagate through the retrieval mechanism. It should be reported only with the matching variables and ESM-C distance diagnostics.

Correlation test: Spearman rank correlation between retrieval change and ESM-C distance, and between retrieval change and activity change.

Why this test: Spearman correlation checks monotone association without assuming linearity. It is diagnostic, not the primary hypothesis test.

How to interpret it: strong correlation with ESM-C distance means retrieval change may mostly reflect sequence perturbation size. Correlation with activity change supports a link between retrieval movement and predicted cell-state effect.

Sensitivity analysis: repeat paired tests within a narrow ESM-C distance band.

Why this test: it directly checks whether retrieval differences persist when sequence perturbation size is similar.

How to interpret it: if retrieval effects disappear within the distance band, the paper should not present retrieval change as clean biological validation.

### Success Criteria

- Pathogenic real variants have larger retrieval changes than matched controls under the same matched design as Experiment 1.
- The effect remains directionally positive within ESM-C distance bands.
- Retrieval change is associated with predicted activity change, not only ESM-C distance.

### Expected Runtime

Low to moderate if retrieval weights can be computed from cached embeddings and existing AlphaGenome keys. Higher if full attention distributions must be materialized for many random controls.

### Blockers

- Memory cost of storing full attention distributions.
- Need for approximate top-k retrieval if full distributions are too large.
- Interpretation remains blocked if Experiment 1 matching fails.

### Do Not Claim Yet

Do not claim that changed retrieval weights identify causal binding sites. The defensible claim is that matched pathogenic variants produce larger changes in the model's retrieved regulatory representation.

## 3. Motif And Binding Evidence Capture For Retrieved Loci

### Biological Question

The main question is not simply whether top-ranked loci are enriched for motifs. The stronger question is:

> Does HopTF assign high ranks to loci that already have independent motif or binding evidence for the queried TF or TF family?

This is a ranking and evidence-capture question. A useful model should place more known motif/binding-supported loci near the top of its retrieval ranking than matched background rankings do.

### Validation Targets

Use several target definitions and report them separately:

- Same-TF motif target: motif matches assigned directly to the queried TF symbol.
- TF-family motif target: motif matches assigned to TFs in the same family as the queried TF.
- Same-TF ChIP target: ChIP-seq peaks assigned directly to the queried TF.
- TF-family ChIP target: ChIP-seq peaks assigned to TF-family members.
- Accessibility-supported motif target: motif-supported loci that also overlap ATAC-seq, DNase-seq, or cCRE regulatory annotations in a relevant or broadly permissive cell context.
- Combined motif/binding target: loci meeting any of the motif or binding definitions above, while preserving columns for each support type.

### Inputs

- HopTF ranked AlphaGenome-key loci for each TF.
- hg38 genome FASTA and `.fai`.
- AlphaGenome-key coordinates, already confirmed against GRCh38/hg38.
- JASPAR CORE vertebrate motifs.
- Optional second motif database such as HOCOMOCO or CIS-BP for sensitivity.
- TF-to-motif mapping with synonyms resolved.
- TF-family mapping.
- ENCODE or ChIP-Atlas ChIP-seq peaks for TFs and TF-family members, preferably in H1 hESC or a close stem-cell context if available.
- ENCODE cCRE, ATAC-seq, or DNase-seq tracks for accessibility-supported analyses.
- Matched background loci by GC content, mappability if available, locus length, promoter/enhancer annotation, and chromosome when feasible.

### Controls

- Matched random locus rankings.
- Shuffled TF-to-ranking assignments.
- Mean-pooled or beta-low retrieval rankings if available.
- GC-, length-, and annotation-matched background windows.

### Exact Outputs

- `motif_binding_locus_support.csv`: one row per TF-locus pair with same-TF motif support, TF-family motif support, same-TF ChIP support, TF-family ChIP support, accessibility support, and combined support columns.
- `motif_binding_capture_by_k.csv`: recall and precision at top-k values for each evidence target.
- `motif_binding_rank_distributions.csv`: ranks of all supported loci for each TF and evidence target.
- `motif_binding_enrichment_tests.csv`: odds ratios and p-values for top-k versus matched background.
- `motif_binding_validation_summary.md`: evidence coverage, missing TFs, primary plots, and caveats.

### Primary Quantities

- Recall at k: among all motif/binding-supported loci for a TF, the fraction found in the top-k HopTF loci.
- Precision at k: among the top-k HopTF loci for a TF, the fraction that are motif/binding-supported.
- Rank distribution: the distribution of ranks assigned to motif/binding-supported loci.
- Top-k enrichment odds ratio: whether support is more frequent in top-k loci than in matched background loci.

Recall answers the user's central biological question: what fraction of known evidence does the model capture? Precision answers a different question: how much of the top-ranked list is already supported by known evidence? Both are needed because a TF can have many known motif instances, and top-k precision can be low even if ranking is better than background.

### Plots And Axes

1. **Evidence recall curve**
   - x-axis: top-k cutoff, for example 10, 25, 50, 100, 250, 500, 1000.
   - y-axis: recall of motif/binding-supported loci.
   - Separate lines: same-TF motif, TF-family motif, same-TF ChIP, TF-family ChIP, accessibility-supported motif.
   - Diagnostic concern: this directly checks whether HopTF captures a high fraction of known evidence as k increases.

2. **Precision at top-k plot**
   - x-axis: top-k cutoff.
   - y-axis: fraction of top-k loci that are motif/binding-supported.
   - Include matched random-ranking baseline.
   - Diagnostic concern: this checks whether top-ranked loci are dominated by unsupported windows even when recall looks acceptable.

3. **Supported-locus rank distribution**
   - x-axis: HopTF rank percentile, with lower percentiles meaning higher retrieval rank.
   - y-axis: density or cumulative fraction of motif/binding-supported loci.
   - Include random-ranking expectation.
   - Diagnostic concern: this shows whether supported loci are globally shifted toward high ranks or whether only a few examples drive the result.

4. **Top-k enrichment plot**
   - x-axis: evidence target.
   - y-axis: log2 odds ratio for support in top-k loci versus matched background.
   - Error bars: confidence intervals if available.
   - Diagnostic concern: this checks whether enrichment survives background matching for GC content and regulatory annotation.

5. **Evidence coverage plot**
   - x-axis: TFs.
   - y-axis: number of motif/binding-supported loci available before ranking.
   - Diagnostic concern: this identifies TFs for which validation is impossible or underpowered because there is little external evidence.

### Statistical Tests

Primary ranking test: permutation test using shuffled rankings or shuffled TF labels.

Why this test: the key question is whether supported loci appear higher in HopTF rankings than expected by chance under the same number of loci and TFs. Permuting the ranking or TF labels creates a direct null distribution for recall at k and rank percentiles.

How to interpret it: if observed recall at k is above almost all permuted values, HopTF ranks known evidence higher than expected under the chosen null. If it is not, the motif/binding result should be reported as weak or negative.

Top-k enrichment test: Fisher exact test for supported versus unsupported loci in top-k loci compared with matched background.

Why this test: the data for a fixed TF and top-k are counts in two groups: top-k retrieved loci and matched background loci. Fisher exact is appropriate for count tables and works with small counts.

How to interpret it: an odds ratio above one means support is more common in top-k loci than in matched background. A small p-value means that enrichment is unlikely under the count-table null. This still does not prove binding; it only shows enrichment against the chosen background.

Multiple-testing correction: Benjamini-Hochberg false discovery rate within each evidence target and top-k family.

Why this correction: many TFs and cutoffs are tested. False discovery rate correction limits the expected fraction of significant calls that are false positives.

How to interpret it: report both the number of nominally significant TFs and the number passing FDR. If only nominal tests pass, the result is exploratory.

Correlation diagnostic: Spearman correlation between beta-driven concentration metrics and motif/binding recall.

Why this test: it checks whether sharper retrieval captures more known evidence or simply concentrates on a small unsupported subset.

How to interpret it: positive correlation suggests that retrieval concentration helps biological evidence capture. No correlation means beta concentration should be presented as model behavior, not biological validation.

### Success Criteria

- Recall at k exceeds matched random or shuffled-ranking baselines for same-TF or TF-family motif/binding targets.
- Precision at k is above matched background for at least a meaningful subset of TFs.
- Results are stronger for TF-family and accessibility-supported targets than for brittle exact symbol-only motif matching.
- Evidence coverage is sufficient to interpret failures: TFs with no available motifs or binding data should not count against the model.

### Expected Runtime

Moderate. Most runtime is motif scanning and intersecting loci with external binding/accessibility files. Once support tables are built, recall, precision, and enrichment summaries should be fast.

### Blockers

- Motif-to-TF synonym mapping.
- TF-family mapping.
- ChIP-seq cell-type mismatch.
- Large external peak files and coordinate consistency.
- Defining locus windows consistently with HopTF AlphaGenome-key coordinates.

### Data Download Plan For `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf`

Download only; do not submit jobs yet from this planning step.

- `references/genomes/hg38/`: hg38 FASTA and `.fai`.
- `references/motifs/jaspar/`: JASPAR CORE vertebrates non-redundant motifs in MEME or PWM format plus metadata.
- `references/motifs/hocomoco_or_cisbp/`: optional second motif database for sensitivity.
- `references/annotations/gencode/`: GENCODE hg38 GTF.
- `references/annotations/encode_ccre/`: ENCODE cCRE hg38 BED.
- `references/accessibility/`: H1 hESC ATAC-seq or DNase-seq peaks if available, plus broad ENCODE accessibility tracks if H1-specific data are unavailable.
- `references/chipseq/`: ENCODE or ChIP-Atlas TF ChIP-seq peaks, with metadata recording TF, cell type, genome build, assay, replicate, and file source.
- `references/tf_metadata/`: TF-family mapping and TF synonym table.
- `references/manifests/`: one JSON or TSV manifest with URL, source, version/date, genome build, checksum if available, and local path.

### Do Not Claim Yet

Do not claim that top-ranked loci are validated binding sites. The supported claim is that HopTF rankings recover or enrich for independent motif/binding evidence under specified databases, thresholds, and backgrounds.

## 4. Natural Isoform Diagnostic Or Domain-Aware Redesign

### Question

Do natural TF isoform differences produce retrieval or predicted activity changes that track observed response differences?

The current broad answer is no. In plain language: when all same-gene isoform pairs were pooled together, sequence differences between natural isoforms did not line up with differences in observed response scores or retrieval changes. That means the broad isoform analysis is too noisy or too biologically mixed for the main paper claim.

### Decision

Do not keep the broad natural isoform analysis as a main result. Keep it as a diagnostic unless it can be redesigned around a more specific subset.

### Redesign Criteria

Only rerun as a main analysis if isoform pairs can be annotated for:

- gain or loss of DNA-binding domain.
- gain or loss of activation or repression domain.
- truncation affecting known functional regions.
- sufficient recovered-cell count for both isoforms.
- comparable expression or perturbation quality if that metadata is available.

### Exact Outputs If Redesigned

- `domain_aware_isoform_pairs.csv`: same-gene isoform pairs with functional-domain differences.
- `isoform_retrieval_activity_results.csv`: retrieval and activity changes for each pair.
- `isoform_diagnostic_summary.md`: whether domain-altering isoforms show larger changes than domain-preserving isoforms.

### Plots And Axes

1. **Domain-aware isoform plot**
   - x-axis: isoform-pair class: DNA-binding-domain altered, other domain altered, no annotated domain change.
   - y-axis: retrieval change or absolute predicted activity change.
   - Diagnostic concern: this tests whether the broad negative result was caused by mixing functionally meaningful and uninformative isoform differences.

2. **Isoform response-difference plot**
   - x-axis: absolute observed response-score difference between isoforms.
   - y-axis: attention Jensen-Shannon divergence or predicted activity difference.
   - Diagnostic concern: this checks whether retrieval changes track observed isoform response differences in the curated subset.

### Statistical Tests

Primary test if redesigned: Mann-Whitney test comparing domain-altering isoform pairs with domain-preserving pairs.

Why this test: the groups are independent and likely skewed. The test asks whether one group tends to have larger changes than the other without assuming normality.

How to interpret it: a significant result means domain-altering isoforms tend to produce larger model changes. A nonsignificant result means the isoform analysis remains a negative diagnostic.

Correlation diagnostic: Spearman correlation between retrieval change and observed response-score difference.

Why this test: it asks whether larger retrieval changes correspond to larger observed response differences.

How to interpret it: near-zero correlation means isoform retrieval should not be used as evidence for the main paper claim.

### Expected Runtime

Low if annotations are available. Moderate if domain coordinates need to be mapped onto local isoforms.

### Blockers

- Domain annotation quality for exact isoforms.
- Too few domain-altering same-gene pairs.
- Confounding by cell count or perturbation quality.

### Do Not Claim Yet

Do not claim that natural isoforms validate HopTF unless the domain-aware subset succeeds. The current broad result should be described as a negative diagnostic.

## 5. Sequence Representation Ablation Recheck

### Question

Does ESM-DBP outperform ESM-C for predicting perturbed-cell state after controlling for protein length, ORF length, and recovered-cell count?

### Current Status

The saved result favors ESM-C projected into AlphaGenome key space. Raw ESM-C, raw ESM-DBP, and ESM-DBP projected into AlphaGenome key space underperformed the protein length, ORF length, and recovered-cell-count baseline in the saved summary. The visual labeling needs to be checked before making this a strong claim.

### Outputs

- Re-rendered heatmap with positive values clearly labeled as percent MSE improvement over the protein length, ORF length, and recovered-cell-count baseline.
- Table of exact values from the saved run.
- Short note deciding whether ESM-C remains the default.

### Plot And Axes

- x-axis: held-out setting: all genes, within protein-length groups, within recovered-cell-count groups, within length and cell-count groups.
- y-axis: sequence representation: ESM-C projected, raw ESM-C, ESM-DBP projected, raw ESM-DBP.
- color: percent MSE improvement over the protein length, ORF length, and recovered-cell-count baseline, where positive is better.
- Diagnostic concern: this checks whether the apparent ESM-C advantage is a real result or a plotting/labeling confusion.

### Statistical Test

Use paired differences in held-out-gene MSE between ESM-C projected and ESM-DBP projected, with bootstrap confidence intervals over held-out genes.

Why this test: both encoders are evaluated on the same held-out genes, so paired differences are the relevant comparison. Bootstrapping over held-out genes gives uncertainty without assuming normally distributed errors.

How to interpret it: if the confidence interval for ESM-C minus ESM-DBP improvement is above zero, keep ESM-C as the default. If ESM-DBP clearly improves, consider rerunning variant analyses with ESM-DBP.

### Do Not Claim Yet

Do not claim ESM-DBP is inferior in general. The current result only says it did not improve this HopTF objective in the saved run.

## Data Download Checklist

- ClinVar variant summary with release date and download date.
- hg38 genome FASTA and `.fai`.
- AlphaGenome-key metadata already used by HopTF.
- JASPAR CORE vertebrate motif database and metadata.
- Optional HOCOMOCO or CIS-BP motif database.
- GENCODE hg38 GTF.
- TF synonym mapping.
- TF-family mapping.
- UniProt, Pfam, or InterPro domain annotations.
- ENCODE cCRE hg38 annotations.
- H1 hESC or closest available ATAC-seq/DNase-seq peaks.
- ENCODE or ChIP-Atlas TF ChIP-seq peaks with metadata.
- ESM-C model snapshot or cluster-accessible embedding path.
- Existing HopTF model checkpoints, AlphaGenome keys, and evaluator inputs.

## Script Changes Likely Needed

- Add ClinVar review-status filters and exclusion rules for ambiguous clinical labels.
- Add ESM-C distance calipers and out-of-caliper reporting.
- Add matched benign and matched random-substitution sampling with multiple controls per pathogenic variant.
- Add domain-coordinate mapping from UniProt/Pfam/InterPro to local HopTF isoforms.
- Add gene-clustered bootstrap summaries.
- Add matched retrieval-change computation using the same control sets as the variant activity analysis.
- Add motif and binding support table construction with explicit support columns.
- Add recall at k, precision at k, rank distribution, and matched-background enrichment summaries.
- Add plot captions or sidecar markdown that states the diagnostic concern for each plot.

## QC Checklist For Plots And Tables

- Every plot axis must state the measured quantity and direction of interpretation.
- Every plot must show sample size after filtering.
- Every plot comparing groups must say whether the comparison is matched or unmatched.
- Every variant plot must include or link to an ESM-C distance diagnostic.
- Every motif/binding plot must state the evidence target and denominator.
- Every enrichment table must state the background-matching variables.
- Every p-value table must state the test, alternative hypothesis, and multiple-testing correction.
- Every result table must include the number of TFs, genes, variants, loci, or matched sets used.
- Out-of-caliper variants must be listed separately from the primary test set.
- Diagnostic plots must explicitly name the concern they diagnose in the caption or adjacent text.

## Clear Do-Not-Claim-Yet Caveats

- Do not claim clinical pathogenicity prediction.
- Do not claim retrieved loci are validated binding sites.
- Do not claim motif support proves binding.
- Do not claim ChIP support applies to the perturbation cell state unless the ChIP data are from that cell state or a close context.
- Do not claim retrieval changes are biologically meaningful unless matched controls also control ESM-C distance.
- Do not use the broad natural isoform result as a negative result against HopTF sequence sensitivity; it is a broad diagnostic.
- Do not replace ESM-C with ESM-DBP unless the controlled sequence-representation ablation clearly supports it.
