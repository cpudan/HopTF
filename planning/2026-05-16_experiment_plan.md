# HopTF Experiment Plan With Intent

## Summary

Plan the empirical section around one foundation experiment plus three biology/retrieval experiments:

1. **Endpoint prediction + ablation sanity check**: prove HopTF adds predictive signal beyond trivial baselines.
2. **Large mutation/isoform panel**: make sequence conditioning central by testing whether biologically meaningful TF sequence changes perturb retrieval and predicted response more than controls.
3. **Beta sensitivity analysis**: show the Hopfield retrieval layer has a controllable retrieval regime.
4. **Motif enrichment analysis**: test whether high-retrieval loci are biologically plausible TF regulatory contexts.

The central paper claim should be: HopTF predicts TF-conditioned perturbation response while exposing a differentiable genomic retrieval distribution; that distribution is evaluated through sequence perturbation, beta-controlled retrieval behavior, motif enrichment, and ablations against trivial baselines.

## Tie-In To `paper/HopTF_full_draft.tex`

The plan should feed the paper draft directly, not produce disconnected analyses. Each experiment should answer one paper-level claim and fill a specific placeholder in the current draft:

- **Endpoint prediction + ablation sanity check** fills `Results > Endpoint prediction performance`, `Results > Ablation studies`, `Results > Artifact and confound controls`, `Table \ref{tab:endpoint_results}`, `Table \ref{tab:ablation_results}`, `Table \ref{tab:artifact_results}`, and Figure 2 / Figure 5 from the appendix figure list.
- **Large mutation / isoform panel** supports the Introduction claim that TF isoform sequence is a molecular cue, the Method claim that ESM-C sequence embeddings can distinguish isoforms/mutants, and the Results story that sequence perturbations propagate through retrieval and endpoint transport. The draft does not yet have a dedicated mutation subsection, so add one after `Ablation studies` or after `Retrieval distributions identify candidate compatible regulatory loci`.
- **Beta sensitivity analysis** fills `Method > Associative retrieval over genomic loci`, `Method > Retrieval diagnostics`, `Results > Beta controls retrieval regime`, and Figure 3 / `fig:beta_sweep`.
- **Motif enrichment analysis** fills `Claim boundary`, `Method > Retrieval diagnostics`, `Experimental Design > Retrieval and biological validation metrics`, `Results > Retrieval distributions identify candidate compatible regulatory loci`, `Results > Motif enrichment and external binding validation`, `Table \ref{tab:motif_results}`, and Figure 4 / `fig:retrieval_tracks`.
- The final abstract/conclusion placeholders should be written only after these results are known, using the plan's success/downshift rules to decide the claim strength.

## Experiment Plan

### 1. Endpoint Prediction And Ablation Sanity Check

Intent: establish that HopTF is not only an interpretability layer.

Paper role: this is the empirical foundation for the draft's claim that `\mu_p` conditions meaningful perturbation transport. It should be the first Results subsection because the rest of the paper asks readers to care about the retrieval distribution.

Run or report the existing controlled PCA-endpoint benchmark with:

- `artifacts`: protein length, ORF length, recovered cell count.
- `esm_c`: raw TF sequence embedding.
- `artifacts_plus_esm_c`.
- `hopfield_query`.
- `artifacts_plus_hopfield_query`.
- shuffled ESM-C / shuffled endpoint controls.
- control-mean baseline.

Primary metrics:

- grouped endpoint MSE by held-out `gene_symbol`.
- median row MSE.
- mean/median cosine similarity.
- matched-context MSE within length bins, cell-count bins, and length+cell bins.

Success criterion:

- Hopfield-query features improve over artifact-only covariates in overall and matched settings.
- If not, paper frames HopTF as a retrieval-diagnostic prototype, not a validated endpoint predictor.

Report status from current evidence:

- Existing notes already support a modest controlled gain for `artifacts_plus_hopfield_query`.
- Use this as the required foundation while mutation/motif/beta results mature.

Draft updates this experiment should enable:

- Replace the `Endpoint prediction performance` placeholder with a concise table using the implemented metrics rather than every aspirational metric currently listed.
- Replace or narrow `Table \ref{tab:endpoint_results}` to match the actual controlled benchmark: control mean, artifact covariates, ESM-C, artifacts+ESM-C, Hopfield query, artifacts+Hopfield query, shuffled controls.
- Use `Artifact and confound controls` to report the length/cell-count/ORF baseline explicitly, not as a side note.
- In the abstract, this experiment provides the sentence of the form: "In an artifact-controlled PCA-endpoint benchmark, HopTF-query features [improved/did not improve] over [baseline] by [metric]."

### 2. Large Mutation / Isoform Panel

Intent: make sequence conditioning central, not decorative.

Paper role: this is the most direct experiment for the paper's sequence-conditioned claim. It tests whether the TF sequence query is more than a label and whether variants/isoforms alter `q_p`, retrieval weights, `\mu_p`, and predicted transport.

Core question:
Do disease-relevant or domain-relevant TF sequence changes produce structured changes in retrieval weights, retrieved regulatory representation, and predicted endpoint response?

Panel design:

- Expand beyond the existing 48 ClinVar variants if feasible.
- Include benign/likely benign vs pathogenic/likely pathogenic TF missense variants.
- Prioritize variants coordinate-checked against local responder isoforms.
- Add domain labels where possible: DNA-binding domain, activation/repression domain, outside-domain.
- Add matched random substitutions as negative controls, matched by TF, region length, amino-acid class if feasible, and mutation count.
- Include natural isoforms with altered domains as a secondary panel, reported separately from point mutations.

Primary outputs:

- `delta_q`: distance between WT and mutant TF embedding.
- `delta_attention`: JSD or cosine distance between WT and mutant retrieval weights.
- `delta_mu_p`: cosine/L2 distance between WT and mutant retrieved representation.
- `delta_endpoint`: mutant-minus-WT predicted response L2 and endpoint MSE.
- stratified plots by benign/pathogenic, domain/outside-domain, and real/random mutation class.
- AUROC and one-sided tests for pathogenic/domain variants being more disruptive than benign/matched controls.

Success criterion:

- Pathogenic or DNA-binding-domain mutations are more disruptive than benign and matched random controls in retrieval and/or endpoint space.
- If endpoint separation remains weak, use retrieval-space sensitivity as the primary sequence-conditioning result and endpoint prediction as exploratory.

Feasibility gate:

- First expand curation and coordinate validation.
- If embedding generation or coordinate mapping blocks expansion, report the existing 48-variant panel as directional but underpowered, with AUROC/p-value caveats.

Draft updates this experiment should enable:

- Add a dedicated Results subsection, suggested title: `Sequence variants perturb retrieval and endpoint predictions`.
- Add a compact table or figure for mutation classes: benign/pathogenic, DNA-binding-domain/outside-domain, real/matched-random, and natural isoforms if available.
- In the Method section's `TF protein sequence query`, add one paragraph explaining how mutant and isoform sequences are embedded and compared to WT.
- In Discussion/Limitations, connect weak or nonsignificant endpoint separation to the existing draft caveat that current supervision is aggregate and endpoint representation may be too coarse.
- In the abstract, this experiment provides the sentence of the form: "Variant-panel analysis showed [strong/weak/no] sequence sensitivity, with [retrieval-space/endpoint-space] effects strongest for [class]."

### 3. Beta Sensitivity Analysis

Intent: validate the associative-memory framing.

Paper role: this is the core experiment for the memory-model claim. It makes `\beta` an analyzed retrieval parameter rather than a decorative equation in the Method section.

Core question:
Does beta control retrieval sharpness and expose a stable, interpretable retrieval regime?

Run beta values:

- `{0.1, 0.3, 1, 3, 10, 30}` unless compute suggests adding `{100}` for collapse behavior.

For each beta, compute:

- attention entropy.
- normalized entropy.
- effective number of loci, `1 / sum(a_i^2)`.
- top-10, top-100, top-1000 attention mass.
- top-k overlap between adjacent beta values.
- top-k overlap versus beta `1`.
- optional rank correlation of locus weights.
- endpoint quality versus beta if the model/evaluation path supports it.
- motif enrichment versus beta if motif outputs are available.

Success criterion:

- Low beta is diffuse, high beta is concentrated, and intermediate beta gives stable retrieval without destroying endpoint quality.
- If results are mixed, frame beta as revealing a tradeoff between retrieval interpretability and predictive stability.

Main figure:

- Panel 1: beta vs effective loci.
- Panel 2: beta vs top-k mass or entropy.
- Panel 3: beta vs endpoint quality or motif enrichment.

Draft updates this experiment should enable:

- Fill `Results > Beta controls retrieval regime` and `fig:beta_sweep`.
- In `Method > Associative retrieval over genomic loci`, keep the current cautious language that beta is an inverse temperature, not a validated TF-concentration proxy.
- In `Retrieval diagnostics`, align the effective-loci definition used in code and text. The draft currently defines `N_eff = exp(H)`, while the plan also proposes inverse Simpson `1 / sum(a_i^2)`. Report both only if useful; otherwise choose one and make the paper/code consistent.
- In the abstract/conclusion, this experiment provides the sentence of the form: "Beta sweeps showed [diffuse-to-concentrated retrieval / collapse / instability], supporting [memory-regime / diagnostic-tradeoff] interpretation."

### 4. Motif Enrichment Analysis

Intent: test whether retrieved loci are biologically plausible, without claiming attention equals binding.

Paper role: this is the main experiment for the draft's claim boundary. It determines whether high-retrieval loci can be discussed as biologically plausible regulatory-context hypotheses or must remain purely model-internal diagnostics.

Core question:
Are high-retrieval loci enriched for cognate TF motifs or same-family motifs compared with matched genomic backgrounds?

Design:

- For each TF, take top-k retrieved loci at default beta and optionally across beta values.
- Use top-k cutoffs such as `100`, `500`, and `1000`, or a fixed retrieval-weight quantile if locus count varies.
- Compare against matched background loci.
- Match backgrounds for GC content and broad locus class at minimum; add accessibility/promoter/enhancer/chromosome matching if available.
- Test exact TF motif and family-level motif matches.
- Include random retrieval or shuffled TF-query controls.

Primary outputs:

- per-TF motif enrichment table with beta, top-k cutoff, motif match type, top-hit fraction, background-hit fraction, odds ratio, p-value/FDR.
- summary fraction of TFs with enriched cognate/family motifs.
- optional browser-style examples for top retrieved loci.
- motif enrichment versus beta.

Success criterion:

- A meaningful fraction of TFs show cognate or family motif enrichment over matched background.
- If weak or absent, paper says retrieval weights are diagnostic hypotheses requiring further biological validation.

Draft updates this experiment should enable:

- Fill `Results > Motif enrichment and external binding validation`, `Table \ref{tab:motif_results}`, and part of `Results > Retrieval distributions identify candidate compatible regulatory loci`.
- Replace the generic motif placeholder in `Experimental Design > Retrieval and biological validation metrics` with the exact motif source, background construction, top-k thresholds, and multiple-testing correction.
- Use motif results to decide wording in the Introduction `Claim boundary`: enriched motifs allow "model-prioritized regulatory-context hypotheses"; weak motifs require "retrieval diagnostics whose biological interpretation remains unresolved."
- In the abstract/conclusion, this experiment provides the sentence of the form: "Top-retrieved loci were [enriched/not enriched] for cognate or family motifs under [background], supporting [claim strength]."

## Priority And Downshift Rules

Full-paper priority:

1. Large mutation/isoform panel.
2. Endpoint/ablation foundation.
3. Motif enrichment.
4. Beta sensitivity.

Deadline/report-ready downshift:

- If mutation expansion blocks, use the existing 48-variant ClinVar panel as exploratory and put effort into motif + beta.
- If motif resources/background matching block, report beta + endpoint + mutation and explicitly list motif enrichment as required future validation.
- If beta sweep blocks, report endpoint + mutation + motif, and describe beta analysis as missing memory-regime validation.
- Do not let any interpretability result replace endpoint/ablation evidence.

## Acceptance Criteria

The empirical section is complete when every major claim maps to a table, figure, output path, or limitation:

- Predictive claim: controlled endpoint baseline table.
- Sequence-conditioning claim: mutation/isoform panel with real-vs-control stratification.
- Memory-mechanism claim: beta concentration/stability plots.
- Biological-plausibility claim: motif enrichment table with matched background.
- Guardrail claim: explicit limitations on PCA-centroid endpoints, length/cell-count artifacts, natural isoform failures, and attention-not-binding wording.
- Paper assembly claim: every filled Results subsection in `paper/HopTF_full_draft.tex` has a corresponding artifact path, and every abstract/conclusion sentence is backed by one of those artifacts or by an explicit limitation.

## Assumptions

- Use existing PCA-centroid endpoint artifacts unless a stronger endpoint representation becomes available.
- Treat current 48-variant ClinVar result as valid exploratory evidence, not statistical validation.
- Keep mutation expansion central, but use feasibility gates to avoid sacrificing all reportable evidence.
- Do not claim retrieved loci are binding sites unless motif enrichment and background controls support that language.
- Keep the LaTeX draft's narrative hierarchy: endpoint prediction establishes utility; mutation tests sequence conditioning; beta tests associative retrieval behavior; motif enrichment tests biological plausibility.
