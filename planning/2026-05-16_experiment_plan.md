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

#### 1B. Protein Encoder Ablation: ESM-C vs ESM-DBP

Intent: test whether a DNA-binding-protein-specialized protein encoder improves HopTF sequence conditioning relative to the current ESM-C 600M embeddings.

Paper role: this is an upstream model-input ablation, not a fifth major experiment. It fills the `TF protein sequence query` implementation placeholder in `paper/HopTF_full_draft.tex` and supports a short ablation statement comparing generic ESM-C with DBP-specialized ESM-DBP under identical endpoint controls.

Core question:
Does ESM-DBP improve controlled endpoint prediction or Hopfield-query performance enough to justify replacing ESM-C as the default TF sequence encoder?

Minimum comparisons:

- `esm_c`
- `esm_dbp`
- `artifacts_plus_esm_c`
- `artifacts_plus_esm_dbp`
- `hopfield_query_from_esm_c`
- `hopfield_query_from_esm_dbp`
- `artifacts_plus_hopfield_query_from_esm_c`
- `artifacts_plus_hopfield_query_from_esm_dbp`
- matching shuffled-feature controls if time allows

Required artifact contract:

- ESM-DBP embedding matrix and vocab must align exactly to the same `isoform_embedding_id` metadata used by ESM-C.
- Record encoder name, model source/version, pooling strategy, embedding dimension, vocab path, embedding path, and any missing isoforms.
- Outputs should be joinable to existing controlled endpoint and Hopfield-query baseline summaries.

Success criterion:

- ESM-DBP must improve controlled endpoint metrics over ESM-C under the same splits and artifact controls before replacing ESM-C as default.
- Stronger evidence if ESM-DBP also improves Hopfield-query performance, mutation/isoform deltas, beta retrieval behavior, or motif enrichment.
- If ESM-DBP improves raw sequence features but not Hopfield-query or artifact-controlled endpoint performance, keep ESM-C as the default and report ESM-DBP as an exploratory ablation.

Draft updates this ablation should enable:

- Add a concise ablation sentence to `Results > Ablation studies`: "We compared generic ESM-C embeddings against DBP-specialized ESM-DBP embeddings under identical artifact-controlled endpoint baselines."
- If ESM-DBP wins, state that DBP-specialized pretraining improves HopTF sequence conditioning.
- If ESM-DBP does not win, state that ESM-C remains the default because DBP specialization did not improve the controlled HopTF objective.

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

Implementation slices:

- **2A. Expanded ClinVar endpoint panel.** Use [planning/2026-05-16_larger_variant_panel_experiment.md](/home/dmeyer/courses/clmm/HopTF/planning/2026-05-16_larger_variant_panel_experiment.md) as the detailed child plan for this slice. It specifies the ClinVar source, inclusion/rejection criteria, nested panels, artifact paths, ESM-C embedding command, frozen endpoint evaluator command, endpoint `percent_delta`, and statistics. This slice asks whether high-confidence deleterious variants reduce predicted HopTF transport more than high-confidence benign variants.
- **2B. Retrieval-delta analysis on the same locked variants.** For every WT/mutant pair from 2A, compute changes after the Hopfield projection and retrieval step: `delta_q`, `delta_attention`, and `delta_mu_p`. This slice is required for the stronger paper claim that sequence variation perturbs the retrieval mechanism, not only the endpoint predictor.
- **2C. Matched random mutation controls.** For each real variant or matched variant group, generate control substitutions matched by TF/gene, residue region or domain status, mutation count, and amino-acid class when feasible. This slice asks whether pathogenic/domain variants are more disruptive than generic sequence perturbations of comparable embedding magnitude.
- **2D. Natural isoform/domain panel.** Use naturally occurring TF isoforms as a separate sequence-variation panel, especially isoforms with altered DNA-binding, activation, repression, or truncation/domain structure. This slice should be reported separately from point mutations because isoforms can differ by many residues and may reflect biology not captured by ClinVar missense pathogenicity.
- **2E. Encoder-choice check for sequence perturbations.** Mutation and isoform deltas depend on the protein encoder. Default to ESM-C unless Experiment 1B shows ESM-DBP improves controlled endpoint or Hopfield-query results. If ESM-DBP clearly wins and embeddings are available, rerun mutation/isoform deltas with ESM-DBP or at least a representative subset.

Panel design across slices:

- Expand beyond the existing 48 ClinVar variants if feasible, using the 2A child plan for concrete curation.
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

Slice-specific outputs:

- **2A ClinVar endpoint panel:** outputs are defined in [planning/2026-05-16_larger_variant_panel_experiment.md](/home/dmeyer/courses/clmm/HopTF/planning/2026-05-16_larger_variant_panel_experiment.md), including locked panel CSVs, mutant sequence FASTA/JSON, ESM-C embeddings, endpoint predictions, annotated validation summary, plots, and `variant_panel_validation_report.md`.
- **2B retrieval-delta analysis:** add one joined table keyed by variant ID with WT/mutant `q_p`, top-k retrieval overlap, attention JSD, attention entropy change, `mu_p` cosine/L2 distance, and endpoint `percent_delta`. Also add plots for retrieval delta by clinical class/domain class and retrieval delta versus endpoint delta.
- **2C random controls:** add a control table linking each real variant or stratum to matched random substitutions, with the same `delta_q`, `delta_attention`, `delta_mu_p`, and `delta_endpoint` metrics. Report whether real pathogenic/domain variants exceed the matched random-control distribution.
- **2D isoform panel:** add isoform-pair metadata, domain-difference annotations, WT/reference-vs-alternative isoform retrieval deltas, and endpoint deltas. Report separately from ClinVar missense statistics.

Success criterion:

- Pathogenic or DNA-binding-domain mutations are more disruptive than benign and matched random controls in retrieval and/or endpoint space.
- If endpoint separation remains weak, use retrieval-space sensitivity as the primary sequence-conditioning result and endpoint prediction as exploratory.

Feasibility gate:

- First expand curation and coordinate validation.
- If embedding generation or coordinate mapping blocks expansion, report the existing 48-variant panel as directional but underpowered, with AUROC/p-value caveats.
- If retrieval-delta computation blocks, keep the 2A endpoint panel as an endpoint-only sequence-sensitivity experiment and explicitly say that retrieval-mechanism validation remains incomplete.
- If matched random controls are too slow, include at least a smaller random-control subset for the top genes or domain-enriched panel.
- If natural isoform annotation is messy, keep natural isoforms as a qualitative guardrail using the existing isoform benchmark rather than merging them into the ClinVar statistical analysis.
- If ESM-DBP embeddings are unavailable or do not clearly beat ESM-C in Experiment 1B, run mutation/isoform panels with ESM-C and treat ESM-DBP mutation deltas as future work.

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

Gap filled:

Without this experiment, `\beta` is only a parameter in the equation `softmax(\beta K q_p)`. The beta sweep turns it into evidence for or against a controllable Hopfield-style retrieval mechanism.

Implementation slices:

- **3A. Retrieval concentration sweep.** Compute how diffuse or concentrated retrieval is as beta changes. This is the required core of the experiment.
- **3B. Retrieval stability sweep.** Compute whether top loci are stable across nearby beta values and relative to a reference beta.
- **3C. Endpoint compatibility sweep.** If the endpoint evaluation path supports it, check whether endpoint quality changes as beta changes. This is useful but should not block the retrieval-only beta result.
- **3D. Biology hook.** If motif enrichment outputs exist, join motif enrichment by beta. If not, record this as a planned bridge to Experiment 4 rather than blocking the beta analysis.

Beta grid:

- Primary grid: `{0.1, 0.3, 1, 3, 10, 30}`.
- Optional collapse check: add `{100}` only if numerical stability and runtime are acceptable.
- Reference beta: use `1` for standardized comparisons, and also record the trained/default checkpoint beta if it differs.

Required concentration metrics for each TF/beta:

- attention entropy.
- normalized entropy.
- primary effective loci: `exp(H)`, matching the definition currently used in `paper/HopTF_full_draft.tex`.
- secondary concentration metric, if useful: inverse Simpson, `1 / sum(a_i^2)`.
- top-10, top-100, and top-1000 attention mass.
- maximum single-locus attention mass.

Required stability metrics:

- top-k overlap between adjacent beta values.
- top-k overlap versus reference beta `1`.
- top-k overlap versus the trained/default checkpoint beta if different.
- optional rank correlation of locus weights across beta values.
- optional stability summary for selected TFs or TF families if global summaries hide heterogeneity.

Endpoint and biology add-ons:

- Endpoint quality versus beta should be included if it can be computed without retraining or changing the core endpoint pipeline. Treat it as a compatibility check, not the primary success condition.
- Motif enrichment versus beta should be included only after motif outputs exist. The beta section should preserve columns/labels needed to join later motif tables by TF, beta, and top-k cutoff.

Required artifacts:

- `beta_sensitivity_metrics.csv`: one row per TF/beta with concentration metrics, top-k mass, optional endpoint metrics, and enough identifiers to join motif outputs later.
- `beta_sensitivity_stability.csv`: adjacent-beta and reference-beta overlap/correlation summaries.
- `beta_sensitivity_summary.json`: beta grid, reference beta, input checkpoint/artifacts, number of TFs, number of loci, and headline regime interpretation.
- `beta_sensitivity_report.md`: paper-ready interpretation and caveats.
- figures for beta vs effective loci, beta vs top-100 mass, and beta stability/top-k overlap.

Success criterion:

- Low beta is diffuse, high beta is concentrated, and intermediate beta gives stable retrieval without destroying endpoint quality.
- If results are mixed, frame beta as revealing a tradeoff between retrieval interpretability and predictive stability.

Failure modes to report explicitly:

- retrieval is already collapsed at low beta.
- retrieval remains diffuse even at high beta.
- top loci are unstable across nearby beta values.
- endpoint quality degrades sharply in the beta range where retrieval becomes interpretable.
- motif enrichment, if available, does not improve with sharper retrieval.

Main figure:

- Panel 1: beta vs effective loci.
- Panel 2: beta vs top-100 mass or entropy.
- Panel 3: beta stability/top-k overlap.
- Panel 4, if available: endpoint quality or motif enrichment versus beta.

Draft updates this experiment should enable:

- Fill `Results > Beta controls retrieval regime` and `fig:beta_sweep`.
- In `Method > Associative retrieval over genomic loci`, keep the current cautious language that beta is an inverse temperature, not a validated TF-concentration proxy.
- In `Retrieval diagnostics`, keep `N_eff = exp(H)` as the primary effective-loci definition for paper consistency. Report inverse Simpson only as a secondary concentration metric if it adds clarity.
- In the abstract/conclusion, this experiment provides the sentence of the form: "Beta sweeps showed [diffuse-to-concentrated retrieval / collapse / instability], supporting [memory-regime / diagnostic-tradeoff] interpretation."

Interpretation rules:

- Strong memory-regime support: beta produces a mostly monotonic diffuse-to-concentrated trend and a recognizable intermediate region with stable top loci.
- Useful mixed result: beta exposes a tradeoff between retrieval interpretability and endpoint stability.
- Weak result: beta changes attention mathematically but does not produce stable or biologically useful retrieval regimes.
- Safe wording: beta is an inverse-temperature parameter controlling retrieval sharpness. Do not describe it as TF concentration or binding affinity without dose-calibrated evidence.

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
