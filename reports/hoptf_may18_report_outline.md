# HopTF May 18 Report Outline

## Operating Frame

Working thesis: HopTF is a sequence-conditioned retrieval and diagnostic feasibility study, not yet a fully validated predictive biology system.

Done-enough version for this draft: every section exists, every strong statement has a figure/table/output-path placeholder or explicit limitation, and Peter/Audrey placeholders are visible enough for them to drop in their work.

## Claim And Artifact Map

| Report claim | Evidence artifact | Owner | Status |
| --- | --- | --- | --- |
| Sequence-conditioned retrieval can connect TF protein sequence representations to regulatory response context. | Pipeline schematic; method description; key/protein embedding setup | Dan + Peter | Draft placeholder |
| Hopfield beta controls retrieval sharpness and stability enough to support diagnostic interpretation. | Beta sweep CSV/plots; entropy/top-k mass/stability summaries | Peter | Awaiting drop-in |
| High-attention loci should be checked against motif or motif-family enrichment before making biological claims. | Motif enrichment table and matched/background definition | Audrey | Awaiting drop-in |
| Mutant sequence transport shows directional exploratory endpoint sensitivity but is not decisive validation yet. | Six-mutant and 48-variant ClinVar summaries; p-value/AUROC; limitation language | Dan + Peter | Usable exploratory summary |
| The May 18 report should emphasize feasibility, guardrails, and next evidence needed. | Main figure set; limitations section; future-work paragraph | Dan + Audrey | Draft placeholder |

## Abstract

- Problem: Need a way to connect TF protein sequence variation to regulatory response hypotheses.
- Approach: Use ESM-C protein representations, AlphaGenome/regulatory keys, Hopfield-style retrieval, and endpoint transport diagnostics.
- Main result: [PLACEHOLDER: strongest supported result after Peter/Audrey updates]
- Conservative interpretation: Feasibility and diagnostic evidence, with mutation effects treated as exploratory unless stronger evidence lands.
- Next step: Stronger endpoint validation and better biological controls.

## Introduction

### Motivation

- TF sequence changes can alter regulatory programs, but mapping protein sequence variation to cell-state response is difficult.
- Existing work often separates protein representation, regulatory sequence modeling, and perturbation response prediction.
- HopTF asks whether a sequence-conditioned retrieval layer can expose useful regulatory response structure.

### Background And Related Work

- Protein language models:
  - ESM-C/ESM-style embeddings encode protein sequence features useful for downstream biological prediction.
  - Link to HopTF: represent TF isoforms and mutants as sequence-derived vectors.
- Regulatory sequence and chromatin models:
  - AlphaGenome-style representations provide regulatory context keys or features.
  - Link to HopTF: retrieve regulatory loci/context conditioned on TF sequence.
- Perturbation response prediction:
  - Response models provide a target space for measuring whether retrieved regulatory context predicts meaningful cellular shifts.
  - Link to HopTF: endpoint transport is a feasibility target, not yet the final validation claim.
- Attention/retrieval mechanisms:
  - Hopfield-style retrieval gives an interpretable diagnostic handle through attention concentration, top-k mass, and stability.
  - Link to HopTF: beta sensitivity can test whether retrieval behavior is controllable and interpretable.
- Motif enrichment and biological guardrails:
  - Motif checks are a sanity check for whether high-attention loci relate to known TF binding preferences.
  - Link to HopTF: motif evidence supports cautious interpretation; absence or weak enrichment limits biological claims.

### Contribution Framing

- Present HopTF as a feasibility prototype with explicit guardrails.
- Avoid claiming validated causal mutation prediction unless supported by stronger endpoint evidence.
- Make every result traceable to an artifact, output path, or limitation.

## Methods

### Data And Representations

- [PLACEHOLDER: dataset inventory and response target]
- TF protein sequences and isoforms:
  - [PETER PLACEHOLDER: ESM-C embedding details, dimensions, pooling, mutant/WT handling]
- Regulatory context keys:
  - [PLACEHOLDER: AlphaGenome/key construction and preprocessing]

### Hopfield Retrieval Layer

- [PETER PLACEHOLDER: projection into key space and Hopfield-style retrieval formulation]
- Diagnostics to report:
  - attention entropy
  - normalized entropy if available
  - effective number of loci
  - top-10/top-100/top-1000 mass
  - top-k overlap and beta stability

### Endpoint Transport / Response Scoring

- [PETER PLACEHOLDER: TorchCFM / OT-CFM endpoint transport setup]
- [PLACEHOLDER: response representation and known caveats]

### Motif Enrichment

- [AUDREY PLACEHOLDER: selected TF panel]
- [AUDREY PLACEHOLDER: top-loci cutoff, matched/background definition, motif source]
- Report table columns:
  - TF
  - beta or default setting
  - top-k cutoff
  - motif match type
  - top hit fraction
  - background hit fraction
  - odds ratio
  - p-value
  - output path

### Mutant Evaluation

- Existing evidence:
  - Six hand-picked HNF4A/GATA2 mutants.
  - 48-variant ClinVar first-pass panel.
  - Current conclusion: directional but not statistically significant; deleterious vs benign one-sided p about 0.10, AUROC about 0.578.
- [PETER PLACEHOLDER: any updated mutant endpoint results or blocker statement]

## Results

### Result 1: Sequence-Conditioned Retrieval Pipeline

- Figure placeholder: pipeline schematic.
- Claim: HopTF connects TF protein sequence representation to regulatory context retrieval and response diagnostics.
- Evidence needed: schematic, method paragraph, input/output examples.

### Result 2: Beta Sensitivity And Retrieval Diagnostics

- [PETER PLACEHOLDER: beta sweep summary]
- Figure/table placeholder:
  - beta vs effective loci
  - beta vs top-100 mass
  - beta stability/top-k overlap table
- Claim wording should stay diagnostic unless outputs are strong.

### Result 3: Motif Enrichment Sanity Check

- [AUDREY PLACEHOLDER: motif enrichment table]
- Figure/table placeholder:
  - motif enrichment summary table
  - optional compact plot if Audrey has one
- Interpretation:
  - motif-supported retrieval if enrichment is clear
  - retrieval-only diagnostic signal if enrichment is weak
  - unresolved if background/control definition is incomplete

### Result 4: Mutant Sequence Transport Is Exploratory

- Current evidence:
  - Six hand-picked mutants and 48 ClinVar variants show weak directional support.
  - Not decisive enough for central validation.
- Figure/table placeholder:
  - compact guardrail panel or table with p-value/AUROC and caveat.
- Safe wording:
  - "directionally consistent exploratory signal"
  - not "validated mutation effect prediction"

### Result 5: Guardrails And Failure Modes

- Protein length and representation artifacts remain important.
- Natural isoform behavior should contextualize mutant effects.
- Endpoint representation may change conclusions; SCARF/cell-level response remains future validation.

## Main Figure Plan

- [AUDREY PLACEHOLDER] Panel A: pipeline schematic.
- [PETER PLACEHOLDER] Panel B: beta/retrieval diagnostic behavior.
- [AUDREY PLACEHOLDER] Panel C: motif enrichment sanity check.
- [DAN/PETER PLACEHOLDER] Panel D: exploratory mutant transport or guardrail summary.

## Discussion

- HopTF is promising as a diagnostic framework for connecting protein sequence representations to regulatory response hypotheses.
- Current evidence supports feasibility and bounded diagnostic interpretation.
- The strongest honest contribution is an evidence-linked story with visible limitations.

## Limitations

- Mutation effects are exploratory and underpowered.
- Endpoint response representation may be too coarse.
- Motif enrichment depends on background definition and motif resource quality.
- Retrieval attention should not be treated as binding without supporting evidence.
- Stronger validation needs larger mutation panels, natural isoform controls, and cell-level endpoint checks.

## Immediate Owner Placeholders

- Peter:
  - Methods paragraph for ESM-C, projection, Hopfield retrieval, and endpoint transport.
  - Beta diagnostics results paragraph with output paths.
  - Mutant endpoint paragraph or blocker statement.
  - Limitation language around mutant interpretation.
- Audrey:
  - Motif enrichment table and interpretation.
  - Main figure layout.
  - Figure-caption claim map for each panel.
  - Any plot files and output paths.

## Final Assembly Checklist

- Add exact output paths for every table/figure.
- Replace unsupported claims with limitations.
- Keep abstract aligned with actual evidence.
- Verify figures and captions do not imply validated binding or causal mutation prediction.
- Reserve Monday May 18 for compile/export/submission mechanics only.
