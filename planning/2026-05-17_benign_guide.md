# Nominating and Using Benign Variants in Human Transcription Factors

## A practical guide for experimental design and evaluation datasets

## 1. Scope and framing

The goal is not to exhaustively identify every benign mutation in a human transcription factor. The goal is to construct a **high-confidence control set**, likely on the order of ~200 variants, that can be used for evaluating a model or designing a key experiment.

That distinction matters. “Benign” can mean several different things:

| Label                                 | Meaning                                                                                                                  | Appropriate use                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **Clinically benign / likely benign** | Variant is interpreted as not causative for a disease context, usually under ACMG/AMP-style clinical evidence frameworks | Clinical negative controls; strongest label when well curated |
| **Population-tolerated**              | Variant is observed at meaningful frequency in human population data without obvious disease signal                      | Putative benign controls; useful at scale                     |
| **Functionally neutral**              | Variant has WT-like behavior in a relevant assay                                                                         | Best for assay-specific negative controls                     |
| **Molecularly neutral**               | Variant does not affect the molecular phenotype under study                                                              | Hardest to prove; requires endpoint-matched assays            |
| **Computationally predicted benign**  | A model such as AlphaMissense, ESM, REVEL, etc. predicts low deleteriousness                                             | Filter or sanity check, not ground-truth label                |

For this project, the safest language is **“putatively benign controls”** or **“high-confidence benign/control variants”**, with each variant assigned an **evidence tier**. Avoid treating all controls as equivalent.

The central recommendation is:

> Build a tiered benign/control set from ClinVar, gnomAD, and functional/MAVE evidence; annotate domain, conservation, structure, and ESM behavior; then evaluate model performance separately on low-, moderate-, and high-ESM-delta benign subsets.

Do not use ESM, AlphaMissense, or related predictors as the primary source of labels if your method uses protein language model geometry, conservation, structural features, or related information.

---

## 2. Key takeaways

1. **Do not try to mechanistically prove benignity for every variant.** That is too slow and will still be incomplete. Instead, build a **high-precision, evidence-tiered control set**.

2. **Use natural human variants first.** The strongest practical benign/control evidence comes from curated ClinVar benign/likely benign variants and common or moderately common gnomAD variants.

3. **Separate clinical benignity from molecular neutrality.** A common human missense variant can be clinically tolerated but still alter TF binding strength, motif specificity, localization, degradation, or cofactor interaction.

4. **For transcription factors, “outside the DNA-binding domain” is not enough.** TFs often have activation/repression domains, dimerization motifs, NLS/NES regions, degrons, PTM clusters, short linear motifs, and cofactor interaction regions. A compendium of human TF effector domains cataloged 924 effector domains across 594 human TFs, emphasizing that non-DBD regions can be central to TF function. ([PMC][1])

5. **Large ESM embedding deltas do not necessarily imply hidden structural damage.** They usually indicate that a mutation moves the sequence away from the model’s learned natural/evolutionary sequence manifold. That manifold encodes conservation, residue grammar, structural constraints, family constraints, and functional constraints.

6. **High-ESM-delta benign variants should not be automatically discarded.** They should be handled as a **stress-test subset**, with stronger independent evidence required for inclusion.

7. **Avoid making the negative set too easy.** If pathogenic variants are mostly in structured DNA-binding domains and benign variants are mostly in disordered regions, the model may simply learn “DBD vs IDR.” This exact issue has been shown for PAX6: apparent VEP performance dropped when disordered-region benign variants were excluded and evaluation focused on the DNA-binding domains. ([Nature][2])

---

## 3. What people usually do in practice

Most variant-effect-prediction benchmarks do something simpler than a mechanistic checklist. A common pattern is:

```text
positives: ClinVar pathogenic / likely pathogenic
negatives: ClinVar benign / likely benign and/or gnomAD population variants
optional: functional assay neutral variants
filters: remove conflicts, bad mappings, low-quality variants, obvious circularity
```

A 2024 Scientific Reports study, for example, benchmarked variant-effect predictors by taking pathogenic/likely pathogenic missense variants from ClinVar and putatively benign missense variants from gnomAD, excluding variants present in the pathogenic set. The authors also highlighted biases in this design, especially when benign variants are enriched in intrinsically disordered or weakly conserved regions. ([Nature][2])

This is broadly the right template for your use case, but with stricter filtering and explicit stratification.

---

## 4. Evidence hierarchy for benign/control labels

### Tier 1: Curated clinical benign variants

These are the cleanest labels when available.

**Inclusion criteria:**

```text
ClinVar germline clinical significance:
  Benign
  Likely benign
  Benign/Likely benign

Preferred review status:
  criteria provided, multiple submitters, no conflicts
  reviewed by expert panel
  practice guideline

Additional requirements:
  exact match to intended transcript/protein isoform
  missense only
  no conflicting pathogenic/likely pathogenic assertion
  no splice-risk flag
  not somatic-only
  not a known disease/cancer hotspot
```

ClinVar review status is useful because NCBI reports the level of review behind a classification. Two stars correspond to “criteria provided, multiple submitters, no conflicts,” three stars to “reviewed by expert panel,” and four stars to “practice guideline.” ([NCBI][3])

**Use:** gold benign controls.
**Weakness:** sparse coverage, especially for TFs with limited clinical annotation.

---

### Tier 2: Population-tolerated variants

These are the practical workhorse for reaching ~200 controls.

**Recommended inclusion criteria:**

```text
gnomAD PASS missense variant
correct genome build and transcript/protein mapping
max/group allele frequency ≥ 0.1%, preferably ≥ 1%
allele count high enough to avoid single-sample artifacts
no ClinVar pathogenic/likely pathogenic/VUS/conflicting interpretation
no splice-risk prediction
no known disease/cancer hotspot annotation
not clearly in a critical TF feature
```

For current gnomAD usage, use the latest release available in your pipeline. gnomAD v4.1.1 was released in March 2026 with updated gene constraint metrics, LOFTEE flags, and related annotations. ([gnomAD][4]) For formal clinical-style allele-frequency reasoning, ClinGen guidance for gnomAD v4 recommends using the exome/genome-combined **Grpmax Filtering AF** when applying BA1/BS1, noting that it is the lower-bound allele frequency from the genetic ancestry group with the highest filtering AF and that bottlenecked populations are excluded from that statistic. ([ClinGen][5])

For an experimental negative set, I would use:

```text
High-confidence population control:
  max/group AF ≥ 1%
  AC ≥ 100 when possible
  preferably observed in multiple ancestry groups or with many carriers
  no clinical or domain red flags

Moderate-confidence population control:
  max/group AF ≥ 0.1%
  AC ≥ 20–50
  no red flags
```

Do not treat frequency thresholds as universal clinical criteria. A variant can be too common for one severe early-onset dominant disease but still compatible with late-onset, low-penetrance, recessive, or modifier phenotypes. For your purpose, the threshold is not proving clinical benignity; it is enriching for variants that are unlikely to cause major deleterious effects under ordinary human population sampling.

---

### Tier 3: Functional-neutral variants

Use these when assay-relevant data exist.

Sources include:

```text
MaveDB
published deep mutational scanning datasets
protein-binding microarray datasets
SELEX-like TF binding assays
reporter assays
stability/localization assays
```

MaveDB is now large enough to check systematically. Its 2024 update reported over 7 million variant-effect measurements across 1,884 datasets as of November 2024, and MaveDB describes itself as an open-access repository for multiplexed assays of variant effect, including DMS and MPRA datasets. ([Springer][6])

For a TF, assay relevance matters. A variant that is neutral in a protein abundance assay may still alter DNA-binding specificity. A variant neutral in a DNA-binding assay may still affect transcriptional activation, repression, localization, or cofactor recruitment.

**Best use:** endpoint-matched negative controls.

For example:

| Your endpoint               | Best neutral evidence                                     |
| --------------------------- | --------------------------------------------------------- |
| TF-DNA binding affinity     | PBM, EMSA, SELEX, DMS binding assay                       |
| DNA motif specificity       | PBM/SELEX-style assay with specificity readout            |
| Transcriptional activation  | reporter assay or endogenous perturbation                 |
| Protein abundance/stability | abundance/stability DMS                                   |
| Localization                | imaging or NLS/NES-specific assay                         |
| Cofactor interaction        | pulldown, two-hybrid, proximity labeling, interaction DMS |

ClinGen’s functional evidence framework treats well-established functional assays as usable evidence for abnormal or normal function, but stresses disease mechanism, assay class, assay validity, and variant-level application; it also notes that a minimum set of pathogenic and benign controls is needed to reach moderate evidence when rigorous statistical calibration is absent. ([ClinGen][7])

---

### Tier 4: Ortholog-supported or family-supported synthetic controls

Use only when natural human variants are insufficient.

A synthetic substitution is more defensible if:

```text
human WT residue -> amino acid observed at same aligned position in orthologs
orthologs have the same domain architecture
position is not in a known specificity/contact/interface residue
position has moderate or high MSA entropy
substitution is chemically plausible
no structural, motif, or splice red flags
```

These are not “clinically benign variants.” They are **evolutionarily tolerated substitutions**. Keep them in a separate category.

### Tier 5: Legacy conservative synthetic fallback controls

Use only to preserve a fixed-size design when Tiers 1--4 do not provide enough rows for an isoform. These rows are deterministic conservative amino-acid substitutions generated from the local isoform sequence. They avoid sequence edges, Cys/His/Gly/Pro/Trp positions when possible, C2H2-like zinc-finger motifs, basic patches, leucine-zipper-like heptads, and low-complexity windows when possible.

Tier 5 rows do **not** have independent clinical, population, functional, or ortholog evidence. They are not clinical benign variants and should not be described as evolutionarily tolerated substitutions. Their only role is to keep an exactly balanced 10-variants-per-isoform file available for software and embedding workflows that require a fixed number of controls per isoform. Analyses should report Tier 5 separately and should treat any result that depends on Tier 5 rows as lower confidence.

---

## 5. Why TF benign controls are unusually tricky

Human transcription factors are not simple globular enzymes where active-site/core logic dominates. A TF mutation can affect:

```text
DNA-binding affinity
DNA-binding specificity
cofactor binding
dimerization
chromatin recruitment
activation/repression domains
intrinsic disorder-mediated interactions
nuclear localization/export
degradation
post-translational modification
protein abundance
cell-type-specific regulatory behavior
```

This matters because a variant can look structurally harmless and still be functionally important.

Homeodomain data provide a useful caution. A 2024 Nature Communications study found that single missense substitutions can alter DNA-binding affinity, specificity, or both, including variants at positions not known to contact DNA directly. The authors also noted that sequence-similarity-based motif prediction approaches are not designed to predict single-missense effects on TF binding specificity. ([Nature][8])

So for TFs, do not use this logic:

```text
outside DBD = benign
no AlphaFold fold change = benign
low structural RMSD = benign
not conserved across all species = benign
```

Use this logic instead:

```text
strong independent benign evidence
AND no TF-specific functional red flags
AND no evolutionary/structural red flags severe enough to undermine the label
```

---

## 6. Recommended dataset design for ~200 benign controls

A reasonable target composition is:

```text
Tier 1: ClinVar B/LB, high review status
  20–60 variants if available

Tier 2: gnomAD population-tolerated
  100–160 variants

Tier 3: functional-neutral MAVE/DMS/PBM/reporter variants
  20–80 variants if available

Tier 4: ortholog-supported synthetic variants
  only as needed, and reported separately

Tier 5: legacy conservative synthetic fallback
  only when a fixed-size file is required and Tiers 1--4 do not fill all slots
```

If the 200 controls must come from **one TF**, truly high-confidence benign missense controls may be sparse. In that case, do not force all 200 into one label class. Use separate labels:

```text
clinical_benign
population_tolerated
functional_neutral
ortholog_supported_synthetic
legacy_conservative_fallback
computationally_low_risk
```

If the 200 controls can come from a **panel of TFs**, high-confidence natural controls become much more feasible.

---

## 7. Inclusion and exclusion criteria

### 7.1 Hard exclusions

Exclude a variant from the high-confidence benign set if any of the following hold:

```text
ClinVar pathogenic / likely pathogenic / conflicting
ClinVar VUS unless there is much stronger independent benign evidence
known disease-associated or cancer-driver hotspot
somatic-only interpretation when germline benignity is needed
poor genome/protein coordinate mapping
multiple isoforms with ambiguous residue mapping
not actually missense in the intended transcript
near splice junction or predicted splice-altering
low-quality population call
only observed in a tiny number of samples
```

Also exclude obvious TF-specific risk features:

```text
canonical DNA-contact residue
known specificity-determining residue
zinc-finger Cys/His ligand
basic DNA-contact region of bZIP/bHLH domain
leucine zipper or obligate dimerization interface
homeodomain recognition helix or flanking specificity region
ETS/forkhead/nuclear receptor core DBD contact region
known cofactor-binding interface
NLS/NES
degron
PTM cluster
short linear motif
activation/repression effector domain when endpoint-relevant
buried structured core residue
metal/cofactor-binding residue
```

The TF-specific exclusions should be interpreted relative to your experimental endpoint. If your assay only measures DNA-binding specificity, an activation-domain variant may be acceptable as a negative control for that endpoint but not as a general TF-function benign control.

---

### 7.2 Soft red flags

These do not automatically exclude a variant, but they should lower confidence or trigger manual review:

```text
large ESM embedding delta
large ESM log-likelihood penalty
high conservation
mutant residue never observed in orthologs/paralogs
high HMC or regional constraint signal
buried structured position
charge reversal
Pro/Gly insertion into helix or beta strand
Cys gain/loss
near domain boundary
in disordered region with motif-like composition
in low-complexity activation domain
rare despite high gnomAD coverage
```

High ESM delta belongs here: **review, stratify, and require stronger independent evidence**, but do not automatically discard.

---

## 8. Domain, family, and conservation annotation

### 8.1 Domain annotation

Use InterPro and Pfam for domain mapping. InterPro integrates signatures from multiple member databases to classify proteins into families, domains, and significant sites; Pfam families are built from seed alignments and profile HMMs. 

Do not ask only:

```text
Is this Pfam family tolerant?
```

Ask:

```text
Is this specific residue/column tolerant?
Is this aligned column constrained across the family?
Are population variants observed at homologous positions?
Are ClinVar pathogenic variants enriched at homologous positions?
Is this column part of a DNA-contact/interface/metal-binding substructure?
```

Family-level tolerance is too coarse. A domain can have permissive loops and intolerant core/contact residues.

---

### 8.2 Constraint metrics

Use domain- and region-level constraint as filters.

Useful annotations include:

```text
gnomAD gene-level constraint
regional missense constraint / MTR
HMC or homologous-domain constraint
MetaDome-like homologous-domain tolerance
```

MTR is a regional measure of missense intolerance based on observed vs expected standing variation, intended to identify regions under purifying selection.  HMC is an amino-acid-level measure of missense intolerance within Pfam domains, using observed vs expected missense variation across homologous domain positions; the UCSC track documentation notes that HMC covers Pfam domains and that lower HMC scores indicate stronger constraint. ([UCSC Genome Browser][9])

Important interpretation:

```text
constrained position = red flag
unconstrained position = not proof of benignity
```

Absence of constraint is weak evidence. Presence of constraint is more informative.

---

### 8.3 MSA/conservation features

For each candidate mutation, compute:

```text
site_entropy
WT amino acid frequency at the aligned column
mutant amino acid frequency at the aligned column
profile-HMM log odds for WT
profile-HMM log odds for mutant
Δ profile score = logP(mutant) - logP(WT)
mutant observed in orthologs? yes/no
mutant observed in paralogs? yes/no
nearest ortholog/paralog carrying mutant residue
```

Use at least two alignments if possible:

```text
ortholog-only MSA
family/paralog MSA
```

The distinction matters. Ortholog variation is often better evidence of tolerability. Paralog variation can be misleading because paralogs may encode different binding specificity or regulatory functions.

For TFs, paralog evidence is especially risky in DNA-binding domains. A residue that varies across paralogs may be a specificity determinant, not a benign-tolerance signal.

---

## 9. Structural annotation

Use structure to detect obvious structural-risk variants, not to prove neutrality.

### 9.1 Structural features to compute

For each variant:

```text
local pLDDT or experimental structure availability
secondary structure
relative solvent accessibility
buried vs exposed
side-chain contacts
hydrogen bonds / salt bridges
metal coordination
disulfide involvement
distance to DNA/protein/ligand interface
distance to known PTM or motif
ΔΔG estimate if structured
inverse-folding compatibility score
WT vs mutant local contact changes
```

AlphaFold confidence should gate structural interpretation. EMBL-EBI describes pLDDT >90 as very high confidence, 70–90 as confident, 50–70 as low, and <50 as very low. ([EMBL-EBI][10]) For pLDDT <50 regions, treat apparent structure cautiously; these regions may be intrinsically disordered or flexible.

### 9.2 Interpreting structural signals

Use the following triage:

| Structural result                                                   | Interpretation                                           |
| ------------------------------------------------------------------- | -------------------------------------------------------- |
| Buried, high-confidence core residue with predicted destabilization | Exclude or downgrade                                     |
| Exposed non-interface residue with no motif/domain signal           | More compatible with benignity                           |
| Disordered region with no known motif                               | Potentially acceptable, but check SLiMs/PTMs/degrons     |
| Disordered activation/repression domain                             | Endpoint-dependent; not automatically benign             |
| DNA/protein interface residue                                       | Exclude unless strong functional-neutral evidence exists |
| No ESMFold/AlphaFold structural change                              | Weak evidence only                                       |

For TFs, predicted fold preservation is not enough. DNA-binding specificity, cofactor binding, localization, and regulatory activity can change without a large backbone perturbation.

---

## 10. Understanding large ESM embedding deltas

### 10.1 What a large ESM delta means

A large ESM embedding change usually means:

```text
the mutant sequence is far from the local learned sequence manifold
```

That manifold is not purely structural. It includes:

```text
evolutionary conservation
amino-acid grammar
local biochemical compatibility
domain-family constraints
secondary-structure preferences
protein interaction signatures
motifs and low-complexity patterns
functional-site regularities
```

ESM-1v showed that protein language models can make zero-shot predictions of mutational effects using learned evolutionary sequence information, without needing a newly trained model for each protein family. ([NeurIPS Proceedings][11]) This is exactly why ESM features are useful, but also why they can confound labels: ESM may penalize mutations that are evolutionarily unusual even when they are clinically tolerated.

A large ESM delta can reflect:

| Cause                        | Possible biology                     | Structural consequence required? |
| ---------------------------- | ------------------------------------ | -------------------------------- |
| Conserved site mutation      | Loss of function or altered function | Sometimes                        |
| Rare substitution class      | Local sequence grammar violation     | Sometimes                        |
| Buried core mutation         | Destabilization                      | Often                            |
| DNA specificity residue      | Altered binding specificity          | No                               |
| NLS/degron/PTM/SLiM mutation | Regulatory change                    | No                               |
| IDR composition disruption   | Cofactor or phase/interaction effect | Usually no                       |
| Model geometry artifact      | Representation effect                | No                               |

### 10.2 Why large ESM deltas may drive large deltas in your model

If your model consumes ESM embeddings or representations trained near ESM geometry, there are three plausible explanations:

```text
1. Real biology:
   the mutation truly changes TF function.

2. Out-of-distribution sensitivity:
   the input moves far from the training manifold and the model extrapolates.

3. Label mismatch:
   the variant is population-tolerated or clinically benign but not molecularly neutral.
```

The third explanation is important. A variant can be common and clinically tolerated while still causing a measurable molecular perturbation. That is particularly plausible for TFs, where modest changes in binding affinity or specificity may not yield a recognizable monogenic disease phenotype.

Use this distinction:

```text
population-benign ≠ molecularly neutral
clinically benign ≠ no TF-binding effect
no fold change ≠ no functional change
```

---

## 11. How to analyze ESM deltas rigorously

Do not use a single raw embedding-distance value. Decompose it.

### 11.1 Recommended ESM-derived quantities

For each candidate mutation:

```text
ESM_LLR_i =
  log P(mutant amino acid at position i | masked context)
  - log P(WT amino acid at position i | masked context)

ΔE_site =
  || h_mut[i] - h_wt[i] ||

ΔE_window(k) =
  mean_j∈[i-k, i+k] || h_mut[j] - h_wt[j] ||

ΔE_global =
  mean_j over full protein || h_mut[j] - h_wt[j] ||

ΔE_CLS / sequence-level embedding delta =
  || pooled_embedding_mut - pooled_embedding_wt ||
```

Compute these for multiple layers. Early, middle, and final ESM layers can emphasize different types of information.

### 11.2 Site-normalized ESM delta

This is one of the most important diagnostics.

For each site, compute the ESM delta for all 19 possible substitutions:

```text
ΔE_site_z =
  (ΔE_variant - mean(ΔE_all_19_substitutions_at_site))
  / sd(ΔE_all_19_substitutions_at_site)
```

Then compute the same for your model:

```text
model_delta_z =
  (model_delta_variant - mean(model_delta_all_19_substitutions_at_site))
  / sd(model_delta_all_19_substitutions_at_site)
```

This separates two phenomena:

```text
site-level sensitivity:
  this position is generally important

substitution-level sensitivity:
  this particular amino-acid change is unusual at this site
```

Then compare:

```text
corr(raw ΔE, raw model delta)
corr(site-normalized ΔE, site-normalized model delta)
```

Interpretation:

| Result                                                           | Meaning                                                         |
| ---------------------------------------------------------------- | --------------------------------------------------------------- |
| Raw correlation high, normalized correlation low                 | Model mostly tracks site-level conservation/sensitivity         |
| Raw and normalized correlations high                             | Model also tracks substitution-specific perturbation            |
| Correlation mostly with ESM_LLR                                  | Model tracks sequence plausibility/naturalness                  |
| Correlation remains after controlling for conservation/structure | ESM embedding contains additional signal relevant to your model |

### 11.3 Regression diagnostic

Run something like:

```text
model_delta ~
  ESM_LLR
  + ΔE_site
  + ΔE_window
  + MSA_entropy
  + mutant_aa_frequency_in_MSA
  + domain_class
  + disorder
  + relative_solvent_accessibility
  + pLDDT
  + buried_core_flag
  + interface_flag
  + substitution_class
  + gene/protein fixed effect
```

Then inspect:

```text
coefficient for ΔE_site
partial R² for ΔE_site
residual model_delta after removing ESM_LLR/conservation/structure
performance by ESM-delta bin
```

If ESM delta remains highly predictive after these controls, it may be carrying additional functional signal. If it disappears after controlling for ESM_LLR and conservation, your model is mostly sensitive to learned naturalness/conservation.

---

## 12. Testing whether high-ESM-delta benign variants have structural consequences

Do not use WT-vs-mutant ESMFold RMSD as the main structural test. ESMFold is useful, but it is not a calibrated point-mutation stability model, and it is not independent of ESM representations.

A better structural screen is:

```text
1. Use experimental structure when available.
2. Otherwise use AlphaFold/ESMFold only where pLDDT is high enough.
3. Compute SASA / buriedness.
4. Compute local contacts.
5. Check DNA/protein/ligand interface distance.
6. Compute ΔΔG with FoldX, Rosetta, MutateX, DDGun, etc.
7. Compute inverse-folding compatibility with ESM-IF1 or ProteinMPNN.
8. Compare WT and mutant structural predictions only as weak supporting evidence.
```

ESM-IF1 is closer to the structural-compatibility question because it scores or designs sequences conditioned on backbone coordinates; the ESM repository describes ESM-IF1 as an inverse-folding model that predicts protein sequences from backbone atom coordinates and can score sequences for a given structure. ([GitHub][12])

Use this interpretation table:

| Pattern                                                           | Interpretation                                                              |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| High ΔESM + buried + high pLDDT + poor inverse-folding + high ΔΔG | Likely structural/stability risk                                            |
| High ΔESM + exposed + no interface/motif + high AF/homozygotes    | Likely sequence-naturalness signal or clinically tolerated molecular effect |
| High ΔESM + DNA-contact/specificity region                        | Possible TF-binding or specificity effect even without fold change          |
| High ΔESM + IDR motif/PTM/degron/NLS                              | Possible regulatory effect, not structural fold effect                      |
| High ΔESM + strong MAVE-neutral evidence                          | Valuable high-ESM-delta benign stress-test control                          |

---

## 13. How to use high-ESM-delta benign variants

Do not throw them all away. Instead, split the benign/control set.

Recommended structure for 200 controls:

```text
Core benign set:
  ~120–160 variants
  strong ClinVar/gnomAD/MAVE evidence
  low to moderate ESM delta
  no major red flags

Matched benign set:
  ~40–80 variants
  matched to positives by domain, disorder, exposure, conservation, or protein region

High-ESM-delta stress-test set:
  ~20–40 variants
  high ESM delta
  strong independent benign evidence
  no structural/domain/motif red flags
```

For the **high-ESM-delta stress-test set**, require one of:

```text
ClinVar B/LB with strong review status
gnomAD AF ≥ 1% with many carriers
observed homozygotes when biologically relevant
WT-like functional assay result
ortholog-supported substitution plus no other red flags
```

Report model performance separately on:

```text
all benign controls
core benign controls
domain-matched benign controls
high-ESM-delta benign controls
structured-domain benign controls
IDR/non-domain benign controls
DBD-adjacent benign controls
```

The stress-test subset directly answers your concern: whether your model systematically calls high-ESM-delta but independently benign variants disruptive.

---

## 14. Practical implementation pipeline

### Step 1: Define scope

Specify:

```text
genes / TFs included
canonical protein isoform
genome build
transcript IDs
experimental endpoint
what “benign” means for this experiment
whether variants must be natural human variants
whether synthetic substitutions are allowed
```

Use stable identifiers:

```text
HGNC gene symbol
Ensembl gene ID
Ensembl transcript ID
UniProt accession
protein isoform
genome build: GRCh38 preferred
HGVS genomic/coding/protein notation
```

### Step 2: Pull candidate variants

Data sources:

```text
ClinVar
gnomAD latest available release
MaveDB
published DMS/MAVE/PBM/reporter studies
UniProt feature annotations
InterPro/Pfam
AlphaFold DB / PDB
JASPAR / CIS-BP / TFClass / HumanTFs
cancer hotspot databases if relevant
splice predictors
```

JASPAR is useful for TF binding-profile context; the current JASPAR site describes CORE as a curated, non-redundant set of experimentally derived TF binding profiles and provides downloadable PFMs/TFFMs and APIs. ([JASPAR][13]) CIS-BP is also directly relevant, describing itself as an online library of TFs and DNA-binding motifs, with a 2024 database build reporting motif and TF coverage across many species. ([cisbp.ccbr.utoronto.ca][14])

### Step 3: Normalize mappings

For each candidate:

```text
genomic coordinate -> transcript consequence -> protein coordinate
check reference allele
check protein residue matches expected UniProt/Ensembl sequence
resolve isoform differences
drop ambiguous mappings
drop variants affecting multiple protein products inconsistently unless intentional
```

This step is mundane but critical. Many errors in variant-control sets come from coordinate or isoform mismatch.

### Step 4: Apply hard filters

Example:

```text
remove if:
  not missense in chosen transcript
  low-quality call
  ClinVar P/LP/conflict/VUS
  splice-risk
  known disease/cancer hotspot
  ambiguous protein mapping
  in explicit excluded residue class
```

### Step 5: Annotate biological features

For each remaining variant, annotate:

```text
protein domain
TF family
DBD status
effector domain status
NLS/NES
dimerization motif
coiled coil / leucine zipper
PTM sites
degron/SLiM
disorder
conservation
MSA entropy
ortholog/paralog support
HMC/MTR/domain constraint
structure confidence
SASA / buriedness
interface distance
ΔΔG
inverse-folding score
ESM_LLR
ESM embedding deltas
your model delta
```

### Step 6: Assign evidence tier

Use a deterministic rule.

Example:

```text
Tier A:
  ClinVar B/LB
  review status ≥ two-star equivalent
  no conflicts
  no red flags

Tier B:
  gnomAD AF ≥ 1%
  AC ≥ 100
  no ClinVar/VUS/pathogenic/conflict
  no red flags

Tier C:
  gnomAD AF ≥ 0.1%
  AC ≥ 20–50
  no red flags
  extra support from conservation/structure/orthologs

Tier D:
  WT-like functional assay
  endpoint relevant
  no contradictory clinical/population evidence

Tier E:
  ortholog-supported synthetic
  no red flags
  kept separate from natural benign controls

Tier F:
  legacy conservative synthetic fallback
  no independent benign/tolerated evidence
  used only for fixed-size coverage when Tiers A--E are insufficient
  always reported separately
```

### Step 7: Balance and stratify

Balance by:

```text
gene
TF family
domain class
DBD vs non-DBD
structured vs disordered
solvent exposure
conservation bin
ESM-delta bin
allele-frequency bin
```

Avoid a control set where all benign variants are common non-domain IDR substitutions and all positives are rare DBD substitutions.

### Step 8: Freeze a provenance table

For each selected variant, record:

```text
variant_id
gene
transcript
protein isoform
protein_change
genomic_coordinate
source_label
evidence_tier
ClinVar significance
ClinVar review status
gnomAD version
gnomAD AF / AC / AN / groupmax
homozygote count if relevant
MAVE/DMS source and score
domain annotation
TF feature annotation
MSA entropy
mutant residue frequency in MSA
ortholog support
HMC/MTR/constraint
pLDDT
SASA
ΔΔG
ESM_LLR
ΔE_site
ΔE_window
ΔE_global
high_ESM_delta_bin
manual_review_notes
final_include/exclude
```

---

## 15. Recommended scoring rubric

Use labels and flags rather than a single scalar score. But if ranking is useful, a simple rubric works.

### Positive evidence for benign/control inclusion

| Evidence                                         | Strength           |
| ------------------------------------------------ | ------------------ |
| ClinVar B/LB, ≥2-star/no conflict                | Very strong        |
| ClinVar B/LB expert panel/practice guideline     | Very strong        |
| gnomAD AF ≥1%, many carriers                     | Strong             |
| gnomAD AF ≥0.1%, sufficient AC                   | Moderate           |
| observed homozygotes, when biologically relevant | Moderate to strong |
| WT-like endpoint-relevant MAVE/DMS/PBM score     | Strong             |
| mutant residue observed in orthologs             | Moderate           |
| low conservation / high entropy                  | Moderate           |
| exposed non-interface residue                    | Weak to moderate   |
| neutral ΔΔG / inverse-folding compatibility      | Weak to moderate   |

### Negative evidence / downgrade flags

| Evidence                            | Action                                                    |
| ----------------------------------- | --------------------------------------------------------- |
| ClinVar P/LP/conflict               | Exclude                                                   |
| splice-risk                         | Exclude or manual review                                  |
| DBD contact/specificity residue     | Exclude unless endpoint-irrelevant and strongly justified |
| zinc ligand / metal ligand          | Exclude                                                   |
| dimerization/core interface         | Exclude or downgrade                                      |
| buried high-confidence core residue | Downgrade/exclude                                         |
| high conservation                   | Downgrade                                                 |
| high HMC/MTR constraint             | Downgrade                                                 |
| high ESM_LLR penalty                | Downgrade or stress-test                                  |
| large ESM embedding delta           | Stratify; require stronger evidence                       |
| known PTM/degron/NLS/SLiM           | Endpoint-dependent downgrade                              |

---

## 16. Model-evaluation design

For evaluating your model, use multiple negative sets rather than one.

### 16.1 Core evaluation sets

```text
Set 1: Gold benign
  ClinVar B/LB, high review status

Set 2: Population benign
  common gnomAD variants, no red flags

Set 3: Functional neutral
  WT-like assay variants

Set 4: Matched benign
  matched to positive variants by domain/region/exposure/conservation

Set 5: High-ESM-delta benign stress test
  independent benign evidence but large ESM shift
```

### 16.2 Report metrics separately

Do not report only aggregate AUROC/AUPRC.

Report:

```text
performance on ClinVar benign
performance on gnomAD benign
performance on MAVE-neutral
performance on matched benign
performance on high-ESM-delta benign
false-positive rate by ESM-delta quantile
false-positive rate by domain
false-positive rate by disorder/structure
false-positive rate by conservation bin
```

This will tell you whether model deltas are being driven by true functional perturbation, ESM geometry, or dataset composition.

### 16.3 Avoid circularity

Do not label variants benign because:

```text
ESM says benign
AlphaMissense says benign
REVEL says benign
conservation says low risk
your own model says low risk
```

Those features can be used as filters or annotations, but they should not define ground truth.

AlphaMissense is especially worth treating carefully. Its database explicitly states that “likely benign” and “likely pathogenic” are AlphaMissense computational prediction categories, not ACMG/AMP clinical classifications. ([AlphaMissense][15]) DeepMind also describes AlphaMissense as being fine-tuned using labels based on variants observed in human/primate populations versus absent variants, and explicitly notes that AlphaMissense does not predict structural change or protein stability effects directly. ([Google DeepMind][16])

---

## 17. Specific advice for the ESM-delta confounding concern

Your observed phenomenon:

```text
mutations with large ESM embedding shifts
→ large deltas in your model
```

is plausible and should be expected if your model is sensitive to pLM sequence geometry.

The key question is not:

```text
Are these high-ESM-delta variants actually structurally damaging?
```

The better questions are:

```text
Are high-ESM-delta variants enriched for real molecular effects?
Are they enriched for structural instability?
Are they enriched for conservation/domain/interface features?
Does your model still respond strongly after controlling for those features?
Are population-benign high-ESM-delta variants false positives or molecularly non-neutral?
```

Recommended analysis:

```text
1. Bin candidate benign variants by ESM delta quantile.
2. For each bin, compute:
   - allele frequency
   - ClinVar evidence
   - domain/DBD status
   - MSA entropy
   - HMC/MTR
   - pLDDT
   - buriedness
   - interface proximity
   - ΔΔG
   - inverse-folding score
   - your model delta

3. Compare high-ESM-delta vs low-ESM-delta benign candidates.
4. Manually inspect the top 20–50 high-ESM-delta benign candidates.
5. Keep only those with strong independent evidence and no obvious biology red flags.
6. Use them as a stress-test subset, not as ordinary negatives.
```

A useful three-way classification:

```text
Keep as high-confidence benign:
  strong independent evidence
  no domain/motif/structure/conservation red flags

Keep as population-tolerated but molecularly ambiguous:
  strong population evidence
  high ESM delta or some biological concern
  useful for stress testing

Exclude:
  weak population evidence
  high ESM delta
  high conservation or functional-region/structure red flags
```

---

## 18. Suggested final dataset format

Use a table with these columns:

```text
variant_uid
gene
hgnc_id
ensembl_gene
ensembl_transcript
uniprot_accession
isoform
genome_build
chrom
pos
ref
alt
hgvs_c
hgvs_p
aa_position
aa_ref
aa_alt

label
label_tier
label_source
clinical_benign_flag
population_tolerated_flag
functional_neutral_flag
synthetic_ortholog_flag
synthetic_conservative_flag

clinvar_significance
clinvar_review_status
clinvar_conflict_flag
clinvar_vcv
clinvar_conditions

gnomad_version
gnomad_AC
gnomad_AN
gnomad_AF
gnomad_grpmax_AF
gnomad_popmax_group
gnomad_hom_count
gnomad_quality_flags

mavedb_accession
functional_assay_type
functional_score
functional_score_normalized
functional_neutral_cutoff
functional_assay_endpoint

tf_family
domain
domain_start
domain_end
in_DBD
in_effector_domain
in_dimerization_region
in_NLS_NES
in_PTM_site
in_SLIM_or_degron
in_known_interface

msa_entropy
wt_freq_in_msa
mut_freq_in_msa
mut_seen_in_orthologs
mut_seen_in_paralogs
profile_delta_logodds

hmc_score
mtr_score
regional_constraint_bin

structure_source
pLDDT_local
relative_SASA
buried_flag
secondary_structure
interface_distance
delta_delta_G
inverse_folding_delta

ESM_LLR
ESM_delta_site
ESM_delta_window_5
ESM_delta_window_15
ESM_delta_global
ESM_delta_site_z
ESM_delta_quantile

model_delta
model_delta_site_z
include_final
exclusion_reason
manual_review_notes
```

This provenance table will be as important as the final list of 200 variants.

---

## 19. Minimal implementation recipe

A direct recipe:

```text
1. Define TF panel and isoforms.

2. Pull ClinVar missense variants.
   Keep B/LB, review status ≥ 2 stars, no conflicts.

3. Pull gnomAD missense variants.
   Keep PASS variants with AF ≥ 0.1%; mark AF ≥ 1% as stronger.
   Use current gnomAD release and record version.

4. Pull MaveDB/published functional variants.
   Keep WT-like variants only if assay endpoint is relevant.

5. Map all variants to the same protein-coordinate system.

6. Remove:
   - splice-risk variants
   - ClinVar P/LP/VUS/conflicts
   - ambiguous mappings
   - obvious TF critical residues
   - structural core/interface red flags

7. Annotate:
   - TF domains/features
   - InterPro/Pfam
   - MSA conservation
   - HMC/MTR
   - AlphaFold/PDB structure
   - ESM_LLR and ESM embedding deltas
   - model delta

8. Rank:
   ClinVar B/LB > common gnomAD > functional neutral > moderate-frequency gnomAD > ortholog-supported synthetic > legacy conservative fallback.

9. Select 200 with balance:
   - avoid overrepresentation of IDRs
   - avoid overrepresentation of one TF family
   - include a high-ESM-delta stress subset
   - keep evidence tiers explicit

10. Freeze final table before evaluating model performance.
```

---

## 20. Practical recommendations for the current experiment

For your specific concern, I would structure the benign set as follows:

```text
Total: 200 benign/control variants

Core high-confidence set: 120
  ClinVar B/LB or gnomAD AF ≥ 1%
  no major red flags
  low/moderate ESM delta

Matched set: 50
  selected to match positives by:
    TF family
    domain/non-domain
    DBD/non-DBD
    disorder/structure
    conservation bin
    solvent exposure

High-ESM-delta stress set: 30
  high ESM delta
  but strong independent benign evidence:
    ClinVar B/LB high review
    or gnomAD AF ≥ 1%
    or WT-like MAVE/PBM/reporter evidence
  no obvious structure/domain/motif red flags
```

Then report evaluation as:

```text
FPR on core benign
FPR on matched benign
FPR on high-ESM-delta benign
FPR by ESM-delta decile
FPR by domain class
FPR by conservation bin
```

If the model fails mostly on the high-ESM-delta stress set, that is not necessarily a bug. It means the model is sensitive to mutations that protein language models view as non-natural. The next question is whether those variants are **truly molecularly neutral** for your endpoint. Some may be clinically/population benign but still biologically meaningful.

---

## 21. Final checklist

### Variant can enter the high-confidence benign/control pool if:

```text
[ ] correct transcript/protein mapping
[ ] missense in intended isoform
[ ] strong ClinVar B/LB, common gnomAD, or functional-neutral evidence
[ ] no ClinVar P/LP/conflict/VUS issue
[ ] no splice-risk issue
[ ] no disease/cancer hotspot issue
[ ] no known DBD/contact/specificity residue issue
[ ] no dimerization/cofactor/interface issue
[ ] no NLS/NES/degron/PTM/SLiM issue, unless endpoint-irrelevant
[ ] no buried-core or obvious structural-risk issue
[ ] conservation not severely contradictory, or strong independent evidence overrides it
[ ] ESM delta recorded and binned
[ ] evidence tier recorded
[ ] manual review note added
```

### Variant should be stress-test only if:

```text
[ ] strong independent benign/population/functional evidence
[ ] large ESM embedding delta or ESM_LLR penalty
[ ] no obvious biological red flag after manual review
[ ] useful for probing model sensitivity
```

### Variant should be excluded if:

```text
[ ] weak evidence
[ ] high ESM delta
[ ] high conservation
[ ] functional-domain or interface concern
[ ] structural-risk concern
[ ] splice-risk concern
[ ] clinical conflict
```

---

## 22. Bottom-line recommendation

For a 200-variant benign/control set, the best design is not a single homogeneous “benign” label. It is a **tiered, provenance-rich, stratified control panel**.

Use:

```text
ClinVar B/LB and common gnomAD variants as primary labels
MAVE/DMS/PBM/reporter neutral variants as endpoint-specific controls
domain/conservation/structure/ESM as filters and stratification variables
high-ESM-delta variants as a dedicated stress-test subset
```

Treat large ESM embedding shifts as a signal of **sequence-manifold departure**, not automatically as hidden structural damage. Some high-ESM-delta variants will be structurally or functionally suspicious and should be excluded. Others will be strong population- or ClinVar-benign variants and should be retained as an explicit challenge set.

The most important experimental-design principle is:

> Do not let the benign set be defined by the same features your model uses. Label from independent evidence; annotate ESM and structure afterward; then stratify evaluation by those annotations.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8818021/?utm_source=chatgpt.com "Compendium of human transcription factor effector domains"
[2]: https://www.nature.com/articles/s41598-024-76202-6 "Understanding the heterogeneous performance of variant effect predictors across human protein-coding genes | Scientific Reports"
[3]: https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/ "Review status in ClinVar"
[4]: https://gnomad.broadinstitute.org/news/2026-03-gnomad-v4-1-1/ "gnomAD v4.1.1 | gnomAD browser"
[5]: https://clinicalgenome.org/site/assets/files/9445/clingen_guidance_to_vceps_regarding_the_use_of_gnomad_v4_march_2024.pdf "March 2024 Communication to ClinGen VCEPs from ClinGen SVI VCEP Review Committee"
[6]: https://link.springer.com/article/10.1186/s13059-025-03476-y "MaveDB 2024: a curated community database with over seven million variant effects from multiplexed functional assays | Genome Biology | Springer Nature Link"
[7]: https://clinicalgenome.org/docs/recommendations-for-application-of-the-functional-evidence-ps3-bs3-criterion-using-the-acmg-amp-sequence-variant-interpretation/ "Recommendations for application of the functional evidence PS3/BS3 criterion using the ACMG/AMP sequence variant interpretation framework - ClinGen | Clinical Genome Resource"
[8]: https://www.nature.com/articles/s41467-024-47396-0 "DNA binding analysis of rare variants in homeodomains reveals homeodomain specificity-determining residues | Nature Communications"
[9]: https://genome.ucsc.edu/cgi-bin/hgTables?db=hg38&hgta_doSchema=describe+table+schema&hgta_group=phenDis&hgta_table=hmc&hgta_track=hmc "Schema for HMC - HMC - Homologous Missense Constraint Score on PFAM domains	"
[10]: https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/ "pLDDT: Understanding local confidence | AlphaFold"
[11]: https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html "Language models enable zero-shot prediction of the effects of mutations on protein function"
[12]: https://github.com/facebookresearch/esm "GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins · GitHub"
[13]: https://jaspar.elixir.no/ "JASPAR - A database of transcription factor binding profiles"
[14]: https://cisbp.ccbr.utoronto.ca/ "CIS-BP Database: Catalog of Inferred Sequence Binding Preferences"
[15]: https://alphamissense.hegelab.org/ "AlphaMissense"
[16]: https://deepmind.google/blog/a-catalogue-of-genetic-mutations-to-help-pinpoint-the-cause-of-diseases/ "A catalogue of genetic mutations to help pinpoint the cause of diseases — Google DeepMind"
