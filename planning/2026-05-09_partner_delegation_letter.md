# HopTF May 18 Sprint Partner Delegation

Subject: Updated HopTF sprint plan for Monday, May 18 report

Hi Peter and Audrey,

I have been reorganizing the HopTF report plan after reviewing the current handoff. The deadline has moved to Monday, May 18, which gives us a little more room to do the science carefully instead of only trying to make the original Wednesday deadline.

The good news is that the basic pipeline now runs end to end: ESM-C isoform embeddings are projected into the AlphaGenome key space, used for Hopfield-style retrieval, and fed into the TorchCFM OT-CFM endpoint model. So I think the report should focus less on proving that the code executes and more on the strongest defensible science we can support.

With the extra time, I think we should aim for a better balanced result: mutant-sequence transport if we can unblock it, beta and motif diagnostics to interpret the retrieval mechanism, and at least one serious check against artifacts or weak endpoint representations. I still want to keep this bounded; the goal is not to turn this into a full method paper in a week.

The core questions I think we should try to answer are:

- Does the sequence-conditioned HopTF transport model respond sensibly to biologically meaningful TF sequence changes, especially DNA-binding-disrupting mutations?
- Does the Hopfield / attention beta parameter control retrieval sharpness in an interpretable way?
- Are high-attention loci enriched for known TF motifs or same-family motifs?
- Do the endpoint and mutation results still look meaningful after checking protein length, recovered cell count, ORF/transduction effects, and natural same-gene isoform behavior?
- If time allows, do stronger endpoint representations such as SCARF or cell-level endpoints change the conclusions relative to PCA centroid endpoints?
- If some of these fail or remain incomplete, what can we honestly say the current HopTF pipeline does and does not show?

Suggested timeline:

- Monday-Tuesday: unblock mutant ESM-C embeddings and strict mutant endpoint evaluation; start beta and motif diagnostics.
- Wednesday: decide which scientific results are actually viable for the report.
- Thursday-Friday: finish validation checks, lock the main figures, and complete a full rough draft.
- Weekend: polish, tighten claims, captions, references, and formatting.
- Monday, May 18: submission mechanics only.

Peter, I would like you to focus first on the mutant-sequence transport evaluation, since that is the most concrete missing piece for answering the sequence-mutation question. The evaluator already exists, but it cannot produce biological mutant predictions until the mutant ESM-C embeddings exist.

Could you generate ESM-C 600M `mean_non_special` embeddings for the six verified mutant sequences in the full-pipeline experiment directory:

```text
experiments/hoptf_full_pipeline_run/mutant_sequences.fasta
```

The mutants are:

- HNF4A: R85W, S87N, R89W, T139I
- GATA2: R396Q, R362Q

Please save:

```text
experiments/hoptf_full_pipeline_run/mutant_esmc_embeddings.npy
experiments/hoptf_full_pipeline_run/mutant_esmc_vocab.json
```

The embeddings should be `float32`, dimension `1152`, and the vocab order should match the embedding matrix rows.

Then please run:

```bash
uv run python scripts/evaluate_mutant_endpoint_predictions.py --strict \
  --outdir experiments/hoptf_full_pipeline_run \
  --mutant-metadata experiments/hoptf_full_pipeline_run/mutant_sequences_metadata.csv \
  --mutant-embedding-matrix experiments/hoptf_full_pipeline_run/mutant_esmc_embeddings.npy \
  --mutant-vocab experiments/hoptf_full_pipeline_run/mutant_esmc_vocab.json \
  --latents experiments/hoptf_full_pipeline_run/perturbation_latents.npz \
  --checkpoint experiments/hoptf_full_pipeline_run/otcfm_real_overfit.pt
```

The immediate goal is for `mutant_endpoint_predictions_report.json` to move from `blocked` to `ok`, and for `mutant_endpoint_predictions.csv` to contain WT-vs-mutant endpoint scores. The interpretation question is whether deleterious DNA-binding mutants reduce predicted response relative to WT and to the benign/weak-effect controls. A negative or ambiguous result is still useful; I mostly want a clean answer with exact output paths.

If you have bandwidth after that, the next most useful thing would be the beta sensitivity / Hopfield attention diagnostic sweep. Please use:

```text
beta = {0.1, 0.3, 1, 3, 10, 30}
```

For each beta, please compute attention entropy, normalized entropy if easy, effective number of loci, top-10/top-100/top-1000 attention mass, adjacent-beta top-k overlap, beta=1 top-k overlap, and rank correlation if feasible. The most useful outputs would be:

- summary CSV;
- stability CSV;
- beta vs effective-loci figure;
- beta vs top-100 mass figure;
- short interpretation of whether beta is making attention more diffuse at low values and more concentrated at high values;
- exact paths to outputs.

You also mentioned that you would be happy to help more with writing, which would be very useful. The pieces where your help would matter most are:

- a concise Methods paragraph explaining the sequence-to-transport pipeline: ESM-C isoform embeddings, projection into AlphaGenome key space, Hopfield-style retrieval, and TorchCFM OT-CFM endpoint transport;
- a Results paragraph for the mutant endpoint evaluation, once we know whether the strict run produces interpretable WT-vs-mutant scores;
- a Results paragraph for the beta sensitivity analysis, if that sweep runs in time;
- a short limitations paragraph on why mutant predictions should be interpreted alongside natural same-gene isoform behavior and artifact controls.

Audrey, I would like you to focus on motif enrichment and main figure design. For motif enrichment, the goal is to test whether high-attention loci for a TF are enriched for that TF's known motif, or at least a same-family motif, relative to matched/background loci.

Please focus first on:

- HNF4A
- TP53
- GATA2
- SOX5
- IKZF3

The most useful motif-enrichment outputs would be:

- inventory of available motif resources/tools;
- top-attention loci at top 100, 500, and 1000;
- matched/background loci from the same eligible key set;
- exact or family-level motif scan results;
- enrichment table with top hit fraction, background hit fraction, odds ratio, p-value, beta/default-beta label, and match type;
- short interpretation of whether motif evidence supports treating attention as biologically meaningful retrieval signal.

I would also like you to own the main figure design. I think the likely figure set is some combination of:

- HopTF pipeline / sequence-conditioned transport schematic;
- WT-vs-mutant predicted endpoint response, if Peter gets the mutant evaluation running;
- beta sensitivity / retrieval sharpness;
- motif enrichment of high-attention loci;
- summary of checks against protein length, recovered cell count, endpoint representation, and natural isoform failure cases.

Please keep the visual claims conservative. Attention should be described as retrieval or diagnostic signal unless motif enrichment supports stronger wording. Mutant transport should be presented carefully unless the checks against protein length, recovered cell count, endpoint representation, and natural isoform failures support the interpretation.

Shared constraints:

- Please do not start jobs expected to run longer than 3 hours without checking first.
- Prefer existing cached embeddings, keys, checkpoints, and outputs wherever possible.
- Please document exact output paths for everything, ideally backed up to HuggingFace if the outputs are needed for the report.
- Negative, ambiguous, or blocked results are useful if they are clear.
- Please do not force TP53 canonical mutations onto the local isoform until the coordinate mismatch is resolved.
- The AlphaGenome genome-build provenance should stay as a caveat unless verified.
- First-pass outputs and short written interpretations by Wednesday would be ideal so we can decide the final scientific scope before locking figures and the full draft.

I will handle the report skeleton, literature table, baseline/artifact integration, and final assembly. I will also keep track of which results support which statements, so that every important sentence in the report is tied to a table, figure, output file, or explicit limitation. My goal is to turn whatever evidence we get into a coherent, honest report rather than trying to make the method look more finished than it is.
