#!/usr/bin/env python3
"""
Generate K matrix from AlphaGenome CHIP_TF predictions on genomic loci.

Each row of K is the mean CHIP_TF signal vector for one 128kb locus,
producing a matrix of shape (N_loci, N_tracks).

Usage:
    python scripts/generate_k_matrix.py

API key is read from the ALPHAGENOME_API_KEY environment variable,
or passed explicitly via --api-key.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

# --- config ---
CHROMOSOME = "chr22"
FASTA_FILE = f"{CHROMOSOME}.fa"
OUTPUT_PATH = "K_matrix_final.npy"
L = 131072  # 128 kb segments
# --------------


def segment_genome(fasta_file: str, segment_len: int, max_n_fraction: float = 0.1) -> list[str]:
    genome = Fasta(fasta_file)
    loci = []
    for chrom in genome.keys():
        length = len(genome[chrom])
        for start in range(0, length - segment_len + 1, segment_len):
            seq = genome[chrom][start : start + segment_len].seq.upper()
            if seq.count("N") / segment_len < max_n_fraction:
                loci.append(seq)
    return loci


def generate_k_matrix(
    fasta_file: str,
    output_path: str,
    api_key: str,
    checkpoint_interval: int = 10,
) -> np.ndarray:
    from alphagenome.models import dna_client

    sequences = segment_genome(fasta_file, L)
    N = len(sequences)
    print(f"Generated {N} loci of length {L} bp.")

    model = dna_client.create(
        model_version=dna_client.ModelVersion.ALL_FOLDS,
        api_key=api_key,
    )
    requested = [dna_client.OutputType.CHIP_TF]

    checkpoint_path = Path(output_path).with_suffix(".checkpoint.npy")
    K: np.ndarray | None = None
    start_idx = 0

    if checkpoint_path.exists():
        K = np.load(checkpoint_path)
        start_idx = int(np.sum(np.any(K != 0, axis=1)))
        print(f"Resuming from checkpoint — {start_idx}/{N} loci already done.")

    print(f"Encoding {N} loci...")
    for i, seq in enumerate(sequences):
        if i < start_idx:
            continue

        response = model.predict_sequence(
            sequence=seq,
            requested_outputs=requested,
            ontology_terms=[],
        )

        data = response.chip_tf.values

        if data is not None and data.size > 0:
            locus_vector = np.mean(data, axis=0)
        else:
            locus_vector = np.zeros(K.shape[1]) if K is not None else None
            if locus_vector is None:
                print(f"  Warning: locus {i} returned no data and K not yet initialised — skipping.")
                continue

        if K is None:
            K = np.zeros((N, locus_vector.shape[0]))
        K[i] = locus_vector

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{N} loci processed.")

        if (i + 1) % checkpoint_interval == 0:
            np.save(checkpoint_path, K)

    if K is None:
        raise RuntimeError("No data was collected — check your FASTA file and API key.")

    print(f"Final shape: {K.shape}")
    np.save(output_path, K)
    print(f"Matrix saved to '{output_path}'")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return K


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate K matrix of CHIP_TF locus embeddings via AlphaGenome."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ALPHAGENOME_API_KEY"),
        help="AlphaGenome API key (falls back to ALPHAGENOME_API_KEY env var).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N loci (default: 10).",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide an API key via --api-key or ALPHAGENOME_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if not Path(FASTA_FILE).exists():
        print(f"Error: FASTA file '{FASTA_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    generate_k_matrix(FASTA_FILE, OUTPUT_PATH, args.api_key, args.checkpoint_interval)


if __name__ == "__main__":
    main()
