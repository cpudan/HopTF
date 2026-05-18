#!/usr/bin/env python3
"""Parse HMMER hmmscan domtblout output into a compact Pfam domain table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domtblout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-domain-evalue", type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.domtblout.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(maxsplit=22)
            if len(parts) < 22:
                continue
            target_name = parts[0]
            target_acc = parts[1]
            query_name = parts[3]
            full_evalue = float(parts[6])
            domain_ievalue = float(parts[12])
            hmm_from = int(parts[15])
            hmm_to = int(parts[16])
            ali_from = int(parts[17])
            ali_to = int(parts[18])
            env_from = int(parts[19])
            env_to = int(parts[20])
            if domain_ievalue > args.max_domain_evalue:
                continue
            protein_id = query_name.split("|")[-1]
            rows.append(
                {
                    "protein_key": query_name,
                    "protein_id": protein_id,
                    "pfam_name": target_name,
                    "pfam_acc": target_acc.split(".", 1)[0],
                    "full_evalue": full_evalue,
                    "domain_ievalue": domain_ievalue,
                    "hmm_start": hmm_from,
                    "hmm_end": hmm_to,
                    "ali_start": ali_from,
                    "ali_end": ali_to,
                    "env_start": env_from,
                    "env_end": env_to,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["protein_id", "ali_start", "ali_end", "domain_ievalue"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} rows={len(df)}")


if __name__ == "__main__":
    main()
