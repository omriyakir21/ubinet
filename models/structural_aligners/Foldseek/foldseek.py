#!/usr/bin/env python3
import os
import re
import sys
import glob
import argparse
import subprocess
import shutil
import pandas as pd

# ---------- helpers ----------
def which(cmd):
    return shutil.which(cmd)

def assert_db_exists(prefix: str, label: str):
    matches = glob.glob(prefix + "*")
    if not matches:
        raise FileNotFoundError(
            f"{label} DB prefix not found or empty: {prefix}\n"
            f"Expected files like: {prefix}, {prefix}_ss, {prefix}_h, {prefix}_ca, ..."
        )

def base_id(s: str) -> str:
    b = os.path.basename(str(s))
    return re.sub(r'\.(pdb|cif)(\.gz)?$', '', b, flags=re.I)

def read_lookup_ids(db_prefix: str) -> list:
    """Read IDs from Foldseek's .lookup (second column is the fasta/structure id)."""
    path = db_prefix + ".lookup"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lookup file not found: {path}")
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()  # fallback: whitespace
            if len(parts) >= 2:
                ids.append(parts[1])
            else:
                ids.append(parts[0])
    return sorted(set(ids))

# def run_foldseek_easy_search(query_db: str,
#                              target_db: str,
#                              out_tsv: str,
#                              tmp_dir: str,
#                              cov: float,
#                              max_seqs: int,
#                              threads: int,
#                              foldseek_bin: str = "foldseek"):
#     if not which(foldseek_bin):
#         raise RuntimeError(f"'{foldseek_bin}' not found on PATH")
#     os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
#     os.makedirs(tmp_dir, exist_ok=True)

#     cmd = [
#         foldseek_bin, "easy-search",
#         query_db, target_db,
#         out_tsv, tmp_dir,
#         "--alignment-type", "1",
#         "--cov-mode", "1", "-c", str(cov),
#         "--max-seqs", str(max_seqs),
#         "--format-output",
#         "query,target,alntmscore,qtmscore,ttmscore,lddt,alnlen,"
#         "qstart,qend,qlen,tstart,tend,tlen,evalue,bits",
#         "--threads", str(threads),
#     ]
#     print("\nRunning:", " ".join(cmd))
#     res = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     if res.stdout.strip():
#         print("Foldseek STDOUT:\n", res.stdout)
#     if res.stderr.strip():
#         print("Foldseek STDERR:\n", res.stderr)

def run_foldseek_easy_search(query_db: str,
                             target_db: str,
                             out_tsv: str,
                             tmp_dir: str,
                             cov: float,
                             max_seqs: int,
                             threads: int,
                             foldseek_bin: str = "foldseek"):
    if not which(foldseek_bin):
        raise RuntimeError(f"'{foldseek_bin}' not found on PATH")

    # Validate DB prefixes exist (catch common path mistakes early)
    assert_db_exists(query_db, "Query")
    assert_db_exists(target_db, "Target")

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    base_cmd = [
        foldseek_bin, "easy-search",
        query_db, target_db,
        out_tsv, tmp_dir,
        "--alignment-type", "1",
        "--cov-mode", "1", "-c", str(cov),
        "--max-seqs", str(max_seqs),
        "--threads", str(threads),
    ]

    # Some Foldseek builds do not support 'bits' in --format-output.
    # Try preferred (with bits), then fall back (without bits).
    format_variants = [
        "query,target,alntmscore,qtmscore,ttmscore,lddt,alnlen,"
        "qstart,qend,qlen,tstart,tend,tlen,evalue,bits",
        "query,target,alntmscore,qtmscore,ttmscore,lddt,alnlen,"
        "qstart,qend,qlen,tstart,tend,tlen,evalue",
    ]

    last_code = None
    last_stdout = ""
    last_stderr = ""

    for fmt in format_variants:
        cmd = base_cmd + ["--format-output", fmt]
        print("\nRunning:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        last_code = res.returncode
        last_stdout = res.stdout
        last_stderr = res.stderr

        if res.stdout.strip():
            print("Foldseek STDOUT:\n", res.stdout)
        if res.stderr.strip():
            print("Foldseek STDERR:\n", res.stderr)

        if res.returncode == 0:
            return  # success

        print(f"foldseek easy-search exited with code {res.returncode}. Trying a fallback format if available...")

    # If we get here, all attempts failed
    raise RuntimeError(
        "foldseek easy-search failed.\n"
        f"Exit code: {last_code}\n"
        f"Command: {' '.join(cmd)}\n"
        f"STDERR (tail):\n{last_stderr[-2000:]}\n"
        "Hints: check that DB prefixes exist and are created with 'foldseek createdb', "
        "and that your Foldseek version supports the requested --format-output fields. "
        "You may need to update Foldseek."
    )


def parse_hits(tsv_path: str) -> pd.DataFrame:
    cols = ["query","target","alntmscore","qtmscore","ttmscore","lddt",
            "alnlen","qstart","qend","qlen","tstart","tend","tlen","evalue","bits"]
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=cols)
    if df.empty:
        return df
    df["qcov"] = (df["qend"] - df["qstart"] + 1) / df["qlen"]
    df["tcov"] = (df["tend"] - df["tstart"] + 1) / df["tlen"]
    df["uniprot"] = df["query"].apply(base_id)
    df["most_similar"] = df["target"].apply(base_id)
    return df

# def write_best_csv(all_query_ids, df_merged: pd.DataFrame, out_csv: str):
#     # Rank within each query, preferring single-domain coverage
#     if not df_merged.empty:
#         df_sorted = df_merged.sort_values(
#             ["uniprot","qtmscore","alntmscore","lddt"],
#             ascending=[True, False, False, False]
#         )
#         best = df_sorted.groupby("uniprot", as_index=False).first()
#     else:
#         best = pd.DataFrame(columns=["uniprot","most_similar","qtmscore","alntmscore","ttmscore",
#                                      "lddt","qcov","tcov","alnlen","evalue","bits"])
#     # Ensure every query appears in the final CSV
#     allq = pd.DataFrame({"uniprot": sorted(all_query_ids)})
#     best = allq.merge(best, on="uniprot", how="left")

#     out_cols = ["uniprot","most_similar","qtmscore","alntmscore","ttmscore",
#                 "lddt","qcov","tcov","alnlen","evalue","bits"]
#     best[out_cols].to_csv(out_csv, index=False)
#     print(f"Wrote {out_csv} ({len(best)} rows)")

def write_best_csv(all_query_ids, df_merged: pd.DataFrame, out_csv: str):
    # Ensure IDs are strings
    all_query_ids = [str(x) for x in (all_query_ids or [])]

    # Rank within each query, preferring single-domain coverage
    if not df_merged.empty:
        df_sorted = df_merged.copy()
        # Ensure merge key is object dtype
        if "uniprot" in df_sorted.columns:
            df_sorted["uniprot"] = df_sorted["uniprot"].astype("object")
        df_sorted = df_sorted.sort_values(
            ["uniprot", "qtmscore", "alntmscore", "lddt"],
            ascending=[True, False, False, False],
        )
        best = df_sorted.groupby("uniprot", as_index=False).first()
    else:
        # Create a correctly typed empty frame
        best = pd.DataFrame({
            "uniprot": pd.Series(dtype="object"),
            "most_similar": pd.Series(dtype="object"),
            "qtmscore": pd.Series(dtype="float64"),
            "alntmscore": pd.Series(dtype="float64"),
            "ttmscore": pd.Series(dtype="float64"),
            "lddt": pd.Series(dtype="float64"),
            "qcov": pd.Series(dtype="float64"),
            "tcov": pd.Series(dtype="float64"),
            "alnlen": pd.Series(dtype="float64"),
            "evalue": pd.Series(dtype="float64"),
            "bits": pd.Series(dtype="float64"),
        })

    # Ensure every query appears in the final CSV
    allq = pd.DataFrame({"uniprot": pd.Series(sorted(all_query_ids), dtype="object")})
    best = allq.merge(best, on="uniprot", how="left")

    out_cols = ["uniprot","most_similar","qtmscore","alntmscore","ttmscore",
                "lddt","qcov","tcov","alnlen","evalue","bits"]
    best[out_cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(best)} rows)")
# ...existing code...

def create_subdb_from_ids(foldseek_bin: str, query_db: str, ids: list, out_prefix: str):
    """Use Foldseek/MMseqs 'createsubdb' to subset a DB to specific IDs."""
    ids_path = out_prefix + ".ids"
    with open(ids_path, "w") as f:
        for i in ids:
            f.write(str(i) + "\n")

    cmd = [foldseek_bin, "createsubdb", ids_path, query_db, out_prefix]
    print("\nCreating sub-DB:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("createsubdb failed, will fall back to re-running full DB for pass-2.")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        return False
    if res.stdout.strip():
        print("createsubdb STDOUT:\n", res.stdout)
    if res.stderr.strip():
        print("createsubdb STDERR:\n", res.stderr)
    return True

# ---------- main pipeline ----------

