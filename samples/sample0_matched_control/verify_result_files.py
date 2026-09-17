"""Verify the complete seed matrix and all exported raw joint-count archives."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def verify_files(out):
    out = Path(out)
    spec = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["spec"]
    nqueries = 0
    for seed in spec["seeds"]:
        for cond in spec["conditions"]:
            folder = out / f"seed_{seed}" / cond["name"]
            complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
            blob = (folder / "raw_joint_counts.jsonl.gz").read_bytes()
            if hashlib.sha256(blob).hexdigest() != complete["raw_counts_sha256"]:
                raise ValueError(f"Raw archive checksum mismatch: {folder}")
            lines = gzip.decompress(blob).decode("utf-8").splitlines()
            if len(lines) != complete["raw_query_count"] or not lines:
                raise ValueError(f"Missing raw queries: {folder}")
            seen = set()
            for line in lines:
                row = json.loads(line)
                key = row["stream"], row["index"]
                if key in seen:
                    raise ValueError(f"Duplicate raw query {key}: {folder}")
                seen.add(key)
                records = row["counts"] if row["mode"] == "sampled" else row["probabilities"]
                limit = 1 << len(row["active_qubits"])
                if any(not 0 <= i < limit or v < 0 for i, v in records):
                    raise ValueError(f"Invalid joint state/count: {folder}")
                target = spec["shots"] if row["mode"] == "sampled" else 1.0
                if abs(sum(v for _, v in records) - target) > 1e-8:
                    raise ValueError(f"Joint count/probability total mismatch: {folder}")
                nqueries += 1
    result = dict(passed=True, seed_conditions=len(spec["seeds"])*len(spec["conditions"]), raw_queries=nqueries)
    (out / "data_integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    args = ap.parse_args()
    print(verify_files(args.input_dir))
