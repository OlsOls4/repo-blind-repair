#!/usr/bin/env python3
"""Near-duplicate audit: held-out tasks vs training sources.

Compares each of the 116 held-out tasks (files["task.py"]) against the
training corpus (train_records_20k.jsonl) using token-trigram containment:
for a held-out task T and a training document D,
    containment(T, D) = |trigrams(T) ∩ trigrams(D)| / |trigrams(T)|.
Containment (not Jaccard) is the decontamination question: how much of the
held-out task is already covered by a single training document.

Tokens are Python `tokenize` NAME / NUMBER / OP / STRING values (comments,
whitespace, INDENT/DEDENT excluded), so `//` is one token and formatting
differences do not count. Training documents are deduplicated by exact
source string first (20,000 records collapse to the unique sources), so
counts are over unique sources, not record copies.

Also reproduces the exact-match check: whether any held-out task.py string
exactly equals a training fixed_code or buggy_code string.

Deterministic (no randomness, no network). Pure standard library.
Writes kaggle/audit_near_dup_results.json and prints a summary + sha256.
"""
import hashlib
import io
import json
import re
import sys
import time
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train_records_20k.jsonl"
HELDOUT_PATH = ROOT / "dev_tasks_heldout.jsonl"
OUT_PATH = ROOT / "audit_near_dup_results.json"

TOKEN_TYPES = (tokenize.NAME, tokenize.NUMBER, tokenize.OP, tokenize.STRING)


def tokenize_code(code):
    """Return the list of significant tokens of a Python source string."""
    toks = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in TOKEN_TYPES:
                toks.append(tok.string)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        # Fall back to a regex split so an unparsable file still yields tokens.
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s\w]", code)
    return toks


def trigrams(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def main():
    t0 = time.time()
    train_records = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    heldout = [json.loads(l) for l in open(HELDOUT_PATH, encoding="utf-8")]

    # Deduplicate training sources by exact string.
    fixed_unique = sorted({r["fixed_code"] for r in train_records})
    buggy_unique = sorted({r["buggy_code"] for r in train_records})
    fixed_set = set(fixed_unique)
    buggy_set = set(buggy_unique)

    # Exact-match check (reproduces the paper's earlier claim).
    exact_fixed = sum(1 for t in heldout if t["files"]["task.py"] in fixed_set)
    exact_buggy = sum(1 for t in heldout if t["files"]["task.py"] in buggy_set)

    def build_index(docs):
        """trigram -> set of doc indices; also returns per-doc trigram sets."""
        doc_tris = [trigrams(tokenize_code(d)) for d in docs]
        index = defaultdict(set)
        for i, ts in enumerate(doc_tris):
            for tri in ts:
                index[tri].add(i)
        return doc_tris, index

    fixed_tris, fixed_index = build_index(fixed_unique)
    buggy_tris, buggy_index = build_index(buggy_unique)

    def max_containment(task_tris, doc_tris, index):
        """Top-3 (containment, doc_idx, shared) for one task against one index."""
        counts = Counter()
        for tri in task_tris:
            for doc_idx in index.get(tri, ()):
                counts[doc_idx] += 1
        n = len(task_tris)
        if n == 0 or not counts:
            return []
        ranked = sorted(
            ((c / n, doc_idx, c) for doc_idx, c in counts.items()),
            key=lambda x: (-x[0], x[1]),
        )
        return [{"containment": round(c, 4), "doc_index": i, "shared_trigrams": s,
                 "doc_trigrams": len(doc_tris[i])} for c, i, s in ranked[:3]]

    per_task = []
    for t in heldout:
        task_code = t["files"]["task.py"]
        task_tris = trigrams(tokenize_code(task_code))
        best_fixed = max_containment(task_tris, fixed_tris, fixed_index)
        best_buggy = max_containment(task_tris, buggy_tris, buggy_index)
        per_task.append({
            "task_id": t["task_id"],
            "bug_class": t.get("bug_class"),
            "n_trigrams": len(task_tris),
            "max_containment_fixed": best_fixed[0]["containment"] if best_fixed else 0.0,
            "max_containment_buggy": best_buggy[0]["containment"] if best_buggy else 0.0,
            "best_fixed": best_fixed,
            "best_buggy": best_buggy,
        })

    def dist(key):
        vals = [p[key] for p in per_task]
        return {
            "max": max(vals),
            "median": sorted(vals)[len(vals) // 2],
            "count_ge_0.5": sum(1 for v in vals if v >= 0.5),
            "count_ge_0.8": sum(1 for v in vals if v >= 0.8),
            "count_ge_0.9": sum(1 for v in vals if v >= 0.9),
            "count_ge_1.0": sum(1 for v in vals if v >= 1.0),
        }

    top_pairs = []
    for p in per_task:
        for kind in ("best_fixed", "best_buggy"):
            for b in p[kind][:1]:
                top_pairs.append({
                    "task_id": p["task_id"],
                    "kind": "fixed" if kind == "best_fixed" else "buggy",
                    "containment": b["containment"],
                    "doc_index": b["doc_index"],
                    "shared_trigrams": b["shared_trigrams"],
                })
    top_pairs.sort(key=lambda x: -x["containment"])
    top_pairs = top_pairs[:5]

    results = {
        "method": "token-trigram containment (tokenize NAME/NUMBER/OP/STRING; "
                  "containment = shared trigrams / held-out task trigrams)",
        "corpus_note": "The generator rebuilt 2,126 unique sources "
                  "(generate_scaled_dataset.py build_sources()); the 20,000 "
                  "records draw from 1,166 of them (distinct source_index in "
                  "the provenance sidecar, indices 0..1573). The audit "
                  "compares against the 1,166 sources actually present, "
                  "deduplicated by exact fixed_code string.",
        "train_path": str(TRAIN_PATH),
        "heldout_path": str(HELDOUT_PATH),
        "n_train_records": len(train_records),
        "n_unique_fixed_sources": len(fixed_unique),
        "n_unique_buggy_sources": len(buggy_unique),
        "n_heldout_tasks": len(heldout),
        "exact_match_heldout_vs_fixed": exact_fixed,
        "exact_match_heldout_vs_buggy": exact_buggy,
        "distribution_fixed": dist("max_containment_fixed"),
        "distribution_buggy": dist("max_containment_buggy"),
        "top_pairs": top_pairs,
        "per_task": per_task,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"Wrote {OUT_PATH}")
    print(f"sha256: {sha}")
    print(f"train records: {len(train_records)}; unique fixed: {len(fixed_unique)}; "
          f"unique buggy: {len(buggy_unique)}; held-out: {len(heldout)}")
    print(f"exact matches vs fixed: {exact_fixed}; vs buggy: {exact_buggy}")
    print(f"fixed containment distribution: {results['distribution_fixed']}")
    print(f"buggy containment distribution: {results['distribution_buggy']}")
    print(f"top pairs: {json.dumps(top_pairs, indent=2)}")
    print(f"runtime: {results['runtime_seconds']}s")


if __name__ == "__main__":
    sys.exit(main())
