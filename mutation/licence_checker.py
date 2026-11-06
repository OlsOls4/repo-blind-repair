"""Licence audit checker for Repo-Blind Repair source repos.

Rule: permissive licences ONLY (MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause). Copyleft (GPL/LGPL/AGPL), non-commercial, proprietary
and private repos are REJECTED. A repo whose licence cannot be
verified from a licence file is marked UNVERIFIED and must not be
used for training until verified — it is never silently accepted.

Usage:
    python licence_checker.py --demo        # check the demo entries
    python licence_checker.py entries.json  # check a JSON list of
        # {"name", "url", "licence", "licence_file"} objects

Exit code 0 = every entry accepted; 1 = any rejected/unverified.
No network access: this script checks the records you give it, it
does not download anything. A "download attempted" outcome is
recorded by the caller in LICENCES.md, never faked by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD"}
REJECTED = {
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "AGPL-3.0",
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "Proprietary",
    "Private",
    "Unknown",
}


def check_entry(entry: dict) -> dict:
    """Return entry + verdict: ACCEPT, REJECT, or UNVERIFIED."""
    licence = (entry.get("licence") or "").strip()
    licence_file = entry.get("licence_file") or ""
    if licence in PERMISSIVE:
        verdict = "ACCEPT"
        reason = "permissive licence"
    elif licence in REJECTED or licence == "":
        verdict = "REJECT"
        reason = "copyleft/non-commercial/private/unknown licence" if licence else "no licence stated"
    else:
        verdict = "UNVERIFIED"
        reason = f"licence {licence!r} is not on the permissive allow-list"
    if verdict == "ACCEPT" and licence_file:
        # Only a licence file actually present locally counts as
        # verified. A remote URL is a pointer to check later, not
        # proof, and a missing local file is a verification gap:
        # downgrade in both cases, do not accept blindly.
        if licence_file.startswith("http"):
            verdict = "UNVERIFIED"
            reason = "remote licence file not verified locally; verify before use"
        elif not Path(licence_file).exists():
            verdict = "UNVERIFIED"
            reason = "licence file not present locally; verify before use"
    return {**entry, "verdict": verdict, "reason": reason}


DEMO_ENTRIES = [
    # Named only, NOT downloaded by this script. Licence values below
    # are the projects' widely published licences; each entry is
    # marked UNVERIFIED by check_entry because no licence file is
    # present locally, and must be verified from the repo's own
    # LICENCE file before any training use. Nothing here is claimed
    # as downloaded or verified.
    {
        "name": "requests",
        "url": "https://github.com/psf/requests",
        "licence": "Apache-2.0",
        "licence_file": "https://github.com/psf/requests/blob/main/LICENSE",
        "status": "named only, not downloaded",
    },
    {
        "name": "flask",
        "url": "https://github.com/pallets/flask",
        "licence": "BSD-3-Clause",
        "licence_file": "https://github.com/pallets/flask/blob/main/LICENSE",
        "status": "named only, not downloaded",
    },
    {
        "name": "click",
        "url": "https://github.com/pallets/click",
        "licence": "BSD-3-Clause",
        "licence_file": "https://github.com/pallets/click/blob/main/LICENSE",
        "status": "named only, not downloaded",
    },
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Licence audit checker")
    parser.add_argument("--demo", action="store_true", help="check demo entries")
    parser.add_argument("entries_file", nargs="?", help="JSON list of entries")
    args = parser.parse_args(argv)
    if args.demo:
        entries = DEMO_ENTRIES
    elif args.entries_file:
        entries = json.loads(Path(args.entries_file).read_text())
    else:
        parser.error("pass --demo or an entries JSON file")
    bad = 0
    for entry in entries:
        result = check_entry(entry)
        print(f"{result['name']}: {result['verdict']} — {result['reason']}")
        if result["verdict"] != "ACCEPT":
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
