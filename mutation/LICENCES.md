# Licence audit list — Repo-Blind Repair source repos

**Rule: permissive licences ONLY — MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause. Reject copyleft (GPL/LGPL/AGPL), non-commercial,
proprietary and private repos. A repo that cannot be verified from
its own licence file is UNVERIFIED and must NOT be used for training
until verified.**

Checker: `python licence_checker.py --demo` (demo entries) or
`python licence_checker.py entries.json`.

## Template (copy one row per repo)

| Repo | URL | Licence | Licence file location | Status |
|---|---|---|---|---|
| _name_ | _url_ | _licence_ | _path or URL to LICENCE file_ | ACCEPT / REJECT / UNVERIFIED |

## Demo entries — named only, NOT downloaded, NOT verified

These are well-known permissively-licensed Python projects, listed
here so a later step can verify each repo's own licence file before
any training use. No download was attempted by the checker script,
and none is claimed here.

| Repo | URL | Licence | Licence file location | Status |
|---|---|---|---|---|
| requests | https://github.com/psf/requests | Apache-2.0 | https://github.com/psf/requests/blob/main/LICENSE | UNVERIFIED — verify licence file before use |
| flask | https://github.com/pallets/flask | BSD-3-Clause | https://github.com/pallets/flask/blob/main/LICENSE | UNVERIFIED — verify licence file before use |
| click | https://github.com/pallets/click | BSD-3-Clause | https://github.com/pallets/click/blob/main/LICENSE | UNVERIFIED — verify licence file before use |

## Download outcomes

| Repo | Attempted? | Outcome |
|---|---|---|
| requests | No — named only | Not downloaded; no outcome claimed |
| flask | No — named only | Not downloaded; no outcome claimed |
| click | No — named only | Not downloaded; no outcome claimed |

_If a later step attempts a download and it fails, record the
failure here verbatim instead of faking a success._
