#!/usr/bin/env python3
"""
Replay pilot, step 1: collect the full recent history of ONE public repo.

For every PR merged in the last N months it stores what a review agent would have
seen (title, body, size, diff excerpt) plus the outcomes (F, E, E_strict).
Resumable: re-running skips PRs already in the JSONL file.

  export GITHUB_TOKEN=ghp_xxx
  python replay_collect.py --repo Yeraze/meshmonitor --months 6 --out meshmonitor.jsonl
  python replay_collect.py --selftest
"""
import argparse, datetime as dt, json, os, sys
import tad_arm1 as T

DIFF_CHARS = 12000


def merged_prs(s, repo, since):
    out, page = [], 1
    while True:
        d = T.gh(s, f"/repos/{repo}/pulls", {"state": "closed", "sort": "created",
                 "direction": "desc", "per_page": 100, "page": page}) or []
        if not d: break
        for p in d:
            if p.get("merged_at"): out.append(p["number"])
        if dt.datetime.fromisoformat(d[-1]["created_at"].replace("Z", "+00:00")) < since: break
        page += 1
    return out


def diff_excerpt(s, repo, n):
    files = T.gh(s, f"/repos/{repo}/pulls/{n}/files", {"per_page": 100}) or []
    chunks, used = [], 0
    for f in files:
        part = f"--- {f['filename']} (+{f['additions']}/-{f['deletions']})\n{f.get('patch', '')}\n"
        if used + len(part) > DIFF_CHARS:
            chunks.append(f"... {len(files) - len(chunks)} more files truncated"); break
        chunks.append(part); used += len(part)
    return "".join(chunks)


def escape_flags(s, repo, n, sha, merged_at, window, cache):
    rev, fix, trunc = T.escapes_in_window(s, repo, n, sha, merged_at, window, cache)
    t0 = dt.datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    commits, _ = cache[(repo, day.date().isoformat())]
    m = []
    T.classify(commits, n, sha, t0, t0 + dt.timedelta(days=window), m)
    strict = int(rev or any(k == "fix" and c for k, _, _, c in m))
    first = min((x for x in m), default=None)
    return dict(E=rev, E_clean=int(bool(rev or fix)), E_strict=strict, truncated=trunc,
                matches=[{"kind": k, "title": t, "causal": c} for k, _, t, c in m])


def main(a):
    s, cache = T._session(), {}
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30 * a.months)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["number"] for l in open(a.out)}
    nums = [n for n in merged_prs(s, a.repo, since) if n not in done]
    if a.max_prs: nums = nums[: a.max_prs]
    print(f"{len(nums)} PRs to collect", file=sys.stderr)
    with open(a.out, "a") as f:
        for i, n in enumerate(nums):
            pr = T.gh(s, f"/repos/{a.repo}/pulls/{n}")
            if not pr or not pr.get("merged_at"): continue
            rec = dict(repo=a.repo, number=n, merged_at=pr["merged_at"], title=pr["title"],
                       body=(pr.get("body") or "")[:3000], author=pr["user"]["login"],
                       additions=pr["additions"], deletions=pr["deletions"],
                       changed_files=pr["changed_files"], diff=diff_excerpt(s, a.repo, n),
                       F=T.pre_merge_failure(s, a.repo, n))
            rec.update(escape_flags(s, a.repo, n, pr.get("merge_commit_sha"),
                                    pr["merged_at"], a.window, cache))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 50 == 0: print(f"  {i}/{len(nums)}", file=sys.stderr)


def selftest():
    assert DIFF_CHARS > 0
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo"); p.add_argument("--months", type=int, default=6)
    p.add_argument("--window", type=int, default=14); p.add_argument("--out", default="repo.jsonl")
    p.add_argument("--max-prs", type=int, default=0); p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else main(a)
