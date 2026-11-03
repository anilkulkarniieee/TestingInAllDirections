#!/usr/bin/env python3
"""
Rank repositories in a GitHub org as TAD pilot candidates (Appendix B, step 1).

A good pilot repo is active enough to produce data, has enough *observable*
escapes to link to changes, has CI, and has a small, stable team. Then pick a
comparison repo that matches on size, language and activity.

  export GITHUB_TOKEN=ghp_xxx          # needs read access to the org's repos
  python pick_pilot.py --org my-org --top 10
  python pick_pilot.py --selftest

Signals (last 365 days unless noted)
  merged_90d     merged PRs in the last 90 days (proxy for deployment volume)
  escapes_365d   merged PRs titled Revert*, plus issues/PRs labelled
                 incident|outage|postmortem|hotfix|sev
  authors_90d    distinct PR authors in the last 90 days (team size)
  has_ci         GitHub Actions workflows present
  prod_deploys   deployments to an environment named like prod* (if used)
Treat the output as a shortlist; confirm with the owning team and with the
incident tracker, which may hold escapes that GitHub does not.
"""
import argparse, datetime as dt, math, os, re, sys, time

ESC_LABEL = re.compile(r"incident|outage|postmortem|hotfix|sev[-_ ]?\d?", re.I)
API = "https://api.github.com"


def score(m, max_authors=15):
    """0-100. Hard requirements zero the score and explain why."""
    why = []
    if m["merged_90d"] < 60:       why.append("fewer than 20 merges/month")
    if m["escapes_365d"] < 6:      why.append("fewer than 6 linkable escapes/year")
    if not m["has_ci"]:            why.append("no CI workflows")
    if not 3 <= m["authors_90d"] <= max_authors: why.append(f"team size outside 3-{max_authors}")
    if why:
        return 0.0, why
    s = 0.0
    s += 35 * min(math.log10(m["merged_90d"] / 60 + 1) / math.log10(4), 1)   # activity
    s += 35 * min(m["escapes_365d"] / 30, 1)                                  # observable escapes
    s += 15 * (1 if m["prod_deploys"] >= 20 else 0.5 if m["prod_deploys"] else 0)
    s += 15 * max(0.0, 1 - abs(m["authors_90d"] - 8) / 8)                # team near 8
    return round(s, 1), []


def match_distance(a, b):
    """Smaller is a better comparison repo."""
    d = abs(math.log((a["merged_90d"] + 1) / (b["merged_90d"] + 1)))
    d += abs(a["authors_90d"] - b["authors_90d"]) / 10
    d += 0 if a["language"] == b["language"] else 1
    return d


def pick_pairs(cands, top):
    ranked = sorted(cands, key=lambda c: -c["score"])[:top]
    pairs = []
    for c in ranked:
        pool = [o for o in cands if o["name"] != c["name"] and o["score"] > 0]
        best = min(pool, key=lambda o: match_distance(c, o), default=None)
        pairs.append((c, best))
    return pairs


# ---------- GitHub ----------
def _get(s, path, params=None):
    import requests
    while True:
        r = s.get(path if path.startswith("http") else API + path, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
            time.sleep(max(int(r.headers["X-RateLimit-Reset"]) - time.time(), 5) + 2); continue
        if r.status_code in (404, 409): return None
        r.raise_for_status(); return r.json()


def pages(s, path, params, limit=5):
    out = []
    for p in range(1, limit + 1):
        d = _get(s, path, {**params, "per_page": 100, "page": p})
        if not d: break
        out += d
        if len(d) < 100: break
    return out


def metrics(s, org, repo):
    now = dt.datetime.now(dt.timezone.utc)
    t90, t365 = now - dt.timedelta(days=90), now - dt.timedelta(days=365)
    prs = pages(s, f"/repos/{org}/{repo['name']}/pulls",
                {"state": "closed", "sort": "updated", "direction": "desc"}, limit=10)
    merged = [p for p in prs if p.get("merged_at")]
    iso = lambda x: dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    m90 = [p for p in merged if iso(p["merged_at"]) >= t90]
    m365 = [p for p in merged if iso(p["merged_at"]) >= t365]
    reverts = sum(bool(re.match(r"(?i)revert", p["title"])) for p in m365)
    labelled = sum(any(ESC_LABEL.search(l["name"]) for l in p.get("labels", [])) for p in m365)
    issues = pages(s, f"/repos/{org}/{repo['name']}/issues",
                   {"state": "all", "since": t365.isoformat()}, limit=5)
    labelled += sum(any(ESC_LABEL.search(l["name"]) for l in i.get("labels", []))
                    and "pull_request" not in i for i in issues)
    wf = _get(s, f"/repos/{org}/{repo['name']}/actions/workflows") or {}
    dep = _get(s, f"/repos/{org}/{repo['name']}/deployments", {"per_page": 100}) or []
    prod = sum(bool(re.search(r"prod", d.get("environment", ""), re.I))
               and iso(d["created_at"]) >= t90 for d in dep)
    return dict(name=repo["name"], language=repo.get("language") or "n/a",
                merged_90d=len(m90), escapes_365d=reverts + labelled,
                authors_90d=len({p["user"]["login"] for p in m90}),
                has_ci=bool(wf.get("total_count")), prod_deploys=prod)


def run(a):
    import requests
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                      "Accept": "application/vnd.github+json"})
    repos = [r for r in pages(s, f"/orgs/{a.org}/repos", {"type": "all"}, limit=10)
             if not r["archived"] and not r["fork"]]
    cands = []
    for r in repos:
        m = metrics(s, a.org, r)
        m["score"], m["why"] = score(m)
        cands.append(m)
        print(f"scanned {m['name']}: score {m['score']}", file=sys.stderr)
    print(f"\n{'pilot':28}{'score':>6}{'merged90':>9}{'escapes':>8}{'authors':>8}   comparison repo")
    for c, o in pick_pairs(cands, a.top):
        print(f"{c['name']:28}{c['score']:6}{c['merged_90d']:9}{c['escapes_365d']:8}"
              f"{c['authors_90d']:8}   {o['name'] if o else '-'}")
    print("\nexcluded (first reason):")
    for c in cands:
        if c["why"]: print(f"  {c['name']}: {c['why'][0]}")


def selftest():
    good = dict(name="a", language="py", merged_90d=150, escapes_365d=20,
                authors_90d=8, has_ci=True, prod_deploys=40)
    s, why = score(good); assert s > 70 and not why, s
    assert score({**good, "has_ci": False})[0] == 0
    assert score({**good, "authors_90d": 40})[0] == 0
    b = {**good, "name": "b", "score": 60}; c = {**good, "name": "c", "language": "go",
                                              "score": 60}; a = {**good, "score": 90}
    (p, o), = pick_pairs([a, b, c], 1); assert o["name"] == "b"   # same language wins
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--org"); p.add_argument("--top", type=int, default=10)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest: selftest()
    elif a.org: run(a)
    else: p.print_help()
