#!/usr/bin/env python3
"""
Pick PUBLIC GitHub repositories for a retrospective TAD pilot.

Two candidate sources
  --source aidev   repos that already have agent-authored PRs in AIDev
                   (ties the pilot to Arm 1 and to AI-generated tests)
  --source search  GitHub repository search (stars, language, recent push)

Each candidate is enriched through the GitHub API and scored with the same
rules as pick_pilot.py (activity, observable escapes, CI, team size), using
releases as the deployment signal because public repos rarely expose
Deployments. Large open-source teams are allowed (default max 50 authors).

  export GITHUB_TOKEN=ghp_xxx
  python pick_public_pilot.py --source aidev --top 10 --csv shortlist.csv
  python pick_public_pilot.py --source search --lang python --min-stars 2000
  python pick_public_pilot.py --selftest

IMPORTANT: on a public repo you can only REPLAY history (layers 1-2: review
and build agents run offline on past PRs, scored against later reverts and
hotfixes). Pre-production and production layers cannot be exercised. Never
post agent comments on maintainers' live PRs.
"""
import argparse, csv, datetime as dt, os, re, sys
from pick_pilot import _get, pages, metrics, score, pick_pairs

AIDEV_PR = "hf://datasets/hao-li/AIDev/pull_request.parquet"
COLS = dict(repo="repo_url", merged_at="merged_at", agent="agent")


def slug(url):
    m = re.search(r"github\.com/(?:repos/)?([^/]+/[^/]+)", str(url))
    return m.group(1).removesuffix(".git") if m else None


def aidev_candidates(min_agent_prs, limit):
    import pandas as pd
    df = pd.read_parquet(AIDEV_PR)
    df["slug"] = df[COLS["repo"]].map(slug)
    g = (df[df[COLS["merged_at"]].notna()].groupby("slug")
         .agg(agent_prs=(COLS["agent"], "size"), agents=(COLS["agent"], "nunique"))
         .query("agent_prs >= @min_agent_prs").sort_values("agent_prs", ascending=False))
    return [(s, int(r.agent_prs)) for s, r in g.head(limit).iterrows()]


def search_candidates(s, lang, min_stars, limit):
    since = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    q = f"stars:>{min_stars} pushed:>{since} archived:false fork:false"
    if lang: q += f" language:{lang}"
    out = []
    for p in range(1, 4):
        d = _get(s, "/search/repositories", {"q": q, "sort": "stars", "per_page": 50, "page": p})
        out += [(r["full_name"], 0) for r in (d or {}).get("items", [])]
    return out[:limit]


def releases_90d(s, owner, name):
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90)
    rel = _get(s, f"/repos/{owner}/{name}/releases", {"per_page": 100}) or []
    return sum(dt.datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")) >= t
               for r in rel if r.get("published_at"))


def run(a):
    import requests
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                      "Accept": "application/vnd.github+json"})
    cands = (aidev_candidates(a.min_agent_prs, a.scan) if a.source == "aidev"
             else search_candidates(s, a.lang, a.min_stars, a.scan))
    rows = []
    for full, agent_prs in cands:
        owner, name = full.split("/")
        repo = _get(s, f"/repos/{full}") or {}
        if repo.get("archived") or repo.get("fork"): continue
        try:
            m = metrics(s, owner, {"name": name, "language": repo.get("language")})
        except Exception as e:
            print(f"skip {full}: {e}", file=sys.stderr); continue
        m["name"] = full
        m["agent_prs"] = agent_prs
        m["stars"] = repo.get("stargazers_count", 0)
        m["prod_deploys"] = max(m["prod_deploys"], releases_90d(s, owner, name))
        m["score"], m["why"] = score(m, max_authors=a.max_authors)
        rows.append(m)
        print(f"scanned {full}: score {m['score']}", file=sys.stderr)
    print(f"\n{'pilot repo':38}{'score':>6}{'merged90':>9}{'escapes':>8}{'authors':>8}{'agentPRs':>9}  comparison")
    for c, o in pick_pairs(rows, a.top):
        print(f"{c['name']:38}{c['score']:6}{c['merged_90d']:9}{c['escapes_365d']:8}"
              f"{c['authors_90d']:8}{c['agent_prs']:9}  {o['name'] if o else '-'}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "language", "stars", "score", "merged_90d",
                                              "escapes_365d", "authors_90d", "has_ci",
                                              "prod_deploys", "agent_prs"], extrasaction="ignore")
            w.writeheader(); w.writerows(sorted(rows, key=lambda r: -r["score"]))
    print("\nexcluded:")
    for c in rows:
        if c["why"]: print(f"  {c['name']}: {c['why'][0]}")


def selftest():
    assert slug("https://api.github.com/repos/foo/bar") == "foo/bar"
    assert slug("https://github.com/foo/bar.git") == "foo/bar"
    m = dict(name="x/y", language="go", merged_90d=300, escapes_365d=25, authors_90d=35,
             has_ci=True, prod_deploys=12)
    assert score(m)[0] == 0                      # 35 authors fails the default
    assert score(m, max_authors=50)[0] > 40      # allowed for open-source scale
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["aidev", "search"], default="aidev")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--scan", type=int, default=60, help="candidates to enrich")
    p.add_argument("--min-agent-prs", type=int, default=20)
    p.add_argument("--lang"); p.add_argument("--min-stars", type=int, default=1000)
    p.add_argument("--max-authors", type=int, default=50)
    p.add_argument("--csv"); p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else run(a)
