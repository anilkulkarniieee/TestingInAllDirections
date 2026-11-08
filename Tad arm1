#!/usr/bin/env python3
"""
TAD Arm 1: public-data estimate of test-to-production correlation (TPC).

For each merged PR c in AIDev-pop (or any CSV of PRs you supply):
  F(c) = 1 if any check run / commit status on any PR commit reported failure
         BEFORE merge (retries and overridden failures count).
  E(c) = 1 if, within W days after merge, a revert commit names c's merge
         commit (or the PR number) in the same repository.
Outputs: TPC (Matthews correlation) with bootstrap CI, escape rate ER,
base rates, and per-agent strata.

Usage
  python tad_arm1.py --selftest
  export GITHUB_TOKEN=ghp_xxx            # 5,000 requests/hour
  python tad_arm1.py collect --out prs.sqlite --sample 2000 --window 14
  python tad_arm1.py report  --db prs.sqlite

Column names for the AIDev "pull_request" table can differ by release; edit
COLS below after inspecting the parquet.

v2 changes (after validating a 20k-sample run)
  * commit listing is now paginated (v1 read only 100 commits per window, which
    silently missed reverts in busy repositories)
  * two escape definitions: E = revert, E_fix = revert OR a later fix/hotfix/
    regression commit that references the PR number
  * stores merge_sha, merged_at and a `truncated` flag
  * `recheck` upgrades a v1 database in place (resumable)
  * per-agent rows print only when there are >= 10 escapes
Escape proxy still UNDER-counts escapes: treat results as a bound.
"""
import argparse, datetime as dt, math, os, re, sqlite3, sys, time
import numpy as np

COLS = dict(id="id", number="number", repo="repo_url", agent="agent",
            merged_at="merged_at", state="state")
AIDEV_PR = "hf://datasets/hao-li/AIDev/pull_request.parquet"
API = "https://api.github.com"


# ---------- metrics (pure functions; tested by --selftest) ----------
def confusion(F, E):
    F, E = np.asarray(F, bool), np.asarray(E, bool)
    return (int((F & E).sum()), int((F & ~E).sum()),
            int((~F & E).sum()), int((~F & ~E).sum()))  # TP, FP, FN, TN


def mcc(tp, fp, fn, tn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float("nan") if den == 0 else (tp * tn - fp * fn) / den


def escape_rate(F, E):
    F, E = np.asarray(F, bool), np.asarray(E, bool)
    return float((~F & E).sum() / len(F)) if len(F) else float("nan")


def bootstrap_ci(F, E, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    F, E = np.asarray(F, bool), np.asarray(E, bool)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(F), len(F))
        v = mcc(*confusion(F[i], E[i]))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def summarize(F, E, label="all"):
    tp, fp, fn, tn = confusion(F, E)
    lo, hi = bootstrap_ci(F, E)
    return (f"{label:>10}: N={len(F):6d}  P(F)={np.mean(F):.3f}  P(E)={np.mean(E):.4f}  "
            f"TP={tp} FP={fp} FN={fn} TN={tn}  TPC={mcc(tp,fp,fn,tn):+.3f} "
            f"[{lo:+.3f},{hi:+.3f}]  ER={escape_rate(F,E):.4f}")


# ---------- GitHub access ----------
def _session():
    import requests
    s = requests.Session()
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    s.headers["Accept"] = "application/vnd.github+json"
    return s


def gh(s, url, params=None):
    while True:
        r = s.get(url if url.startswith("http") else API + url, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time(), 5)
            print(f"rate limited; sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait + 2)
            continue
        if r.status_code in (404, 409, 451):
            return None
        r.raise_for_status()
        return r.json()


def repo_slug(url):
    m = re.search(r"github\.com/(?:repos/)?([^/]+/[^/]+)", url)
    return m.group(1).removesuffix(".git") if m else None


def pre_merge_failure(s, slug, number):
    """F(c): any failed check run or status on any PR commit."""
    commits = gh(s, f"/repos/{slug}/pulls/{number}/commits", {"per_page": 100}) or []
    for c in commits:
        sha = c["sha"]
        cr = gh(s, f"/repos/{slug}/commits/{sha}/check-runs", {"per_page": 100}) or {}
        if any(x.get("conclusion") in ("failure", "timed_out", "cancelled", "action_required")
               for x in cr.get("check_runs", [])):
            return 1
        st = gh(s, f"/repos/{slug}/commits/{sha}/status") or {}
        if any(x.get("state") in ("failure", "error") for x in st.get("statuses", [])):
            return 1
    return 0


def classify(commits, number, merge_sha, t0, t1):
    """Pure function. Returns (E_revert, E_fix) for commits inside [t0, t1]."""
    rev = fix = 0
    for c in commits:
        when = dt.datetime.fromisoformat(c["commit"]["committer"]["date"].replace("Z", "+00:00"))
        if not (t0 <= when <= t1) or c.get("sha") == merge_sha:
            continue
        msg = c["commit"]["message"]
        first = msg.splitlines()[0] if msg else ""
        if re.search(rf"\(#{number}\)\s*$", first):      # the PR's own squash commit
            continue
        refs = (merge_sha and merge_sha[:7] in msg) or re.search(rf"(?<![\w]){'#'}{number}\b", msg)
        if re.match(r"(?i)\s*revert", msg) and refs:
            rev = 1
        elif refs and re.search(r"(?i)\b(fix|fixes|fixed|hotfix|bugfix|regression)\b", first):
            fix = 1
    return rev, fix


def escapes_in_window(s, slug, number, merge_sha, merged_at, window_days, cache, max_pages=10):
    """Returns (E_revert, E_fix, truncated). One paginated listing per repo per merge date."""
    t0 = dt.datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    t1 = t0 + dt.timedelta(days=window_days)
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    key = (slug, day.date().isoformat())
    if key not in cache:
        commits, trunc = [], 0
        for p in range(1, max_pages + 1):
            d = gh(s, f"/repos/{slug}/commits",
                   {"since": day.isoformat(),
                    "until": (day + dt.timedelta(days=window_days + 1)).isoformat(),
                    "per_page": 100, "page": p}) or []
            commits += d
            if len(d) < 100:
                break
        else:
            trunc = 1                       # still full after max_pages
        cache[key] = (commits, trunc)
    commits, trunc = cache[key]
    return (*classify(commits, number, merge_sha, t0, t1), trunc)


# ---------- pipeline ----------
def collect(a):
    import pandas as pd
    df = pd.read_parquet(AIDEV_PR)
    missing = [v for v in COLS.values() if v not in df.columns]
    if missing:
        sys.exit(f"Edit COLS: missing {missing}. Available: {list(df.columns)}")
    df = df[df[COLS["merged_at"]].notna()].sample(frac=1, random_state=1).head(a.sample)
    db = sqlite3.connect(a.out)
    db.execute("create table if not exists pr(id, slug, number, agent, F, E, E_fix, merge_sha, merged_at, truncated)")
    done = {r[0] for r in db.execute("select id from pr")}
    s, cache = _session(), {}
    for _, r in df.iterrows():
        if r[COLS["id"]] in done:
            continue
        slug, n = repo_slug(r[COLS["repo"]]), int(r[COLS["number"]])
        if not slug:
            continue
        pr = gh(s, f"/repos/{slug}/pulls/{n}")
        if not pr or not pr.get("merged"):
            continue
        F = pre_merge_failure(s, slug, n)
        E, EF, tr = escapes_in_window(s, slug, n, pr.get("merge_commit_sha"),
                                      pr["merged_at"], a.window, cache)
        db.execute("insert into pr values (?,?,?,?,?,?,?,?,?,?)",
                   (r[COLS["id"]], slug, n, r[COLS["agent"]], F, E, EF,
                    pr.get("merge_commit_sha"), pr["merged_at"], tr))
        db.commit()


def recheck(a):
    """Upgrade a v1 database: add columns, then recompute E/E_fix with pagination."""
    db = sqlite3.connect(a.db)
    cols = {r[1] for r in db.execute("pragma table_info(pr)")}
    for c in ("E_fix", "merge_sha", "merged_at", "truncated"):
        if c not in cols:
            db.execute(f"alter table pr add column {c}")
    db.commit()
    todo = db.execute("select id, slug, number from pr where merged_at is null").fetchall()
    print(f"rechecking {len(todo)} PRs", file=sys.stderr)
    s, cache = _session(), {}
    for i, (pid, slug, n) in enumerate(todo):
        pr = gh(s, f"/repos/{slug}/pulls/{n}")
        if not pr or not pr.get("merged_at"):
            db.execute("update pr set merged_at='unavailable' where id=?", (pid,)); continue
        E, EF, tr = escapes_in_window(s, slug, n, pr.get("merge_commit_sha"),
                                      pr["merged_at"], a.window, cache)
        db.execute("update pr set E=?, E_fix=?, merge_sha=?, merged_at=?, truncated=? where id=?",
                   (E, EF, pr.get("merge_commit_sha"), pr["merged_at"], tr, pid))
        if i % 200 == 0:
            db.commit()
    db.commit()


def report(a):
    rows = sqlite3.connect(a.db).execute(
        "select agent, F, E, coalesce(E_fix,0), coalesce(truncated,0) from pr "
        "where coalesce(merged_at,'') != 'unavailable'").fetchall()
    if not rows:
        sys.exit("no rows")
    agents = np.array([r[0] for r in rows])
    F = np.array([r[1] for r in rows]); E = np.array([r[2] for r in rows])
    EB = np.array([bool(r[2] or r[3]) for r in rows]); T = np.array([r[4] for r in rows])
    print(f"rows={len(rows)}  windows truncated after pagination: {int(T.sum())}")
    print("-- E = revert only")
    print(summarize(F, E))
    print("-- E_broad = revert OR fix/hotfix/regression commit referencing the PR")
    print(summarize(F, EB))
    for name, ev in (("E", E), ("E_broad", EB)):
        p1 = ev[F == 1].mean() if (F == 1).any() else float("nan")
        p0 = ev[F == 0].mean() if (F == 0).any() else float("nan")
        print(f"   {name}: P(E|F=1)={p1:.4f}  P(E|F=0)={p0:.4f}  RR=P(E|F=0)/P(E|F=1)="
              f"{(p0 / p1) if p1 else float('nan'):.2f}")
    print("-- per agent (E_broad; shown only with >= 10 escapes)")
    for ag in sorted(set(agents)):
        m = agents == ag
        if EB[m].sum() >= 10:
            print(summarize(F[m], EB[m], str(ag)[:10]))
        else:
            print(f"{str(ag)[:10]:>10}: N={m.sum()} escapes={int(EB[m].sum())} (too few for a CI)")


def selftest():
    rng = np.random.default_rng(0)
    # independent F and E -> TPC ~ 0
    F = rng.random(20000) < 0.25
    E = rng.random(20000) < 0.01
    v = mcc(*confusion(F, E)); assert abs(v) < 0.05, v
    # perfectly coupled -> TPC = 1
    assert abs(mcc(*confusion(F, F)) - 1) < 1e-9
    # ER counts only escapes among changes that passed everything
    assert abs(escape_rate([0, 0, 1, 1], [1, 0, 1, 0]) - 0.25) < 1e-9
    # degenerate (no escapes) -> NaN, not a crash
    assert math.isnan(mcc(*confusion(F, np.zeros_like(F))))
    # classifier: revert, fix-reference, own squash commit ignored, out-of-window ignored
    t0 = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc); t1 = t0 + dt.timedelta(days=14)
    mk = lambda m, d, sha="x": {"sha": sha, "commit": {"message": m,
         "committer": {"date": (t0 + dt.timedelta(days=d)).isoformat()}}}
    assert classify([mk('Revert "Add x (#42)" (#50)', 2)], 42, "abcdef123", t0, t1) == (1, 0)
    assert classify([mk("fix: regression from #42", 3)], 42, "abcdef123", t0, t1) == (0, 1)
    assert classify([mk("Add x (#42)", 0)], 42, "abcdef123", t0, t1) == (0, 0)
    assert classify([mk('Revert "Add x (#42)" (#50)', 20)], 42, "abcdef123", t0, t1) == (0, 0)
    assert classify([mk("fix: unrelated #421", 3)], 42, "abcdef123", t0, t1) == (0, 0)
    print(summarize(F, E, "independent"))
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    c = sub.add_parser("collect"); c.add_argument("--out", default="prs.sqlite")
    c.add_argument("--sample", type=int, default=2000); c.add_argument("--window", type=int, default=14)
    r = sub.add_parser("report"); r.add_argument("--db", default="prs.sqlite")
    k = sub.add_parser("recheck"); k.add_argument("--db", default="prs.sqlite")
    k.add_argument("--window", type=int, default=14)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or a.cmd == "selftest": selftest()
    elif a.cmd == "collect": collect(a)
    elif a.cmd == "recheck": recheck(a)
    elif a.cmd == "report": report(a)
    else: p.print_help()
