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
import argparse, csv, datetime as dt, math, os, re, sqlite3, sys, time
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


FIXWORD = re.compile(r"(?i)\b(fix|fixes|fixed|hotfix|bugfix|regression)\b")
NOT_A_FIX = re.compile(r"(?i)(\bchangelog\b|release notes|^\s*(docs?|chore\(release\)|chore: release|release)\b)")


def causal_ref(msg, number):
    """True if the message ties a fix/regression to this PR number (stricter than a bare mention)."""
    n = rf"#{number}\b"
    kw = r"(regression|broke|broken|caused by|introduced|follow[- ]?up|hotfix|fix(?:es|ed)? (?:a |the )?(?:bug|issue|crash|problem))"
    return bool(re.search(rf"(?i)\b{kw}\b[^\n]{{0,80}}{n}", msg) or
                re.search(rf"(?i){n}[^\n]{{0,40}}\b(regression|broke|broken|hotfix)\b", msg))


def classify(commits, number, merge_sha, t0, t1, matches=None):
    """Pure function. Returns (E_revert, E_fix) for commits inside [t0, t1].
    Merge commits (including the PR's own), docs/release/changelog commits and the PR's own
    squash commit are ignored, except that revert merges are still recognised.
    If `matches` is a list, (kind, sha, title, causal) tuples are appended for audits."""
    rev = fix = 0
    for c in commits:
        when = dt.datetime.fromisoformat(c["commit"]["committer"]["date"].replace("Z", "+00:00"))
        if not (t0 <= when <= t1) or c.get("sha") == merge_sha:
            continue
        msg = c["commit"]["message"]
        first = msg.splitlines()[0] if msg else ""
        if re.search(rf"\(#{number}\)\s*$", first):      # the PR's own squash commit
            continue
        refs = (merge_sha and merge_sha[:7] in msg) or re.search(rf"(?<![\w])#{number}\b", msg)
        is_merge = bool(re.match(r"(?i)\s*merge\b", first))
        is_revert = bool(re.match(r"(?i)\s*revert", msg) or
                         (is_merge and (re.search(r"(?im)^\s*revert\b", msg) or re.search(r"/revert[-_]", first))))
        if is_revert and refs:
            rev = 1
            if matches is not None: matches.append(("revert", c.get("sha"), first, 1))
        elif is_merge or NOT_A_FIX.search(first):
            continue
        elif refs and FIXWORD.search(first):
            fix = 1
            if matches is not None: matches.append(("fix", c.get("sha"), first, int(causal_ref(msg, number))))
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


def refine(a):
    """Recompute escape flags ONLY for PRs that had a hit (stricter rules can only remove hits).
    Adds E_clean (revert or non-merge, non-docs fix commit), E_strict (revert or causal fix),
    and a `matches` table that lets `audit` run offline."""
    db = sqlite3.connect(a.db)
    cols = {r[1] for r in db.execute("pragma table_info(pr)")}
    for c in ("E_clean", "E_strict"):
        if c not in cols:
            db.execute(f"alter table pr add column {c}")
    db.execute("create table if not exists matches(pr_id, kind, sha, title, causal)")
    db.execute("update pr set E_clean=0, E_strict=0 where E_clean is null "
               "and coalesce(E,0)=0 and coalesce(E_fix,0)=0")
    todo = db.execute("select id, slug, number, merge_sha, merged_at from pr where E_clean is null "
                      "and merged_at is not null and merged_at != 'unavailable'").fetchall()
    print(f"refining {len(todo)} PRs with a hit", file=sys.stderr)
    s, cache = _session(), {}
    for pid, slug, n, sha, merged_at in todo:
        t0 = dt.datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
        escapes_in_window(s, slug, n, sha, merged_at, a.window, cache)
        day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        commits, _ = cache[(slug, day.date().isoformat())]
        m = []
        rev, fix = classify(commits, n, sha, t0, t0 + dt.timedelta(days=a.window), m)
        strict = int(rev or any(k == "fix" and ca for k, _, _, ca in m))
        db.execute("delete from matches where pr_id=?", (pid,))
        db.executemany("insert into matches values (?,?,?,?,?)", [(pid, k, c, t, ca) for k, c, t, ca in m])
        db.execute("update pr set E_clean=?, E_strict=? where id=?", (int(bool(rev or fix)), strict, pid))
        db.commit()


def audit(a):
    """Offline: sample fix-type matches (not reverts) for manual labelling. Run `refine` first."""
    db = sqlite3.connect(a.db)
    if not db.execute("select name from sqlite_master where name='matches'").fetchone():
        sys.exit("run `refine` first")
    rows = db.execute("select p.slug, p.number, m.sha, m.title, m.causal from matches m "
                      "join pr p on p.id = m.pr_id where m.kind='fix' and p.truncated=0 "
                      "order by random() limit ?", (a.n,)).fetchall()
    w = csv.writer(sys.stdout)
    w.writerow(["pr_url", "commit_url", "commit_title", "auto_causal(1/0)", "is_real_escape(y/n)"])
    for slug, n, sha, title, causal in rows:
        w.writerow([f"https://github.com/{slug}/pull/{n}", f"https://github.com/{slug}/commit/{sha}",
                    title, causal, ""])


def report(a):
    rows = sqlite3.connect(a.db).execute(
        "select agent, F, E, coalesce(E_fix,0), coalesce(truncated,0) from pr "
        "where merged_at is not null and merged_at != 'unavailable'").fetchall()
    pending = sqlite3.connect(a.db).execute(
        "select count(*) from pr where merged_at is null").fetchone()[0]
    print(f"not yet rechecked (excluded): {pending}")
    if not getattr(a, "include_truncated", False):
        n0 = len(rows)
        rows = [r for r in rows if not r[4]]
        print(f"excluded {n0 - len(rows)} PRs whose 14-day commit window exceeded 1,000 commits "
              f"(escapes cannot be detected reliably there)")
    if not rows:
        sys.exit("no rows")
    agents = np.array([r[0] for r in rows])
    F = np.array([r[1] for r in rows]); E = np.array([r[2] for r in rows])
    EB = np.array([bool(r[2] or r[3]) for r in rows]); T = np.array([r[4] for r in rows])
    print(f"rows analysed={len(rows)}")
    print("-- E = revert only")
    print(summarize(F, E))
    print("-- E_broad = revert OR fix/hotfix/regression commit referencing the PR")
    print(summarize(F, EB))
    for name, ev in (("E", E), ("E_broad", EB)):
        p1 = ev[F == 1].mean() if (F == 1).any() else float("nan")
        p0 = ev[F == 0].mean() if (F == 0).any() else float("nan")
        print(f"   {name}: P(E|F=1)={p1:.4f}  P(E|F=0)={p0:.4f}  RR=P(E|F=0)/P(E|F=1)="
              f"{(p0 / p1) if p1 else float('nan'):.2f}")
    cols = {r[1] for r in sqlite3.connect(a.db).execute("pragma table_info(pr)")}
    if {"E_clean", "E_strict"} <= cols:
        rr = sqlite3.connect(a.db).execute(
            "select F, coalesce(E_clean,0), coalesce(E_strict,0), coalesce(truncated,0), agent from pr "
            "where merged_at is not null and merged_at != 'unavailable' and E_clean is not null").fetchall()
        if not getattr(a, "include_truncated", False):
            rr = [r for r in rr if not r[3]]
        F2 = np.array([r[0] for r in rr])
        for name, idx in (("E_clean (revert or real fix commit)", 1), ("E_strict (revert or causal fix)", 2)):
            ev = np.array([r[idx] for r in rr])
            print(f"-- {name}"); print(summarize(F2, ev))
            p1, p0 = ev[F2 == 1].mean(), ev[F2 == 0].mean()
            print(f"   P(E|F=1)={p1:.4f}  P(E|F=0)={p0:.4f}  RR={(p0 / p1) if p1 else float('nan'):.2f}")
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
    own = {"sha": "m1", "commit": {"message": "Merge pull request #42 from a/fix/thing\n\nAdd x",
           "committer": {"date": (t0 + dt.timedelta(days=0.5)).isoformat()}}}
    assert classify([own], 42, "abcdef123", t0, t1) == (0, 0)                      # own merge commit
    assert classify([mk("docs: update CHANGELOG for fix of #42", 2)], 42, "abcdef123", t0, t1) == (0, 0)
    ui_rev = {"sha": "m2", "commit": {"message": "Merge pull request #50 from a/revert-42-thing\n\nRevert \"Add x (#42)\"",
              "committer": {"date": (t0 + dt.timedelta(days=1)).isoformat()}}}
    assert classify([ui_rev], 42, "abcdef123", t0, t1) == (1, 0)                   # revert via merge commit
    mm = []; classify([mk("fix: crash introduced in #42 (#60)", 2)], 42, "abcdef123", t0, t1, mm)
    assert mm[0][3] == 1 and causal_ref("fix: crash introduced in #42", 42)
    assert not causal_ref("chore: see #42 for context", 42)
    print(summarize(F, E, "independent"))
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    c = sub.add_parser("collect"); c.add_argument("--out", default="prs.sqlite")
    c.add_argument("--sample", type=int, default=2000); c.add_argument("--window", type=int, default=14)
    r = sub.add_parser("report"); r.add_argument("--db", default="prs.sqlite")
    au = sub.add_parser("audit"); au.add_argument("--db", default="prs.sqlite")
    au.add_argument("--n", type=int, default=50); au.add_argument("--window", type=int, default=14)
    rf = sub.add_parser("refine"); rf.add_argument("--db", default="prs.sqlite")
    rf.add_argument("--window", type=int, default=14)
    r.add_argument("--include-truncated", action="store_true")
    k = sub.add_parser("recheck"); k.add_argument("--db", default="prs.sqlite")
    k.add_argument("--window", type=int, default=14)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or a.cmd == "selftest": selftest()
    elif a.cmd == "collect": collect(a)
    elif a.cmd == "refine": refine(a)
    elif a.cmd == "audit": audit(a)
    elif a.cmd == "recheck": recheck(a)
    elif a.cmd == "report": report(a)
    else: p.print_help()
