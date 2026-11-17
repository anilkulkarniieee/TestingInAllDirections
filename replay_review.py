#!/usr/bin/env python3
"""
Replay pilot, step 2: run a review agent over history, time-ordered, and score it.

For each PR (oldest first) the agent sees the PR plus a defect memory that contains ONLY
escapes already observable at merge time (escape PRs merged at least W days earlier).
It returns a risk score 0-100. We compare it with two baselines against E_strict / E:
  F      pre-merge CI failure (the status quo signal)
  size   lines changed
Metrics: AUROC, PR-AUC (average precision), recall of escapes in the top 10% riskiest PRs.

  export ANTHROPIC_API_KEY=sk-ant-...
  python replay_review.py --data meshmonitor.jsonl --label E_strict --max-prs 500
  python replay_review.py --data meshmonitor.jsonl --mock      # no API calls, random scores
  python replay_review.py --selftest
"""
import argparse, datetime as dt, json, math, os, random, re, sys
import requests

MODEL = "claude-sonnet-5-5"
SYSTEM = ("You are a code review agent. Estimate the probability (0-100) that this merged change "
          "will cause a production problem that needs a revert or a corrective fix within 14 days. "
          "Use the past escapes of this repository as evidence. Answer with JSON only: "
          '{"risk": <0-100>, "flags": ["short reason", ...]}')


def parse_t(x): return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))


def memory_for(rows, i, label, window=14, k=8):
    """Escapes observable at merge time of rows[i]: merged >= window days before it. Never the future."""
    t = parse_t(rows[i]["merged_at"]) - dt.timedelta(days=window)
    past = [r for r in rows[:i] if r[label] and parse_t(r["merged_at"]) <= t]
    return [f"PR #{r['number']}: {r['title']} -> later fix: "
            f"{(r['matches'][0]['title'] if r['matches'] else 'revert')}" for r in past[-k:]]


def build_prompt(r, memory):
    mem = "\n".join(memory) or "(none yet)"
    return (f"Past escapes in this repository:\n{mem}\n\nNew change #{r['number']}: {r['title']}\n"
            f"{r['body']}\nSize: +{r['additions']}/-{r['deletions']} in {r['changed_files']} files\n\n"
            f"Diff excerpt:\n{r['diff']}")


def ask(prompt):
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 300, "system": SYSTEM,
              "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"])
    m = re.search(r"\{.*\}", text, re.S)
    return float(json.loads(m.group(0))["risk"]) if m else float("nan")


# ---------- metrics ----------
def auroc(score, y):
    pos = [s for s, v in zip(score, y) if v]; neg = [s for s, v in zip(score, y) if not v]
    if not pos or not neg: return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def avg_precision(score, y):
    order = sorted(range(len(y)), key=lambda i: -score[i]); tp = 0; ap = 0.0
    for rank, i in enumerate(order, 1):
        if y[i]: tp += 1; ap += tp / rank
    return ap / max(sum(y), 1)


def recall_top(score, y, frac=0.10):
    k = max(1, int(len(y) * frac)); order = sorted(range(len(y)), key=lambda i: -score[i])[:k]
    return sum(y[i] for i in order) / max(sum(y), 1)


def boot_ci(fn, score, y, n=500, seed=5):
    rng = random.Random(seed); vals = []
    for _ in range(n):
        idx = [rng.randrange(len(y)) for _ in y]; v = fn([score[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(v): vals.append(v)
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]) if vals else (float("nan"),) * 2


def report(rows, scores, label, name="review agent"):
    y = [int(r[label]) for r in rows]
    sets = {name: scores, "CI failure (F)": [float(r["F"]) for r in rows],
            "size (lines)": [math.log1p(r["additions"] + r["deletions"]) for r in rows]}
    print(f"PRs={len(y)}  escapes ({label})={sum(y)}  base rate={sum(y)/len(y):.3%}")
    for name, sc in sets.items():
        lo, hi = boot_ci(auroc, sc, y)
        print(f"{name:16} AUROC={auroc(sc, y):.3f} [{lo:.3f},{hi:.3f}]  "
              f"PR-AUC={avg_precision(sc, y):.3f}  recall@top10%={recall_top(sc, y):.2f}")


def main(a):
    rows = sorted((json.loads(l) for l in open(a.data)), key=lambda r: r["merged_at"])
    rows = [r for r in rows if not r["truncated"]]
    if a.max_prs: rows = rows[-a.max_prs:]
    scores = []
    for i, r in enumerate(rows):
        if a.mock: scores.append(random.random() * 100); continue
        try: scores.append(ask(build_prompt(r, memory_for(rows, i, a.label))))
        except Exception as e: print(f"PR {r['number']}: {e}", file=sys.stderr); scores.append(float("nan"))
        if i % 25 == 0: print(f"  {i}/{len(rows)}", file=sys.stderr)
    keep = [i for i, s in enumerate(scores) if not math.isnan(s)]
    json.dump({"scores": {rows[i]["number"]: scores[i] for i in keep}}, open(a.data + ".scores.json", "w"))
    report([rows[i] for i in keep], [scores[i] for i in keep], a.label)


def selftest():
    y = [0, 0, 1, 1]; assert auroc([0.1, 0.2, 0.8, 0.9], y) == 1.0 and auroc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert abs(avg_precision([0.1, 0.2, 0.8, 0.9], y) - 1.0) < 1e-9
    assert recall_top([0.1, 0.2, 0.8, 0.9], y, 0.5) == 1.0
    base = dict(F=0, additions=1, deletions=0, changed_files=1, diff="", body="", E=0, E_strict=0, matches=[])
    rows = [dict(base, number=i, title=f"t{i}", merged_at=(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(days=i)).isoformat(), **{}) for i in range(40)]
    rows[5]["E_strict"] = 1; rows[5]["matches"] = [{"title": "fix: x", "causal": 1}]
    assert memory_for(rows, 10, "E_strict") == []              # escape from 5 days earlier is not yet knowable
    assert len(memory_for(rows, 25, "E_strict")) == 1          # observable 20 days later
    assert memory_for(rows, 3, "E_strict") == []               # never sees the future
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--label", default="E_strict", choices=["E", "E_strict", "E_clean"])
    p.add_argument("--max-prs", type=int, default=0); p.add_argument("--mock", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else main(a)
