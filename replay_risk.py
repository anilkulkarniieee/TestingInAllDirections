#!/usr/bin/env python3
"""
Replay pilot WITHOUT an LLM: the paper's risk model (Eq. 4), walk-forward.

R(c) = sigmoid(w . x(c)) with features that the review layer could see at merge time:
size, files touched, config/test/doc-only changes, title type, author and file-path history
of KNOWN escapes, and the recent escape rate. Weights are learned online. Labels are
revealed only W days after a PR merged, so no score ever uses the future.

  python replay_risk.py --data data/Yeraze_meshmonitor.jsonl --label E_strict
  python replay_risk.py --selftest
Output: AUROC / PR-AUC / recall@top10% for the risk model vs two baselines
(pre-merge CI failure; PR size) on PRs after the warm-up period.
"""
import argparse, datetime as dt, json, math, re
import numpy as np
from replay_review import report

CONFIG = re.compile(r"(\.ya?ml$|\.json$|\.toml$|\.ini$|\.env|dockerfile|\.lock$|requirements|config|\.gradle|pom\.xml)", re.I)
KW = ["fix", "feat", "refactor", "bump|deps|dependenc", "revert"]
NF = 13


def parse_t(x): return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))


def files_of(r): return set(re.findall(r"^--- (\S+) \(\+", r.get("diff", ""), re.M))


def sigmoid(z): return 1 / (1 + math.exp(-max(min(z, 30), -30)))


class State:
    """Knowledge available at a point in time (only revealed labels)."""
    def __init__(self): self.author = {}; self.esc_files = []; self.recent = []

    def reveal(self, r, y):
        a = self.author.setdefault(r["author"], [0, 0]); a[0] += y; a[1] += 1
        if y: self.esc_files.append(files_of(r))
        self.recent = (self.recent + [y])[-50:]

    def features(self, r):
        fs = files_of(r); t = r["title"].lower(); a = self.author.get(r["author"], [0, 0])
        overlap = max((len(fs & e) / len(fs | e) for e in self.esc_files if fs | e), default=0.0)
        return np.array([
            math.log1p(r["additions"] + r["deletions"]) / 10, math.log1p(r["changed_files"]) / 5,
            float(any(CONFIG.search(f) for f in fs)), float(any("test" in f.lower() for f in fs)),
            float(bool(fs) and all(f.lower().endswith(".md") for f in fs)),
            *[float(bool(re.search(k, t))) for k in KW],
            (a[0] + 0.05) / (a[1] + 1), overlap, sum(self.recent[-50:]) / max(len(self.recent), 1)])


# Model v2 (frozen before the confirmatory repository was collected): start from the PR-size
# baseline and move away from it only when evidence supports it. Fewer, less noisy features
# (size, files, config, tests, docs-only, path overlap, recent rate), strong L2 pull toward a
# size-only prior, smaller steps, capped positive weight.
V2_MASK = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], float)
V2_PRIOR = np.array([3.0] + [0.0] * (NF - 1))


def walk_forward(rows, label, window=14, lr=0.2, l2=1e-3, model="v1"):
    rows = sorted(rows, key=lambda r: r["merged_at"])
    mask, w0, cap = (V2_MASK, V2_PRIOR, 10.0) if model == "v2" else (np.ones(NF), np.zeros(NF), 30.0)
    if model == "v2": lr, l2 = 0.05, 0.05
    w, b, st, nxt = w0.copy(), -3.0, State(), 0
    npos = nneg = 1; scores = []
    for r in rows:
        cut = parse_t(r["merged_at"]) - dt.timedelta(days=window)
        while nxt < len(rows) and parse_t(rows[nxt]["merged_at"]) <= cut:   # labels now observable
            q = rows[nxt]; y = int(q[label]); x = st.features(q) * mask
            wt = min(cap, nneg / npos) if y else 1.0
            g = sigmoid(w @ x + b) - y
            w -= lr * (wt * g * x + l2 * (w - w0)); b -= lr * wt * g
            npos += y; nneg += 1 - y
            st.reveal(q, y); nxt += 1
        scores.append(sigmoid(w @ (st.features(r) * mask) + b))                         # score BEFORE learning from r
    return rows, scores


def paired(rows, sc, label, n=500, seed=11):
    """Paired bootstrap of the AUROC difference: risk model minus each baseline (same PRs)."""
    import random
    from replay_review import auroc
    y = [int(r[label]) for r in rows]
    base = {"CI failure (F)": [float(r["F"]) for r in rows],
            "size (lines)": [math.log1p(r["additions"] + r["deletions"]) for r in rows]}
    rng = random.Random(seed)
    for name, b in base.items():
        diffs = []
        for _ in range(n):
            idx = [rng.randrange(len(y)) for _ in y]
            yy = [y[i] for i in idx]
            if not any(yy) or all(yy): continue
            diffs.append(auroc([sc[i] for i in idx], yy) - auroc([b[i] for i in idx], yy))
        diffs.sort()
        d0 = auroc(sc, y) - auroc(b, y)
        print(f"AUROC difference, model minus {name}: {d0:+.3f} "
              f"[{diffs[int(.025*len(diffs))]:+.3f},{diffs[int(.975*len(diffs))]:+.3f}]")


def main(a):
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if not r["truncated"]]
    rows, sc = walk_forward(rows, a.label, a.window, model=a.model)
    k = int(len(rows) * a.warmup)
    print(f"warm-up: first {k} PRs used for learning only")
    report(rows[k:], sc[k:], a.label, name=f"TAD risk model {a.model}")
    paired(rows[k:], sc[k:], a.label)


def selftest():
    import random
    random.seed(2); rows = []
    t0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(1500):
        cfg = random.random() < 0.3; y = int(random.random() < (0.15 if cfg else 0.01))
        rows.append(dict(number=i, merged_at=(t0 + dt.timedelta(hours=8 * i)).isoformat(), title="fix x" if cfg else "feat y",
                         body="", author=f"u{i % 7}", additions=random.randint(1, 300), deletions=5, changed_files=2,
                         diff="--- " + ("app/config.yml" if cfg else "src/a.py") + " (+1/-0)\n", F=int(random.random() < .3),
                         E_strict=y, matches=[], truncated=0))
    r2, sc = walk_forward(rows, "E_strict")
    from replay_review import auroc
    y = [r["E_strict"] for r in r2][300:]; assert auroc(sc[300:], y) > 0.7, auroc(sc[300:], y)
    # no leakage: flipping labels of the future must not change earlier scores
    flipped = [dict(r, E_strict=1 - r["E_strict"]) if r["number"] >= 900 else r for r in rows]
    _, sc2 = walk_forward(flipped, "E_strict")
    assert all(abs(a - b) < 1e-12 for a, b in zip(sc[:900], sc2[:900])), "leakage"
    r3, sc3 = walk_forward(rows, "E_strict", model="v2")
    assert auroc(sc3[300:], [r["E_strict"] for r in r3][300:]) > 0.7
    _, sc4 = walk_forward(flipped, "E_strict", model="v2")
    assert all(abs(a - b) < 1e-12 for a, b in zip(sc3[:900], sc4[:900])), "leakage v2"
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--label", default="E_strict", choices=["E", "E_strict", "E_clean"])
    p.add_argument("--window", type=int, default=14); p.add_argument("--warmup", type=float, default=0.3)
    p.add_argument("--model", default="v1", choices=["v1", "v2"])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(); selftest() if a.selftest else main(a)
