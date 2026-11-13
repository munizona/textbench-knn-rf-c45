"""
textbench.py - Benchmark of KNN, Random Forest and C4.5 for small-scale text mining.

Core library used by run_experiments.py (CLI) and streamlit_app.py (GUI).
Authors: Abdul Muni, Serjon Defit, Irman Efendi (corresponding: abdulmuni@live.com)
"""
from __future__ import annotations

import io
import itertools
import os
import pickle
import time
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------------- #
# 1. Datasets (all public, fetched from GitHub mirrors of the original corpora)
# --------------------------------------------------------------------------- #
NLTK_RAW = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"
SMS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

DATASET_INFO = {
    "movie_reviews": dict(
        title="Movie Reviews (polarity v2.0)", task="sentiment (2 classes)",
        origin="Pang & Lee (2004), cs.cornell.edu/people/pabo/movie-review-data",
        url=NLTK_RAW.format("movie_reviews")),
    "sms_spam": dict(
        title="SMS Spam Collection", task="spam detection (2 classes)",
        origin="Almeida et al. (2011), UCI ML Repository, archive.ics.uci.edu/dataset/228",
        url=SMS_URL),
    "reuters8": dict(
        title="Reuters-21578 (single-label, top-8 categories)", task="topic (8 classes)",
        origin="Reuters-21578 Distribution 1.0 via NLTK, kdd.ics.uci.edu/databases/reuters21578",
        url=NLTK_RAW.format("reuters")),
    "subjectivity": dict(
        title="Subjectivity v1.0", task="subjective vs objective sentences (2 classes)",
        origin="Pang & Lee (2004), cs.cornell.edu/people/pabo/movie-review-data",
        url=NLTK_RAW.format("subjectivity")),
}


def _fetch(url: str, cache_dir: str) -> bytes:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, "rb") as f:
        return f.read()


def load_dataset(name: str, cache_dir: str = "data_cache"):
    """Return (texts: list[str], labels: np.ndarray[str])."""
    if name == "movie_reviews":
        z = zipfile.ZipFile(io.BytesIO(_fetch(DATASET_INFO[name]["url"], cache_dir)))
        texts, labels = [], []
        for n in sorted(z.namelist()):
            if n.endswith(".txt") and ("/pos/" in n or "/neg/" in n):
                texts.append(z.read(n).decode("latin-1"))
                labels.append("pos" if "/pos/" in n else "neg")
    elif name == "sms_spam":
        raw = _fetch(SMS_URL, cache_dir).decode("utf-8", errors="replace")
        texts, labels = [], []
        for line in raw.splitlines():
            if "\t" in line:
                lab, txt = line.split("\t", 1)
                labels.append(lab)
                texts.append(txt)
    elif name == "reuters8":
        z = zipfile.ZipFile(io.BytesIO(_fetch(DATASET_INFO[name]["url"], cache_dir)))
        cats = {}
        for line in z.read("reuters/cats.txt").decode().splitlines():
            p = line.split()
            if len(p) >= 2:
                cats[p[0]] = p[1:]
        single = {k: v[0] for k, v in cats.items() if len(v) == 1}
        top = [c for c, _ in pd.Series(list(single.values())).value_counts().head(8).items()]
        texts, labels = [], []
        for k in sorted(single):
            if single[k] in top:
                texts.append(z.read("reuters/" + k).decode("latin-1"))
                labels.append(single[k])
    elif name == "subjectivity":
        z = zipfile.ZipFile(io.BytesIO(_fetch(DATASET_INFO[name]["url"], cache_dir)))
        texts, labels = [], []
        for fn, lab in (("subjectivity/quote.tok.gt9.5000", "subjective"),
                        ("subjectivity/plot.tok.gt9.5000", "objective")):
            for line in z.read(fn).decode("latin-1").splitlines():
                if line.strip():
                    texts.append(line.strip())
                    labels.append(lab)
    else:
        raise ValueError(f"unknown dataset {name}")
    return texts, np.asarray(labels)


def stratified_subsample(y, n, seed, min_per_class=5):
    """Class-proportional subsample of size n with at least min_per_class per class."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    if n >= len(y):
        return np.arange(len(y))
    alloc = np.maximum(min_per_class, np.round(n * counts / counts.sum()).astype(int))
    while alloc.sum() > n:                       # trim the largest class
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n:                       # pad the largest class
        alloc[np.argmax(counts - alloc)] += 1
    idx = []
    for c, a in zip(classes, alloc):
        members = np.where(y == c)[0]
        idx.extend(rng.choice(members, size=min(a, len(members)), replace=False))
    return np.sort(np.asarray(idx))


# --------------------------------------------------------------------------- #
# 2. C4.5 (gain ratio, average-gain filter, pessimistic error pruning)
# --------------------------------------------------------------------------- #
def _entropy_rows(counts):
    """Row-wise Shannon entropy (bits) of a (m x C) count matrix."""
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, counts / tot, 0.0)
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return h


def _ucb_errors(n, e, cf):
    """Upper confidence bound on the number of errors (Clopper-Pearson, as in C4.5/J48)."""
    if n <= 0:
        return 0.0
    if e >= n:
        return float(n)
    return float(n * beta.ppf(1.0 - cf, e + 1, n - e))


class C45Tree(BaseEstimator, ClassifierMixin):
    """C4.5-style tree on binary term-presence features.

    * split criterion: gain ratio, restricted to attributes whose information gain is at
      least the average gain of all admissible attributes (Quinlan's heuristic);
    * binary splits on "term present / absent"; min_samples_leaf mirrors J48's minNumObj;
    * pessimistic error-based pruning with confidence factor cf (default 0.25).
    """

    def __init__(self, min_samples_leaf=2, cf=0.25, max_depth=40, prune=True):
        self.min_samples_leaf = min_samples_leaf
        self.cf = cf
        self.max_depth = max_depth
        self.prune = prune

    # -- training ----------------------------------------------------------
    def fit(self, X, y):
        B = (X > 0)
        B = np.asarray(B.todense() if hasattr(B, "todense") else B, dtype=np.float32)
        self.classes_, yi = np.unique(y, return_inverse=True)
        C = len(self.classes_)
        Y = np.eye(C, dtype=np.float32)[yi]
        self.feature_, self.left_, self.right_, self.counts_, self.depth_arr_ = [], [], [], [], []
        self.gain_n_ = np.zeros(B.shape[1])          # n * gain per feature (importance)
        self._build(B, Y, np.arange(B.shape[0]), 0)
        self.feature_ = np.asarray(self.feature_)
        self.left_ = np.asarray(self.left_)
        self.right_ = np.asarray(self.right_)
        self.counts_ = np.asarray(self.counts_, dtype=np.float64)
        if self.prune:
            self._prune(0)
            self._compact()
        self.n_nodes_ = len(self.feature_)
        self.n_leaves_ = int((self.feature_ < 0).sum())
        self.max_depth_reached_ = int(self._depths().max())
        return self

    def _new_node(self, counts, depth):
        self.feature_.append(-1)
        self.left_.append(-1)
        self.right_.append(-1)
        self.counts_.append(counts)
        self.depth_arr_.append(depth)
        return len(self.feature_) - 1

    def _build(self, B, Y, idx, depth):
        counts = Y[idx].sum(axis=0)
        node = self._new_node(counts, depth)
        n = len(idx)
        if (counts > 0).sum() <= 1 or n < 2 * self.min_samples_leaf or depth >= self.max_depth:
            return node
        Bs = B[idx]
        present = Bs.T @ Y[idx]                      # F x C
        absent = counts[None, :] - present
        n_p, n_a = present.sum(1), absent.sum(1)
        ok = (n_p >= self.min_samples_leaf) & (n_a >= self.min_samples_leaf)
        if not ok.any():
            return node
        h0 = _entropy_rows(counts[None, :])[0]
        gain = h0 - (n_p / n) * _entropy_rows(present) - (n_a / n) * _entropy_rows(absent)
        fp, fa = n_p / n, n_a / n
        with np.errstate(divide="ignore", invalid="ignore"):
            split_info = -(np.where(fp > 0, fp * np.log2(fp), 0) + np.where(fa > 0, fa * np.log2(fa), 0))
            ratio = np.where(split_info > 1e-12, gain / split_info, 0.0)
        gain = np.where(ok, gain, -np.inf)
        cand = np.isfinite(gain) & (gain > 1e-12)
        if not cand.any():
            return node
        avg_gain = gain[cand].mean()
        cand &= gain >= avg_gain - 1e-12
        f = int(np.argmax(np.where(cand, ratio, -np.inf)))
        self.gain_n_[f] += n * gain[f]
        mask = Bs[:, f] > 0
        self.feature_[node] = f
        self.left_[node] = self._build(B, Y, idx[~mask], depth + 1)   # term absent
        self.right_[node] = self._build(B, Y, idx[mask], depth + 1)   # term present
        return node

    # -- pruning -----------------------------------------------------------
    def _prune(self, node):
        c = self.counts_[node]
        n, e = c.sum(), c.sum() - c.max()
        leaf_est = _ucb_errors(n, e, self.cf)
        if self.feature_[node] < 0:
            return leaf_est
        sub_est = self._prune(self.left_[node]) + self._prune(self.right_[node])
        if leaf_est <= sub_est + 0.1:
            self.feature_[node], self.left_[node], self.right_[node] = -1, -1, -1
            return leaf_est
        return sub_est

    def _compact(self):
        keep, stack = [], [0]
        while stack:
            u = stack.pop()
            keep.append(u)
            if self.feature_[u] >= 0:
                stack.extend([self.right_[u], self.left_[u]])
        keep = sorted(keep)
        remap = {old: new for new, old in enumerate(keep)}
        self.feature_ = self.feature_[keep]
        self.counts_ = self.counts_[keep]
        self.depth_arr_ = [self.depth_arr_[k] for k in keep]
        self.left_ = np.array([remap.get(self.left_[k], -1) if self.left_[k] >= 0 else -1 for k in keep])
        self.right_ = np.array([remap.get(self.right_[k], -1) if self.right_[k] >= 0 else -1 for k in keep])

    def _depths(self):
        return np.asarray(self.depth_arr_)

    # -- inference ---------------------------------------------------------
    def _leaf_ids(self, X):
        B = (X > 0)
        B = np.asarray(B.todense() if hasattr(B, "todense") else B)
        node = np.zeros(B.shape[0], dtype=int)
        steps = np.zeros(B.shape[0], dtype=int)
        while True:
            f = self.feature_[node]
            internal = f >= 0
            if not internal.any():
                break
            rows = np.where(internal)[0]
            go_right = B[rows, f[rows]] > 0
            node[rows] = np.where(go_right, self.right_[node[rows]], self.left_[node[rows]])
            steps[rows] += 1
        return node, steps

    def predict(self, X):
        leaf, _ = self._leaf_ids(X)
        return self.classes_[np.argmax(self.counts_[leaf], axis=1)]

    def path_lengths(self, X):
        return self._leaf_ids(X)[1]

    @property
    def feature_importances_(self):
        s = self.gain_n_.sum()
        return self.gain_n_ / s if s > 0 else self.gain_n_


class C45Classifier(BaseEstimator, ClassifierMixin):
    """chi2 top-k term selection followed by C45Tree (selection time is counted as training)."""

    def __init__(self, k_features=1000, min_samples_leaf=2, cf=0.25):
        self.k_features = k_features
        self.min_samples_leaf = min_samples_leaf
        self.cf = cf

    def fit(self, X, y):
        k = min(self.k_features, X.shape[1])
        self.sel_ = SelectKBest(chi2, k=k).fit(X, y)
        self.tree_ = C45Tree(min_samples_leaf=self.min_samples_leaf, cf=self.cf).fit(self.sel_.transform(X), y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(self.sel_.transform(X))

    def importances_full(self, n_features):
        full = np.zeros(n_features)
        full[self.sel_.get_support(indices=True)] = self.tree_.feature_importances_
        return full


# --------------------------------------------------------------------------- #
# 3. Models and evaluation protocol
# --------------------------------------------------------------------------- #
MODELS = ("KNN", "RF", "C4.5")


def make_model(name, seed=0, k=5, n_trees=200):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    if name == "RF":
        return RandomForestClassifier(n_estimators=n_trees, max_features="sqrt", n_jobs=1, random_state=seed)
    if name == "C4.5":
        return C45Classifier()
    raise ValueError(name)


def make_vectorizer():
    return TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_features=5000, sublinear_tf=True)


def model_size_kb(model):
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0


def complexity_descriptors(name, model, Xte):
    """Interpretability-related descriptors: (structure size, mean explanation length)."""
    if name == "C4.5":
        t = model.tree_
        return t.n_leaves_, float(t.path_lengths(model.sel_.transform(Xte)).mean())
    if name == "RF":
        ind, _ = model.decision_path(Xte)
        total_tests = np.asarray(ind.sum(axis=1)).ravel() - model.n_estimators   # edges, all trees
        return int(sum(e.tree_.node_count for e in model.estimators_)), float(total_tests.mean())
    return 0, float(model.n_neighbors)       # KNN: k neighbour documents to inspect


def top_terms(name, model, vocab, k=10):
    if name == "C4.5":
        imp = model.importances_full(len(vocab))
    elif name == "RF":
        imp = model.feature_importances_
    else:
        return None
    order = np.argsort(-imp)[:k]
    return set(vocab[order[imp[order] > 0]])


def evaluate_config(texts, y, n, seeds, n_splits=5, models=MODELS, k=5, n_trees=200, progress=None):
    """Repeated stratified k-fold on class-proportional subsamples of size n.

    Returns (fold-level DataFrame, dict model -> list of top-term sets per fold)."""
    rows, tops = [], {m: [] for m in models}
    texts = np.asarray(texts, dtype=object)
    for r, seed in enumerate(seeds):
        sub = stratified_subsample(y, n, seed)
        Xs, ys = texts[sub], y[sub]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(skf.split(Xs, ys)):
            vec = make_vectorizer()
            Xtr = vec.fit_transform(Xs[tr])
            Xte = vec.transform(Xs[te])
            vocab = np.asarray(vec.get_feature_names_out())
            for m in models:
                mdl = make_model(m, seed=seed, k=k, n_trees=n_trees)
                t0 = time.perf_counter(); mdl.fit(Xtr, ys[tr]); t_fit = time.perf_counter() - t0
                t0 = time.perf_counter(); pred = mdl.predict(Xte); t_pred = time.perf_counter() - t0
                size, expl = complexity_descriptors(m, mdl, Xte if m != "C4.5" else Xte)
                rows.append(dict(model=m, n=n, repeat=r, fold=f, n_train=len(tr), n_test=len(te),
                                 accuracy=accuracy_score(ys[te], pred),
                                 macro_f1=f1_score(ys[te], pred, average="macro"),
                                 fit_s=t_fit, predict_ms_per_doc=1000 * t_pred / len(te),
                                 model_kb=model_size_kb(mdl), structure_size=size, explanation_len=expl))
                tops[m].append(top_terms(m, mdl, vocab))
            if progress:
                progress(r, f)
    return pd.DataFrame(rows), tops


def jaccard_stability(sets):
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]))


# --------------------------------------------------------------------------- #
# 4. Statistical significance
# --------------------------------------------------------------------------- #
def corrected_resampled_ttest(d, n_train, n_test):
    """Nadeau-Bengio corrected paired t-test for repeated k-fold differences d (length J)."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return (0.0 if d.mean() == 0 else np.inf), (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cohen_dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def pairwise_config_tests(df, metric, models=MODELS):
    """Per (dataset, n): corrected resampled t-test, Holm over the model pairs, effect sizes."""
    out = []
    for (ds, n), g in df.groupby(["dataset", "n"]):
        piv = g.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
        ntr, nte = g["n_train"].mean(), g["n_test"].mean()
        res = []
        for a, b in itertools.combinations(models, 2):
            d = (piv[a] - piv[b]).values
            t, p = corrected_resampled_ttest(d, ntr, nte)
            res.append(dict(dataset=ds, n=n, a=a, b=b, mean_diff=d.mean(), t=t, p=p,
                            cohen_dz=cohen_dz(d), cliffs=cliffs_delta(piv[a].values, piv[b].values)))
        adj = holm([r["p"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = pa
            out.append(r)
    return pd.DataFrame(out)


def friedman_nemenyi(mean_table, higher_is_better=True):
    """mean_table: DataFrame configs x models. Returns dict with Friedman, Iman-Davenport, ranks, CD."""
    N, k = mean_table.shape
    ranks = mean_table.rank(axis=1, ascending=not higher_is_better, method="average")
    chi2_f, p_f = stats.friedmanchisquare(*[mean_table[c].values for c in mean_table.columns])
    ff = (N - 1) * chi2_f / (N * (k - 1) - chi2_f) if N * (k - 1) != chi2_f else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]       # Nemenyi critical values, alpha = 0.05
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * N))
    return dict(chi2=chi2_f, p=p_f, F=ff, p_F=p_ff, avg_ranks=ranks.mean(), cd=cd, N=N, k=k)


def wilcoxon_holm(mean_table, models=MODELS):
    pairs = list(itertools.combinations(models, 2))
    res = []
    for a, b in pairs:
        d = mean_table[a] - mean_table[b]
        try:
            w, p = stats.wilcoxon(d, zero_method="wilcox", method="auto")
        except ValueError:
            w, p = np.nan, 1.0
        res.append(dict(a=a, b=b, W=w, p=p, median_diff=float(np.median(d)),
                        wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum())))
    adj = holm([r["p"] for r in res])
    for r, pa in zip(res, adj):
        r["p_holm"] = pa
    return pd.DataFrame(res)


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))
