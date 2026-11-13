"""Statistical analysis + figures.  python analyze.py --results results --out analysis"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textbench as tb

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
COL = {"KNN": "#1f77b4", "RF": "#d62728", "C4.5": "#2ca02c"}
MK = {"KNN": "o", "RF": "s", "C4.5": "^"}
LABEL = {"movie_reviews": "Movie Reviews", "sms_spam": "SMS Spam", "reuters8": "Reuters-8", "subjectivity": "Subjectivity"}


def cd_diagram(avg_ranks, cd, path, title):
    """Demsar-style critical-difference diagram (rank 1 at the right)."""
    names = list(avg_ranks.index); r = avg_ranks.values.astype(float); k = len(r)
    X = lambda rank: (k + 1) - rank                      # mirror so that rank 1 is on the right
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    ax.set_xlim(-0.6, k + 1.6); ax.set_ylim(-0.1, 1.0); ax.axis("off")
    y0 = 0.62
    ax.hlines(y0, X(k), X(1), color="k", lw=1)
    for t in range(1, k + 1):
        ax.vlines(X(t), y0 - 0.025, y0 + 0.025, color="k", lw=1)
        ax.text(X(t), y0 + 0.07, str(t), ha="center", fontsize=7)
    order = np.argsort(r)
    right = [i for i in order if r[i] <= (k + 1) / 2]
    left = [i for i in order if r[i] > (k + 1) / 2][::-1]
    for side, idxs in (("r", right), ("l", left)):
        for j, i in enumerate(idxs):
            y = y0 - 0.17 - 0.13 * j
            xe = k + 1.0 if side == "r" else 0.0
            ax.vlines(X(r[i]), y, y0, color="k", lw=0.7)
            ax.hlines(y, X(r[i]), xe, color="k", lw=0.7)
            ax.text(xe + (0.08 if side == "r" else -0.08), y, f"{names[i]} ({r[i]:.2f})",
                    ha="left" if side == "r" else "right", va="center", fontsize=7)
    ax.hlines(0.92, X(1), X(1) - cd, color="k", lw=1.6)
    ax.vlines([X(1), X(1) - cd], 0.9, 0.94, color="k", lw=1)
    ax.text(X(1) - cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    # thick bars join methods that are not significantly different (Nemenyi)
    srt = list(order); drawn = []
    for a_ in range(k):
        grp = [srt[a_]]
        for b_ in range(a_ + 1, k):
            if r[srt[b_]] - r[srt[a_]] < cd:
                grp.append(srt[b_])
        if len(grp) > 1 and not any(set(grp) <= set(g) for g in drawn):
            drawn.append(grp)
    for gi, g in enumerate(drawn):
        ax.hlines(y0 - 0.06 - 0.03 * gi, X(r[g[0]]), X(r[g[-1]]), color="k", lw=2.8)
    ax.set_title(title, fontsize=8, y=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.results, "fold_results.csv"))
    stab = pd.read_csv(os.path.join(a.results, "stability.csv"))
    models = list(tb.MODELS)
    out = {}

    # ---- descriptive table: mean (sd) per dataset x n x model -----------------
    metrics = ["accuracy", "macro_f1", "fit_s", "predict_ms_per_doc", "model_kb", "structure_size", "explanation_len"]
    # average over the 5 folds of a repeat first (fold-level SD is also kept)
    desc = df.groupby(["dataset", "n", "model"])[metrics].agg(["mean", "std"])
    desc.columns = ["_".join(c) for c in desc.columns]; desc = desc.reset_index()
    desc.to_csv(os.path.join(a.out, "descriptive.csv"), index=False)

    # ---- configuration-level means (rows = dataset x n) -----------------------
    cfg = {m: df.groupby(["dataset", "n", "model"])[m].mean().unstack("model")[models] for m in metrics}
    out["n_configs"] = int(cfg["macro_f1"].shape[0])
    out["folds_per_config"] = int(df.groupby(["dataset", "n", "model"]).size().iloc[0])
    out["overall_mean"] = {m: cfg[m].mean().round(5).to_dict() for m in metrics}
    out["overall_sd_over_configs"] = {m: cfg[m].std().round(5).to_dict() for m in metrics}

    # ---- global tests -----------------------------------------------------------
    tests = {}
    for metric, hib in [("macro_f1", True), ("accuracy", True), ("fit_s", False),
                        ("predict_ms_per_doc", False), ("model_kb", False)]:
        fr = tb.friedman_nemenyi(cfg[metric], higher_is_better=hib)
        wh = tb.wilcoxon_holm(cfg[metric])
        diffs = {}
        for x, y in [("RF", "KNN"), ("RF", "C4.5"), ("KNN", "C4.5")]:
            d = (cfg[metric][x] - cfg[metric][y]).values
            if metric in ("fit_s", "predict_ms_per_doc", "model_kb"):
                d = np.log10(cfg[metric][x].values) - np.log10(cfg[metric][y].values)
            lo, hi = tb.bootstrap_ci(d)
            diffs[f"{x}-{y}"] = dict(mean=float(d.mean()), lo=lo, hi=hi)
        tests[metric] = dict(chi2=float(fr["chi2"]), p=float(fr["p"]), F=float(fr["F"]), p_F=float(fr["p_F"]),
                             cd=float(fr["cd"]), N=int(fr["N"]), k=int(fr["k"]),
                             ranks=fr["avg_ranks"].round(3).to_dict(),
                             wilcoxon=wh.round(5).to_dict(orient="records"), boot=diffs)
        if metric == "macro_f1":
            cd_diagram(fr["avg_ranks"], fr["cd"], os.path.join(a.out, "fig_cd_macro_f1.png"),
                       f"Macro-F1 (Friedman, N = {fr['N']})")
    out["tests"] = tests

    # ---- per-config corrected resampled t-tests --------------------------------
    pc = {}
    for metric in ("macro_f1", "fit_s"):
        tt = tb.pairwise_config_tests(df, metric)
        tt.to_csv(os.path.join(a.out, f"pairwise_{metric}.csv"), index=False)
        summ = []
        for (x, y), g in tt.groupby(["a", "b"]):
            sig = g[g["p_holm"] < 0.05]
            summ.append(dict(a=x, b=y, n_cfg=len(g), a_sig_better=int((sig["mean_diff"] > 0).sum()),
                             b_sig_better=int((sig["mean_diff"] < 0).sum()),
                             n_ns=int(len(g) - len(sig)), median_dz=float(g["cohen_dz"].median()),
                             median_cliffs=float(g["cliffs"].median())))
        pc[metric] = summ
    out["per_config_tests"] = pc

    # ---- stability + complexity -------------------------------------------------
    st = stab.groupby("model")["top10_jaccard"].agg(["mean", "std"]).round(4)
    out["stability"] = st.to_dict(orient="index")
    out["stability_by_dataset"] = stab.groupby(["dataset", "model"])["top10_jaccard"].mean().round(4).unstack().to_dict()
    big = df[df["n"] == df["n"].max()].groupby("model")[["structure_size", "explanation_len", "model_kb"]].mean().round(2)
    out["complexity_at_max_n"] = big.to_dict(orient="index")

    # ---- Pareto analysis (accuracy vs. training time vs. model size) -------------
    cnt = {m: 0 for m in models}
    for (ds, n), g in df.groupby(["dataset", "n"]):
        mm = g.groupby("model")[["macro_f1", "fit_s", "model_kb"]].mean()
        for m in models:
            dom = False
            for o in models:
                if o == m: continue
                if (mm.loc[o, "macro_f1"] >= mm.loc[m, "macro_f1"] and mm.loc[o, "fit_s"] <= mm.loc[m, "fit_s"] and
                        mm.loc[o, "model_kb"] <= mm.loc[m, "model_kb"] and
                        (mm.loc[o] != mm.loc[m]).any()):
                    dom = True
            cnt[m] += int(not dom)
    out["pareto_nondominated_configs"] = cnt

    # ---- best model per config ---------------------------------------------------
    best = cfg["macro_f1"].idxmax(axis=1).value_counts().to_dict()
    out["best_macro_f1_counts"] = best
    out["config_means_macro_f1"] = cfg["macro_f1"].round(4).reset_index().to_dict(orient="records")
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)

    # ---- figures --------------------------------------------------------------------
    dss = sorted(df["dataset"].unique(), key=lambda d: list(LABEL).index(d))
    fig, axs = plt.subplots(1, len(dss), figsize=(6.6, 1.9), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, ds in zip(axs, dss):
        for m in models:
            g = df[(df.dataset == ds) & (df.model == m)].groupby("n")["macro_f1"]
            mu, sd, cnt_ = g.mean(), g.std(), g.count()
            ci = 1.96 * sd / np.sqrt(cnt_)
            ax.plot(mu.index, mu.values, color=COL[m], marker=MK[m], ms=3, lw=1, label=m)
            ax.fill_between(mu.index, mu - ci, mu + ci, color=COL[m], alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_title(LABEL[ds]); ax.set_xlabel("Sample size n")
        ax.set_xticks(sorted(df.n.unique())); ax.set_xticklabels([str(x) for x in sorted(df.n.unique())], rotation=45)
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Macro-F1")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(a.out, "fig_learning_curves.png"), dpi=300); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1))
    for m in models:
        g = df.groupby(["dataset", "n", "model"])[["macro_f1", "fit_s", "predict_ms_per_doc", "model_kb"]].mean().xs(m, level="model")
        axs[0].scatter(g["fit_s"], g["macro_f1"], s=10, color=COL[m], marker=MK[m], label=m, alpha=0.8)
        axs[1].scatter(g["model_kb"], g["macro_f1"], s=10, color=COL[m], marker=MK[m], alpha=0.8)
    axs[0].set_xscale("log"); axs[1].set_xscale("log")
    axs[0].set_xlabel("Training time (s, log)"); axs[1].set_xlabel("Serialized model size (KB, log)")
    axs[0].set_ylabel("Macro-F1"); axs[0].legend(frameon=False)
    for ax in axs: ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_tradeoff.png"), dpi=300); plt.close(fig)
    print(json.dumps({k: out[k] for k in ("n_configs", "folds_per_config", "overall_mean", "best_macro_f1_counts",
                                          "pareto_nondominated_configs")}, indent=1, default=float))


if __name__ == "__main__":
    main()


def pipeline_figure(path):
    """Fig. 1: evaluation pipeline (drawn with matplotlib)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
    def box(x, y, w, h, t, fc="#f2f2f2", fs=5.6):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.7))
    xs = [0, 20.5, 41, 61.5, 82]; w = 18; y = 17; h = 14
    labels = ["4 public corpora\n(English text)", "Stratified\nsub-sampling\nn = 100 ... 2000", "5-fold stratified CV\n10 repeats\n(50 folds / config)",
              "TF-IDF on train fold\n+ KNN | RF | C4.5", "Per-fold metrics"]
    for x, t in zip(xs, labels): box(x, y, w, h, t)
    for x in xs[:-1]: arr(x + w, y + h / 2, x + 20.5, y + h / 2)
    box(41, 0, w, 9, "Paired fold\ndifferences", "#ffffff"); arr(50, y, 50, 9)
    box(61.5, 0, 38.5, 9, "Friedman-Nemenyi | Wilcoxon-Holm\ncorrected resampled t-test | bootstrap CI", "#ffffff"); arr(59, 4.5, 61.5, 4.5)
    box(61.5, 35, 38.5, 8, "Accuracy | macro-F1 | time | size | complexity | stability", "#ffffff"); arr(91, y + h, 91, 35)
    arr(91, y, 91, 9)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import sys
    pipeline_figure(os.path.join("analysis", "fig_pipeline.png"))
