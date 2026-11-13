"""CLI: run the full benchmark.  python run_experiments.py --out results --repeats 10"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
import textbench as tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--datasets", nargs="+", default=list(tb.DATASET_INFO))
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--trees", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames, stab = [], []
    t_all = time.time()
    for ds in a.datasets:
        texts, y = tb.load_dataset(ds, a.cache)
        for n in a.sizes:
            t0 = time.time()
            df, tops = tb.evaluate_config(texts, y, n, seeds=list(range(a.repeats)), n_splits=a.folds,
                                          k=a.knn_k, n_trees=a.trees)
            df.insert(0, "dataset", ds)
            frames.append(df)
            for m in ("RF", "C4.5"):
                stab.append(dict(dataset=ds, n=n, model=m, top10_jaccard=tb.jaccard_stability(tops[m])))
            pd.concat(frames).to_csv(os.path.join(a.out, "fold_results.csv"), index=False)
            pd.DataFrame(stab).to_csv(os.path.join(a.out, "stability.csv"), index=False)
            print(f"{ds:14s} n={n:5d} done in {time.time()-t0:6.1f}s (total {time.time()-t_all:7.1f}s)", flush=True)
    json.dump(dict(vars(a), numpy=np.__version__), open(os.path.join(a.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
