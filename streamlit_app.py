"""Streamlit GUI:  streamlit run streamlit_app.py"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import textbench as tb

st.set_page_config(page_title="KNN vs RF vs C4.5 - small-scale text mining", layout="wide")
st.title("Accuracy, Efficiency and Interpretability: KNN vs Random Forest vs C4.5")
st.caption("Abdul Muni, Serjon Defit, Irman Efendi - companion application of the paper "
           "(corresponding author: abdulmuni@live.com)")

with st.sidebar:
    st.header("Experiment")
    source = st.radio("Data source", ["Built-in public dataset", "Upload CSV (text,label)"])
    if source == "Built-in public dataset":
        ds = st.selectbox("Dataset", list(tb.DATASET_INFO), format_func=lambda k: tb.DATASET_INFO[k]["title"])
        info = tb.DATASET_INFO[ds]
        st.markdown(f"**Task:** {info['task']}  \n**Origin:** {info['origin']}  \n**Download:** {info['url']}")
    else:
        up = st.file_uploader("CSV with columns 'text' and 'label'", type="csv")
    n = st.select_slider("Sample size n", options=[100, 250, 500, 1000, 2000], value=500)
    repeats = st.slider("Repeats (x 5-fold CV)", 1, 10, 3)
    k = st.slider("KNN: k", 1, 15, 5)
    trees = st.slider("RF: number of trees", 50, 500, 200, step=50)
    chosen = st.multiselect("Models", list(tb.MODELS), default=list(tb.MODELS))
    run = st.button("Run benchmark", type="primary")

@st.cache_data(show_spinner=False)
def _load(name):
    return tb.load_dataset(name, "data_cache")

if run:
    if source == "Built-in public dataset":
        texts, y = _load(ds)
    else:
        if up is None:
            st.error("Upload a CSV first."); st.stop()
        d = pd.read_csv(up); texts, y = d["text"].astype(str).tolist(), d["label"].astype(str).values
    bar = st.progress(0.0)
    total = repeats * 5
    done = {"i": 0}
    def cb(r, f):
        done["i"] += 1; bar.progress(done["i"] / total)
    df, tops = tb.evaluate_config(texts, y, n, seeds=list(range(repeats)), models=tuple(chosen), k=k,
                                  n_trees=trees, progress=cb)
    df["dataset"] = ds if source != "Upload CSV (text,label)" else "uploaded"
    summ = df.groupby("model")[["accuracy", "macro_f1", "fit_s", "predict_ms_per_doc", "model_kb",
                                "structure_size", "explanation_len"]].agg(["mean", "std"]).round(4)
    st.subheader("Mean and SD over folds")
    st.dataframe(summ)
    c1, c2 = st.columns(2)
    c1.subheader("Macro-F1"); c1.bar_chart(df.groupby("model")["macro_f1"].mean())
    c2.subheader("Training time (s)"); c2.bar_chart(df.groupby("model")["fit_s"].mean())
    if len(chosen) >= 2:
        st.subheader("Pairwise corrected resampled t-test (Nadeau-Bengio), Holm-adjusted")
        tt = tb.pairwise_config_tests(df, "macro_f1", models=tuple(chosen))
        st.dataframe(tt[["a", "b", "mean_diff", "t", "p", "p_holm", "cohen_dz", "cliffs"]].round(4))
        st.caption("Significance is evaluated on one configuration only; use run_experiments.py + analyze.py "
                   "for the multi-dataset Friedman/Nemenyi analysis reported in the paper.")
    for m in ("RF", "C4.5"):
        if m in chosen:
            st.write(f"Top-10 term stability (mean pairwise Jaccard, {m}): {tb.jaccard_stability(tops[m]):.3f}")
    st.download_button("Download fold-level results (CSV)", df.to_csv(index=False), "fold_results.csv")
else:
    st.info("Choose a dataset and press **Run benchmark**. Datasets are downloaded from public GitHub mirrors on first use.")
