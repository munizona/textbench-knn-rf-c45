# textbench: KNN vs Random Forest vs C4.5 in small-scale text mining

Companion code and results for the paper *Accuracy, Efficiency, and Interpretability Trade-offs of KNN,
Random Forest, and C4.5 in Small-Scale Text Mining: A Statistical Significance Study*
(A. Muni, S. Defit, I. Efendi; corresponding author: abdulmuni@live.com).

The study compares KNN, Random Forest and a C4.5-style gain-ratio tree on four public English corpora
(Movie Reviews, SMS Spam, Reuters-8, Subjectivity) at five sample sizes (100 to 2,000 documents), using
10 repeats of stratified 5-fold cross-validation (20 configurations, 3,000 model evaluations).

## Requirements
Python 3.12.3 was used for the paper. Install the packages with

    pip install -r requirements.txt

or use `requirements-paper.txt` for the exact versions reported in the paper
(NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0).

## How to run
    python run_experiments.py --out results --repeats 10      # full study (about 20 min on 1 CPU core)
    python analyze.py --results results --out analysis        # tests, tables (JSON/CSV) and figures
    streamlit run streamlit_app.py                             # interactive GUI (any dataset or your own CSV)
    python verify_refs.py references.json                      # check DOIs against Crossref (needs internet)

Datasets are downloaded on first use from public GitHub mirrors into `data_cache/`
(see `DATASET_INFO` in `textbench.py` for the original sources).

To reproduce the tables and figures without re-running the experiments, run only the second command:
`analyze.py` reads the supplied `results/fold_results.csv` and regenerates the files in `analysis/`.
Timing columns (`fit_s`, `predict_ms_per_doc`) depend on the hardware; the paper used one virtualised
Intel Xeon core at 2.8 GHz with 3 GB of memory.

## Files
| Path | Content |
|---|---|
| `textbench.py` | data loaders, C4.5 classifier (`C45Classifier`: gain ratio, pessimistic pruning), evaluation loop, statistical tests |
| `run_experiments.py` | runs the benchmark |
| `analyze.py` | statistical analysis and figures |
| `streamlit_app.py` | graphical application |
| `verify_refs.py`, `references.json` | DOI check for the reference list |
| `results/fold_results.csv` | fold-level results (3,000 rows) |
| `results/stability.csv` | top-10 term stability (Jaccard) per configuration |
| `results/config.json` | settings used for the run |
| `analysis/` | summary.json, descriptive and pairwise CSVs, figures |

## Data
The corpora are public and are not redistributed here:
Movie Reviews (Pang and Lee), SMS Spam Collection (Almeida et al.), Reuters-21578 and Subjectivity v1.0
(Pang and Lee). Source links are in the paper (Data and Code Availability section).

## Note on the C4.5 implementation
`C45Classifier` is a C4.5-style tree written for sparse text (binary term presence, top-1,000 terms by
chi-square). It is not a port of Quinlan's original program or of Weka's J48, so the C4.5 results apply
to this implementation.

## License and citation
Code is released under the MIT License (see `LICENSE`). Please cite the paper; see `CITATION.cff`.
