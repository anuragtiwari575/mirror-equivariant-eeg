"""
Reference baseline for cross-subject (LOSO) MI decoding: the honest shallow wall
the proposed model must beat.

Pipeline: Euclidean Alignment (EA) -> Riemannian tangent-space features -> LDA,
evaluated leave-one-subject-out. EA recenters each subject's mean per-trial
covariance to the identity using that subject's own (unlabeled) trials, so the
target subject needs no labels: the protocol is calibration-free / transductive.

Reads the UNNORMALIZED epoched cache from mi_dataset.build_subject_cached, so it
reuses whatever the data pipeline already cached for the same band-pass/window
(no z-score is applied; Riemannian methods operate on raw-signal covariances).

Logic verified in test_baselines.py on synthetic covariance-separable data.
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import mi_dataset

try:
    import config
    _RESULTS = config.RESULTS
    _DEFAULT_CFG = dict(config.PREPROC)
except Exception:
    config = None
    _RESULTS = "."
    _DEFAULT_CFG = {"tmin": 0.0, "tmax": 4.0, "l_freq": 4.0, "h_freq": 40.0,
                    "resample": None, "normalize": "trial"}


def _inv_sqrt(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** -0.5)) @ V.T


def euclidean_align(X):
    """Euclidean Alignment (He & Wu, 2020). X: (n, ch, t) -> aligned (n, ch, t).

    Recenters the mean per-trial covariance of this recording to the identity.
    Uses the trials themselves (no labels), so it is calibration-free.
    """
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[2]
    R_inv_sqrt = _inv_sqrt(covs.mean(axis=0))
    return np.einsum("cd,ndt->nct", R_inv_sqrt, X)


def riemann_ts_lda(estimator="oas"):
    """Covariances -> Riemannian tangent space -> LDA."""
    return make_pipeline(
        Covariances(estimator=estimator),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(),
    )


def evaluate_loso(dataset, clf_factory=riemann_ts_lda, align=True, cfg=None,
                  use_cache=True, save=True):
    """Leave-one-subject-out evaluation. Returns a results dict with per-subject
    accuracy and kappa plus mean/std."""
    cfg = cfg or _DEFAULT_CFG
    subjects = list(mi_dataset.DATASETS[dataset]["subjects"])

    data = {}
    for s in subjects:
        d = mi_dataset.build_subject_cached(dataset, s, cfg, use_cache)
        X, y = np.asarray(d["X_train"], dtype=np.float64), np.asarray(d["y_train"])
        if align:
            X = euclidean_align(X)
        data[s] = (X, y)

    per = []
    for test_s in subjects:
        src = [s for s in subjects if s != test_s]
        Xtr = np.concatenate([data[s][0] for s in src], axis=0)
        ytr = np.concatenate([data[s][1] for s in src], axis=0)
        Xte, yte = data[test_s]
        clf = clf_factory()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        per.append({"subject": test_s,
                    "acc": float(accuracy_score(yte, pred)),
                    "kappa": float(cohen_kappa_score(yte, pred))})

    accs = np.array([r["acc"] for r in per])
    kappas = np.array([r["kappa"] for r in per])
    result = {"dataset": dataset, "align": bool(align), "per_subject": per,
              "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
              "mean_kappa": float(kappas.mean()), "std_kappa": float(kappas.std())}
    if save:
        result["csv"] = _save_csv(result)
    return result


def _save_csv(result):
    os.makedirs(_RESULTS, exist_ok=True)
    tag = f"{result['dataset']}_riemannTS-LDA_EA{int(result['align'])}"
    path = os.path.join(_RESULTS, f"loso_{tag}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("subject,acc,kappa\n")
        for r in result["per_subject"]:
            f.write(f"{r['subject']},{r['acc']:.4f},{r['kappa']:.4f}\n")
        f.write(f"mean,{result['mean_acc']:.4f},{result['mean_kappa']:.4f}\n")
        f.write(f"std,{result['std_acc']:.4f},{result['std_kappa']:.4f}\n")
    return path


def print_result(result):
    print(f"\n=== LOSO {result['dataset']} | Riemann TS+LDA | "
          f"EA={'on' if result['align'] else 'off'} ===")
    print("  subject   acc(%)   kappa")
    for r in result["per_subject"]:
        print(f"  S{r['subject']:02d}      {r['acc']*100:6.2f}   {r['kappa']:6.3f}")
    print(f"  mean      {result['mean_acc']*100:6.2f}   {result['mean_kappa']:6.3f}")
    print(f"  std       {result['std_acc']*100:6.2f}   {result['std_kappa']:6.3f}")


# Riemannian methods do best on the mu/beta band with a post-cue window and no
# z-score (they operate on raw-signal covariances). This is the FAIR shallow
# wall; the deep model keeps the project's broadband config in config.PREPROC.
RIEMANN_CFG = {"tmin": 0.5, "tmax": 2.5, "l_freq": 8.0, "h_freq": 30.0,
               "resample": None, "normalize": None}


def main():
    for ds in ("2a", "2b"):
        for align in (False, True):
            res = evaluate_loso(ds, align=align, cfg=RIEMANN_CFG)
            print_result(res)
    print(f"\nPer-run CSVs saved under {_RESULTS}")
    print("(band 8-30 Hz, window 0.5-2.5 s -- the tuned/fair shallow wall)")


if __name__ == "__main__":
    main()
