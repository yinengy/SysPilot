"""Forest-cover classification with Dask-ML grid search and XGBoost."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def prepare(directory):
    import numpy as np
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import StratifiedShuffleSplit

    directory.mkdir(parents=True, exist_ok=True)
    forest = fetch_covtype(data_home=str(directory / "cache"))
    split = StratifiedShuffleSplit(n_splits=1, train_size=120000, random_state=20260919)
    indices, _ = next(split.split(forest.data, forest.target))
    indices.sort()
    np.savez_compressed(
        directory / "covtype_120k.npz",
        X=forest.data[indices].astype("float32"),
        y=(forest.target[indices] - 1).astype("int32"),
        source_indices=indices,
    )
    digest = hashlib.sha256((directory / "covtype_120k.npz").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({"dataset_sha256": digest}) + "\n")


def run(args):
    import numpy as np
    from dask_ml.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from xgboost import XGBClassifier

    workers = int(os.environ.get("DASK_CV_WORKERS", "0"))
    threads = int(os.environ.get("DASK_XGB_THREADS", "0"))
    scheduler = os.environ.get("DASK_SCHEDULER", "threads")
    if not (0 <= workers <= 48 and 0 <= threads <= 48):
        raise ValueError("Worker and thread counts must be between 0 and 48")
    if scheduler not in ("threads", "processes"):
        raise ValueError("DASK_SCHEDULER must be threads or processes")

    source = args.data_dir / "covtype_120k.npz"
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch; run prepare again")
    with np.load(source) as data:
        x, y = data["X"], data["y"]
    if x.shape != (120000, 54) or y.shape != (120000,):
        raise ValueError("Expected the 120,000-row Covertype subset")
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=20260919, stratify=y
    )
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=20260919).split(x[train], y[train])
    )
    model = XGBClassifier(
        n_estimators=80,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        n_jobs=threads or None,
        random_state=20260919,
        objective="multi:softprob",
        num_class=7,
        subsample=1,
        colsample_bytree=1,
        max_bin=256,
    )
    search = GridSearchCV(
        model,
        {"max_depth": [4, 8], "min_child_weight": [1, 5, 10]},
        cv=folds,
        n_jobs=workers or -1,
        scoring="neg_log_loss",
        refit=True,
        error_score="raise",
        scheduler=scheduler,
        cache_cv=True,
    )
    start = time.perf_counter()
    search.fit(x[train], y[train])
    fit_time = time.perf_counter() - start
    probabilities = search.predict_proba(x[test])
    elapsed = time.perf_counter() - start
    predicted = np.argmax(probabilities, axis=1)
    scores = np.stack([search.cv_results_[f"split{i}_test_score"] for i in range(3)])
    if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite classifier output")
    if args.reference:
        with np.load(args.reference / "outputs.npz") as ref:
            np.testing.assert_array_equal(train, ref["train_indices"])
            np.testing.assert_array_equal(test, ref["test_indices"])
            np.testing.assert_array_equal(predicted, ref["predicted"])
            np.testing.assert_allclose(probabilities, ref["probabilities"], rtol=0, atol=1e-5)
            np.testing.assert_allclose(scores, ref["cv_scores"], rtol=0, atol=1e-7)
        reference = json.loads((args.reference / "result.json").read_text())
        if search.best_params_ != reference["quality"]["best_params"]:
            raise ValueError("Selected model changed")

    result = {
        "wall_time_sec": elapsed,
        "fit_time_sec": fit_time,
        "predict_time_sec": elapsed - fit_time,
        "config": {
            "DASK_CV_WORKERS": workers,
            "DASK_XGB_THREADS": threads,
            "DASK_SCHEDULER": scheduler,
        },
        "quality": {
            "accuracy": float(accuracy_score(y[test], predicted)),
            "log_loss": float(log_loss(y[test], probabilities)),
            "best_params": search.best_params_,
        },
    }
    np.savez_compressed(
        args.output_dir / "outputs.npz",
        probabilities=probabilities,
        predicted=predicted,
        cv_scores=scores,
        train_indices=train,
        test_indices=test,
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"wall_time_sec": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path, help="Directory containing a baseline run")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data_dir)
    else:
        if args.output_dir is None:
            parser.error("run requires --output-dir")
        if args.reference and args.output_dir.resolve() == args.reference.resolve():
            parser.error("Output and reference directories must differ")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Start each workload with the same native defaults as the experiment.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "OMP_WAIT_POLICY",
            "OMP_MAX_ACTIVE_LEVELS",
            "GOTO_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ.pop(name, None)
        os.environ.update(OMP_DISPLAY_AFFINITY="false", OMP_DISPLAY_ENV="false", MKL_VERBOSE="0")
        run(args)


if __name__ == "__main__":
    main()
