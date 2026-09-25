"""Train an ALS recommender and produce ten recommendations for every user."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile


def prepare(directory):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "ml-20m.zip"
    if not archive.exists():
        with (
            urllib.request.urlopen(
                "https://files.grouplens.org/datasets/movielens/ml-20m.zip", timeout=120
            ) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    expected = "96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("MovieLens archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle, bundle.open("ml-20m/ratings.csv") as source:
        frame = pd.read_csv(
            source,
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
                "timestamp": np.int64,
            },
        )
    users, user_indices = np.unique(frame.userId.to_numpy(), return_inverse=True)
    movies, movie_indices = np.unique(frame.movieId.to_numpy(), return_inverse=True)
    ratings = sp.coo_matrix(
        (frame.rating.to_numpy(), (user_indices, movie_indices)),
        shape=(len(users), len(movies)),
        dtype=np.float32,
    ).tocsr()
    ratings.sort_indices()
    sp.save_npz(directory / "ratings.npz", ratings)
    np.savez_compressed(directory / "ids.npz", users=users, movies=movies)


def check_outputs(arrays, reference):
    import numpy as np

    with np.load(reference / "outputs.npz") as previous:
        for key, values in arrays.items():
            if key == "ids":
                np.testing.assert_array_equal(
                    np.sort(values, axis=1), np.sort(previous[key], axis=1)
                )
                changed = np.flatnonzero(np.any(values != previous[key], axis=1))
                for user in changed:
                    scores_by_id = dict(zip(previous[key][user], previous["scores"][user]))
                    reordered = np.array([scores_by_id[item] for item in values[user]])
                    np.testing.assert_allclose(
                        reordered, previous["scores"][user], rtol=0, atol=1e-5
                    )
            else:
                np.testing.assert_allclose(values, previous[key], rtol=1e-4, atol=1e-5)


def run(args):
    import numpy as np
    import scipy.sparse as sp
    from implicit.cpu.als import AlternatingLeastSquares

    threads = int(os.environ.get("REC_THREADS", "0"))
    if not 0 <= threads <= 48:
        raise ValueError("REC_THREADS must be between 0 and 48")
    matrix = sp.load_npz(args.data_dir / "ratings.npz")
    matrix.sort_indices()
    if matrix.shape != (138493, 26744) or matrix.nnz != 20000263:
        raise ValueError("Expected all 20,000,263 MovieLens20M ratings")

    start = time.perf_counter()
    model = AlternatingLeastSquares(
        num_threads=threads,
        factors=128,
        iterations=15,
        regularization=0.01,
        alpha=1.0,
        dtype=np.float32,
        random_state=20260919,
    )
    model.fit(matrix, show_progress=False)
    fit_time = time.perf_counter() - start
    ids, scores = model.recommend(
        np.arange(matrix.shape[0]), matrix, N=10, filter_already_liked_items=True
    )
    elapsed = time.perf_counter() - start
    arrays = {
        "user_factors": model.user_factors,
        "item_factors": model.item_factors,
        "ids": ids,
        "scores": scores,
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Non-finite recommendation output")
    if args.reference:
        check_outputs(arrays, args.reference)
    np.savez_compressed(args.output_dir / "outputs.npz", **arrays)
    result = {
        "wall_time_sec": elapsed,
        "fit_time_sec": fit_time,
        "recommend_time_sec": elapsed - fit_time,
        "config": {
            key: os.environ.get(key)
            for key in ("REC_THREADS", "OPENBLAS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
        "users": matrix.shape[0],
        "movies": matrix.shape[1],
        "ratings": matrix.nnz,
    }
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
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "OMP_WAIT_POLICY",
            "OMP_MAX_ACTIVE_LEVELS",
            "GOTO_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ.pop(name, None)
        os.environ.update(OMP_DISPLAY_AFFINITY="false", OMP_DISPLAY_ENV="false", MKL_VERBOSE="0")
        for name, value in {
            "REC_THREADS": "0",
            "OPENBLAS_NUM_THREADS": "48",
            "OMP_PROC_BIND": "false",
            "OMP_PLACES": "cores",
        }.items():
            os.environ.setdefault(name, value)
        run(args)


if __name__ == "__main__":
    main()
