"""Encode scientific abstracts and retrieve the ten nearest documents per query."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def prepare(directory):
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=directory / "model",
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    archive = directory / "scifact.zip"
    if not archive.exists():
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    expected = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("SciFact archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            bundle.extract("scifact/" + name, directory)


def run(args):
    controls = {
        name: int(os.environ.get(name, str(default)))
        for name, default in {
            "ST_PROCESSES": 4,
            "ST_INTRA_THREADS": 0,
            "ST_TOKENIZER_THREADS": 0,
            "ST_BATCH_SIZE": 32,
            "ST_CHUNK_SIZE": 0,
        }.items()
    }
    bounds = {
        "ST_PROCESSES": (1, 8),
        "ST_INTRA_THREADS": (0, 48),
        "ST_TOKENIZER_THREADS": (0, 48),
        "ST_BATCH_SIZE": (1, 128),
        "ST_CHUNK_SIZE": (0, 2048),
    }
    for name, (low, high) in bounds.items():
        if not low <= controls[name] <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    for alias, names in (
        ("ST_INTRA_THREADS", ("OMP_NUM_THREADS", "MKL_NUM_THREADS")),
        ("ST_TOKENIZER_THREADS", ("RAYON_NUM_THREADS",)),
    ):
        for name in names:
            if controls[alias]:
                os.environ[name] = str(controls[alias])
            else:
                os.environ.pop(name, None)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="true",
    )

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    folder = args.data_dir / "scifact"
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text().splitlines()]
    queries = {
        row["_id"]: row
        for row in map(json.loads, (folder / "queries.jsonl").read_text().splitlines())
    }
    ids = sorted(
        {line.split("\t")[0] for line in (folder / "qrels/test.tsv").read_text().splitlines()[1:]}
    )
    documents = [row["title"] + " " + row["text"] for row in corpus]
    texts = [queries[key]["text"] for key in ids]
    if len(documents) != 5183 or len(texts) != 300:
        raise ValueError("Expected 5,183 abstracts and 300 test queries")
    model = SentenceTransformer(str(args.data_dir / "model"), device="cpu")
    model.eval()

    start = time.perf_counter()
    pool = model.start_multi_process_pool(target_devices=["cpu"] * controls["ST_PROCESSES"])
    try:
        options = dict(
            batch_size=controls["ST_BATCH_SIZE"],
            chunk_size=controls["ST_CHUNK_SIZE"] or None,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = model.encode_multi_process(documents, pool, **options)
        query_embeddings = model.encode_multi_process(texts, pool, **options)
    finally:
        model.stop_multi_process_pool(pool)
    encoding_time = time.perf_counter() - start
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    scores, indices = index.search(query_embeddings, 10)
    elapsed = time.perf_counter() - start

    arrays = {
        "embeddings": embeddings,
        "query_embeddings": query_embeddings,
        "ids": indices,
        "scores": scores,
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Non-finite retrieval output")
    if args.reference:
        with np.load(args.reference / "outputs.npz") as previous:
            for key, values in arrays.items():
                if key == "ids":
                    np.testing.assert_array_equal(values, previous[key])
                else:
                    np.testing.assert_allclose(values, previous[key], rtol=1e-4, atol=2e-6)
    np.savez_compressed(args.output_dir / "outputs.npz", **arrays)
    result = {
        "wall_time_sec": elapsed,
        "encoding_time_sec": encoding_time,
        "retrieval_time_sec": elapsed - encoding_time,
        "config": controls,
        "documents": len(documents),
        "queries": len(texts),
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
