"""Train a visual codebook with Spark MLlib and export all descriptor assignments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request

SOURCE = Path(__file__).resolve().parent
TASK = {
    "vectors": 100000,
    "dimensions": 128,
    "input_partitions": 48,
    "k": 1024,
    "seed": 20260922,
    "init_mode": "k-means||",
    "init_steps": 2,
    "max_iterations": 20,
    "tolerance": 0.0001,
    "distance_measure": "euclidean",
    "solver": "auto",
    "max_block_size_mb": 0.0,
}


def download(directory, url, expected):
    archive = directory / url.rsplit("/", 1)[-1]
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {archive}")
    return archive


def prepare(args):
    directory = args.data_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.java_home:
        jdk = args.java_home.resolve()
    else:
        url = (
            "https://github.com/adoptium/temurin21-binaries/releases/download/"
            "jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz"
        )
        archive = download(
            directory, url, "ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94"
        )
        jdk = directory / "jdk-21.0.12.1+1"
        if not jdk.exists():
            with tarfile.open(archive) as bundle:
                bundle.extractall(directory, filter="data")
    if args.spark_home:
        spark = args.spark_home.resolve()
    else:
        url = "https://archive.apache.org/dist/spark/spark-4.1.1/spark-4.1.1-bin-hadoop3.tgz"
        archive = directory / "spark-4.1.1-bin-hadoop3.tgz"
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        digest = hashlib.sha512()
        with archive.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        expected = (
            "9f39e588e7d4c70ec0126109679f386eb9bfa26979dc42669fe4f3e3446a082dca8ff"
            "bf5e8dbe8ad411cf2ce5bf803ce670341620bf52d968067acf86626106e"
        )
        if digest.hexdigest() != expected:
            raise ValueError("Spark archive checksum mismatch")
        spark = directory / "spark-4.1.1-bin-hadoop3"
        if not spark.exists():
            with tarfile.open(archive) as bundle:
                bundle.extractall(directory, filter="data")

    data = directory / "sift_learn.fvecs"
    if not data.exists():
        archive = download(
            directory,
            "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a",
        )
        with tarfile.open(archive) as bundle:
            with bundle.extractfile("sift/sift_learn.fvecs") as source, data.open("wb") as output:
                shutil.copyfileobj(source, output)
    expected = "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea"
    if hashlib.sha256(data.read_bytes()).hexdigest() != expected:
        raise ValueError("SIFT learning split checksum mismatch")

    classes = directory / "classes"
    classes.mkdir(exist_ok=True)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(jdk / "include"),
            "-I" + str(jdk / "include/linux"),
            "-o",
            str(directory / "libblas_observer.so"),
            str(SOURCE / "blas_observer.c"),
            "-ldl",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(jdk / "bin/javac"),
            "-cp",
            str(spark / "jars/*"),
            "-d",
            str(classes),
            str(SOURCE / "BlasObserver.java"),
            str(SOURCE / "VisualCodebook.java"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(jdk / "bin/jar"),
            "--create",
            "--file",
            str(directory / "codebook.jar"),
            "-C",
            str(classes),
            ".",
        ],
        check=True,
    )
    (directory / "task.json").write_text(json.dumps(TASK) + "\n")
    (directory / "paths.json").write_text(json.dumps({"jdk": str(jdk), "spark": str(spark)}) + "\n")
    (directory / "spark-conf").mkdir(exist_ok=True)


def check_outputs(output, reference):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    centers = np.array(json.loads((output / "centers.json").read_text()))
    previous = np.array(json.loads((reference / "centers.json").read_text()))
    rows, columns = linear_sum_assignment(cdist(centers, previous, metric="sqeuclidean"))
    np.testing.assert_allclose(centers[rows], previous[columns], rtol=1e-7, atol=1e-6)
    mapping = np.empty(1024, dtype=int)
    mapping[rows] = columns
    labels = np.loadtxt(output / "assignments.csv", delimiter=",", skiprows=1, dtype=int)
    old_labels = np.loadtxt(reference / "assignments.csv", delimiter=",", skiprows=1, dtype=int)
    np.testing.assert_array_equal(labels[:, 0], old_labels[:, 0])
    np.testing.assert_array_equal(mapping[labels[:, 1]], old_labels[:, 1])
    current = json.loads((output / "model_result.json").read_text())
    old = json.loads((reference / "model_result.json").read_text())
    if current["iterations"] != old["iterations"]:
        raise ValueError("Iteration count changed")
    for metric in ("training_cost", "assigned_sse"):
        np.testing.assert_allclose(current[metric], old[metric], rtol=1e-8, atol=1e-3)


def run(args):
    task_cpus = int(os.environ.get("SPARK_TASK_CPUS", "0"))
    blas_threads = int(os.environ.get("SPARK_BLAS_THREADS", "0"))
    if not (0 <= task_cpus <= 48 and 0 <= blas_threads <= 48):
        raise ValueError("CPU controls must be between 0 and 48")
    if json.loads((args.data_dir / "task.json").read_text()) != TASK:
        raise ValueError("Task settings changed; run prepare again")
    paths = json.loads((args.data_dir / "paths.json").read_text())
    jdk, spark = Path(paths["jdk"]), Path(paths["spark"])
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "LD_PRELOAD",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_WAIT_POLICY",
        "OMP_MAX_ACTIVE_LEVELS",
        "BLIS_NUM_THREADS",
    ):
        environment.pop(name, None)
    environment.update(
        JAVA_HOME=str(jdk),
        PATH=str(jdk / "bin") + ":" + environment["PATH"],
        SPARK_HOME=str(spark),
        SPARK_CONF_DIR=str(args.data_dir / "spark-conf"),
        SPARK_LOCAL_IP="127.0.0.1",
        SPARK_LOCAL_DIRS=str(args.output_dir / "spark-local"),
    )
    if blas_threads:
        environment["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    command = [
        str(spark / "bin/spark-submit"),
        "--class",
        "org.libcontentbench.VisualCodebook",
        "--master",
        "local[*]",
        "--conf",
        "spark.ui.enabled=false",
        "--conf",
        "spark.driver.memory=4g",
    ]
    if task_cpus:
        command += ["--conf", f"spark.task.cpus={task_cpus}"]
    output = args.output_dir / "output"
    command += [
        str(args.data_dir / "codebook.jar"),
        str(args.data_dir / "task.json"),
        str(args.data_dir / "sift_learn.fvecs"),
        str(args.data_dir / "libblas_observer.so"),
        str(output),
    ]
    start = time.perf_counter()
    with (args.output_dir / "spark.log").open("w") as log:
        child = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    elapsed = time.perf_counter() - start
    if code:
        raise RuntimeError(f"Spark exited with {code}; see {args.output_dir / 'spark.log'}")
    result = json.loads((output / "model_result.json").read_text())
    if result["status"] != "OK" or result["assigned_vectors"] != 100000:
        raise ValueError("Incomplete codebook output")
    if not result["actual_auto_block_selected"]:
        raise ValueError("Spark did not select the dense block solver")
    if args.reference:
        check_outputs(output, args.reference / "output")
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "wall_time_sec": elapsed,
                "SPARK_TASK_CPUS": task_cpus,
                "SPARK_BLAS_THREADS": blas_threads,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"wall_time_sec": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path, help="Directory containing a baseline run")
    parser.add_argument("--java-home", type=Path, help="Use an installed JDK 21 during prepare")
    parser.add_argument(
        "--spark-home", type=Path, help="Use an installed Spark 4.1.1 during prepare"
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.action == "prepare":
        prepare(args)
    else:
        if args.output_dir is None:
            parser.error("run requires --output-dir")
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        run(args)


if __name__ == "__main__":
    main()
