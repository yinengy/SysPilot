package org.libcontentbench;

import java.io.BufferedWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.ludovic.netlib.blas.NativeBLAS;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/** One real-data codebook fit, full assignment export, and native model save. */
public final class VisualCodebook {
    private static final ObjectMapper JSON = new ObjectMapper();

    private static void event(String name, Map<String, Object> values) throws Exception {
        Map<String, Object> output = new LinkedHashMap<>();
        output.put("event", name);
        output.put("nano_time", System.nanoTime());
        output.putAll(values);
        System.out.println("TASK_EVENT " + JSON.writeValueAsString(output));
        System.out.flush();
    }

    private static Map<String, Object> nativeSettings() {
        String[] values = BlasObserver.snapshot();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("openblas_threads", Integer.parseInt(values[0]));
        result.put("openblas_parallel_mode", Integer.parseInt(values[1]));
        result.put("openblas_config", values[2]);
        result.put("openblas_core", values[3]);
        result.put("openblas_symbol_owner", values[4]);
        return result;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) throw new IllegalArgumentException("task.json learn.fvecs observer.so output-directory");
        JsonNode spec = JSON.readTree(Path.of(args[0]).toFile());
        int n = spec.path("vectors").asInt();
        int dimensions = spec.path("dimensions").asInt();
        int partitions = spec.path("input_partitions").asInt();
        int k = spec.path("k").asInt();
        if (n != 100000 || dimensions != 128 || partitions != 48 || k != 1024)
            throw new IllegalArgumentException("Unexpected task shape");
        Path output = Path.of(args[3]);
        Files.createDirectory(output);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("pid", ProcessHandle.current().pid());
        result.put("task", spec);
        event("jvm_main_started", Map.of("pid", ProcessHandle.current().pid()));

        try (SparkSession spark = SparkSession.builder().appName("SIFT visual codebook training")
                .master("local[*]").config("spark.ui.enabled", false).getOrCreate()) {
            event("spark_started", Map.of("version", spark.version()));
            result.put("spark_version", spark.version());
            result.put("master", spark.sparkContext().master());
            result.put("default_parallelism", spark.sparkContext().defaultParallelism());
            result.put("task_cpus", Integer.parseInt(spark.sparkContext().getConf().get("spark.task.cpus", "1")));
            result.put("java_version", System.getProperty("java.version"));
            Map<String, String> env = new LinkedHashMap<>();
            for (String key : List.of("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS"))
                env.put(key, System.getenv(key));
            result.put("thread_environment", env);
            result.put("blas_class", NativeBLAS.getInstance().getClass().getName());
            System.load(Path.of(args[2]).toAbsolutePath().toString());
            result.put("native_before_fit", nativeSettings());

            byte[] raw = Files.readAllBytes(Path.of(args[1]));
            if (raw.length != n * (4 + dimensions * 4)) throw new IllegalStateException("Incomplete input");
            ByteBuffer input = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            List<Row> inputRows = new ArrayList<>(n);
            double[][] vectors = new double[n][dimensions];
            for (int i = 0; i < n; i++) {
                if (input.getInt() != dimensions) throw new IllegalStateException("Wrong vector dimension");
                double[] feature = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    float value = input.getFloat();
                    if (!Float.isFinite(value) || value < 0 || value > 255)
                        throw new IllegalStateException("Invalid SIFT descriptor");
                    vectors[i][j] = value;
                    feature[j] = value;
                }
                inputRows.add(RowFactory.create((long)i, scala.jdk.javaapi.CollectionConverters.asScala(Arrays.stream(feature).boxed().toList()).toSeq()));
            }
            if (input.hasRemaining()) throw new IllegalStateException("Trailing input bytes");
            JavaSparkContext sc = JavaSparkContext.fromSparkContext(spark.sparkContext());
            StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("features", DataTypes.createArrayType(DataTypes.DoubleType, false), false);
            Dataset<Row> data = spark.createDataFrame(sc.parallelize(inputRows, partitions), schema);
            result.put("input_schema", data.schema().json());
            result.put("input_partitions", data.rdd().getNumPartitions());
            event("fit_started", Map.of("records", n, "partitions", partitions,
                  "native", nativeSettings()));
            long fitStart = System.nanoTime();
            KMeans trainer = new KMeans().setK(k).setSeed(spec.path("seed").asLong())
                .setInitMode(spec.path("init_mode").asText())
                .setInitSteps(spec.path("init_steps").asInt())
                .setMaxIter(spec.path("max_iterations").asInt())
                .setTol(spec.path("tolerance").asDouble())
                .setDistanceMeasure(spec.path("distance_measure").asText())
                .setSolver(spec.path("solver").asText())
                .setMaxBlockSizeInMB(spec.path("max_block_size_mb").asDouble());
            java.lang.reflect.Method selection = KMeans.class.getDeclaredMethod("preferBlockSolver", Dataset.class);
            selection.setAccessible(true);
            boolean blockSelected = (Boolean) selection.invoke(trainer, data);
            result.put("actual_auto_block_selected", blockSelected);
            if (!blockSelected) throw new IllegalStateException("Dense-array AUTO did not select block solver");
            event("auto_solver_verified", Map.of("block_selected", blockSelected, "schema", data.schema().json()));
            KMeansModel model = trainer.fit(data);
            result.put("fit_seconds", (System.nanoTime() - fitStart) / 1e9);
            result.put("iterations", model.summary().numIter());
            result.put("training_cost", model.summary().trainingCost());
            result.put("native_after_fit", nativeSettings());
            event("fit_finished", Map.of("iterations", model.summary().numIter(),
                "training_cost", model.summary().trainingCost(), "native", nativeSettings()));

            Vector[] clusterCenters = model.clusterCenters();
            double[][] centers = Arrays.stream(clusterCenters).map(Vector::toArray).toArray(double[][]::new);
            if (centers.length != k) throw new IllegalStateException("Incomplete codebook");
            JSON.writeValue(output.resolve("centers.json").toFile(), centers);
            model.write().save(output.resolve("spark_model").toString());
            List<Row> assigned = model.transform(data).select("id", "prediction").orderBy("id").collectAsList();
            if (assigned.size() != n) throw new IllegalStateException("Incomplete assignments");
            long[] counts = new long[k];
            double sse = 0;
            try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("assignments.csv"))) {
                writer.write("id,cluster\n");
                for (int i = 0; i < n; i++) {
                    Row row = assigned.get(i);
                    int label = row.getInt(1);
                    if (row.getLong(0) != i || label < 0 || label >= k)
                        throw new IllegalStateException("Invalid assignment identity/label");
                    writer.write(i + "," + label + "\n");
                    counts[label]++;
                    for (int j = 0; j < dimensions; j++) {
                        double residual = vectors[i][j] - centers[label][j];
                        sse += residual * residual;
                    }
                }
            }
            result.put("cluster_sizes", counts);
            result.put("assigned_sse", sse);
            result.put("assigned_vectors", assigned.size());
            result.put("native_after_export", nativeSettings());
            event("model_and_assignments_saved", Map.of("assigned_vectors", assigned.size(), "assigned_sse", sse));
            result.put("status", "OK");
        }
        event("spark_stopped", Map.of());
        JSON.writerWithDefaultPrettyPrinter().writeValue(output.resolve("model_result.json").toFile(), result);
    }
}
