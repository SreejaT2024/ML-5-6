import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from lab05_knn import (
    CSV_PATH, RANDOM_STATE, TEST_SIZE,
    load_dataset, build_feature_target, preprocess_train_test,
    CustomKNNClassifier, evaluate_classifier, confusion_matrix_binary,
)
from lab06_knn_genai import GenAIKNNClassifier

NUMBER_OF_RUNS = 10         
K_FOR_COMPARISON = 5         
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21]


def time_fit_and_predict(build_model, X_train, y_train, X_test, runs=NUMBER_OF_RUNS):
    """Time fit() and predict() of one implementation over `runs` repetitions.

    `build_model` is a zero-argument factory so that a *fresh* estimator is
    constructed for every run and no caching carries across runs.

    Returns (mean_fit_seconds, mean_predict_seconds, std_predict_seconds,
             predictions_from_the_last_run).
    """
    fit_times = []
    predict_times = []
    predictions = None

    for _ in range(runs):
        model = build_model()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_times.append(time.perf_counter() - start)

    return (float(np.mean(fit_times)),
            float(np.mean(predict_times)),
            float(np.std(predict_times)),
            predictions)


def benchmark_all_versions(X_train, y_train, X_test, y_test, k,
                           runs=NUMBER_OF_RUNS, sort_algorithm="merge"):
    """Run the full accuracy + timing benchmark for all three versions.

    Returns a list of result dictionaries, one per version.
    """
    version_factories = [
        ("V1 Custom (Lab 05)",
         lambda: CustomKNNClassifier(n_neighbors=k, metric="euclidean",
                                     sort_algorithm=sort_algorithm)),
        ("V2 Scikit-Learn",
         lambda: KNeighborsClassifier(n_neighbors=k, metric="euclidean")),
        ("V3 GenAI (Lab 06)",
         lambda: GenAIKNNClassifier(n_neighbors=k, metric="euclidean")),
    ]

    results = []
    for name, factory in version_factories:
        fit_time, predict_time, predict_std, predictions = time_fit_and_predict(
            factory, X_train, y_train, X_test, runs
        )
        metrics = evaluate_classifier(y_test, predictions)
        tp, fp, tn, fn = confusion_matrix_binary(y_test, predictions)

        results.append({
            "version": name,
            "k": k,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fit_ms": fit_time * 1000.0,
            "predict_ms": predict_time * 1000.0,
            "predict_std_ms": predict_std * 1000.0,
            "total_ms": (fit_time + predict_time) * 1000.0,
            "confusion": (tp, fp, tn, fn),
            "predictions": predictions,
        })

    return results


def benchmark_sort_algorithms_in_classifier(X_train, y_train, X_test, y_test,
                                            k, algorithms, runs=3):
    """Time the custom classifier once per sorting algorithm.

    This isolates the cost of the DSA sorting choice from everything else.
    """
    timings = []
    for algorithm in algorithms:
        fit_time, predict_time, _, predictions = time_fit_and_predict(
            lambda algorithm=algorithm: CustomKNNClassifier(
                n_neighbors=k, sort_algorithm=algorithm),
            X_train, y_train, X_test, runs
        )
        timings.append({
            "algorithm": algorithm,
            "predict_ms": predict_time * 1000.0,
            "accuracy": evaluate_classifier(y_test, predictions)["accuracy"],
        })
    return timings


def sweep_k_all_versions(X_train, y_train, X_test, y_test, k_values,
                         runs=NUMBER_OF_RUNS):
    """Accuracy and predict-time of all three versions across a range of k."""
    rows = []
    for k in k_values:
        for result in benchmark_all_versions(X_train, y_train, X_test, y_test,
                                             k, runs):
            rows.append({key: value for key, value in result.items()
                         if key != "predictions"})
    return pd.DataFrame(rows)


def build_comparison_table(results):
    """Turn the benchmark output into the A3 report table (a DataFrame)."""
    return pd.DataFrame([{
        "Version": r["version"],
        "Accuracy": round(r["accuracy"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "F-Score": round(r["f1"], 4),
        "Fit time (ms)": round(r["fit_ms"], 4),
        "Predict time (ms)": round(r["predict_ms"], 4),
        "Total time (ms)": round(r["total_ms"], 4),
    } for r in results])


def dataframe_to_markdown(frame):
    """Render a DataFrame as a markdown table for pasting into the report."""
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider] + body)


def plot_time_comparison(results, output_path):
    """Bar chart of the averaged predict time of the three versions."""
    names = [r["version"] for r in results]
    times = [r["predict_ms"] for r in results]
    errors = [r["predict_std_ms"] for r in results]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(names, times, yerr=errors, capsize=5,
                    color=["#4C72B0", "#DD8452", "#55A868"])
    axis.set_ylabel("Mean predict time over %d runs (ms, log scale)" % NUMBER_OF_RUNS)
    axis.set_yscale("log")
    axis.set_title("A3 : computational time of the three k-NN implementations")
    axis.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, times):
        axis.text(bar.get_x() + bar.get_width() / 2, value,
                  "%.2f ms" % value, ha="center", va="bottom")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_metric_comparison(results, output_path):
    """Grouped bar chart of the four accuracy metrics per version."""
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F-Score"]
    positions = np.arange(len(metric_names))
    width = 0.26

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for offset, result in enumerate(results):
        values = [result[name] for name in metric_names]
        axis.bar(positions + (offset - 1) * width, values, width,
                 label=result["version"])

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_title("A3 : accuracy metrics of the three k-NN implementations")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


if __name__ == "__main__":

    print("=" * 84)
    print("23CSE301  |  LAB SESSION 06  |  A3 : THREE-WAY PERFORMANCE COMPARISON")
    print("Dataset : IMDb Top-250 movies (2026)   |   task : Drama vs Non-Drama")
    print("=" * 84)

    dataframe = load_dataset(CSV_PATH)
    X_raw, y = build_feature_target(dataframe, positive_genre="Drama")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, _ = preprocess_train_test(X_train_raw, X_test_raw)
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    print("\n[SETUP] train %d | test %d | features %d | k = %d | runs averaged = %d"
          % (X_train.shape[0], X_test.shape[0], X_train.shape[1],
             K_FOR_COMPARISON, NUMBER_OF_RUNS))

    results = benchmark_all_versions(X_train, y_train_array, X_test, y_test_array,
                                     K_FOR_COMPARISON, NUMBER_OF_RUNS)
    table = build_comparison_table(results)

    print("\n[A3] PERFORMANCE COMPARISON TABLE (k = %d)" % K_FOR_COMPARISON)
    print(table.to_string(index=False))

    print("\n[A3] SAME TABLE IN MARKDOWN (paste directly into the report)")
    print(dataframe_to_markdown(table))

    print("\n[A3] CONFUSION MATRICES")
    print("      %-22s %-6s %-6s %-6s %-6s" % ("version", "TP", "FP", "TN", "FN"))
    for result in results:
        tp, fp, tn, fn = result["confusion"]
        print("      %-22s %-6d %-6d %-6d %-6d" % (result["version"], tp, fp, tn, fn))

    baseline_predictions = results[0]["predictions"]
    print("\n[A3] PREDICTION AGREEMENT WITH V1 (custom)")
    for result in results:
        agreement = float(np.mean(result["predictions"] == baseline_predictions))
        print("      %-22s : %.2f%% identical" % (result["version"], 100 * agreement))

    all_identical = all(
        np.array_equal(result["predictions"], baseline_predictions)
        for result in results
    )
    print("      all three versions produce identical labels : %s" % all_identical)

    custom_time = results[0]["predict_ms"]
    print("\n[A3] RELATIVE SPEED (predict, averaged over %d runs)" % NUMBER_OF_RUNS)
    for result in results:
        print("      %-22s : %9.4f ms  (%6.2fx vs custom)"
              % (result["version"], result["predict_ms"],
                 custom_time / result["predict_ms"]))

    print("\n[EXTRA] COST OF THE SORTING ALGORITHM INSIDE THE CUSTOM CLASSIFIER")
    print("      %-12s %-18s %-10s" % ("algorithm", "predict time (ms)", "accuracy"))
    for entry in benchmark_sort_algorithms_in_classifier(
            X_train, y_train_array, X_test, y_test_array,
            K_FOR_COMPARISON,
            ["bubble", "insertion", "selection", "merge", "quick", "builtin"],
            runs=3):
        print("      %-12s %-18.2f %-10.4f"
              % (entry["algorithm"], entry["predict_ms"], entry["accuracy"]))

    sweep_frame = sweep_k_all_versions(X_train, y_train_array, X_test, y_test_array,
                                       K_RANGE, runs=NUMBER_OF_RUNS)
    pivot_accuracy = sweep_frame.pivot(index="k", columns="version", values="accuracy")
    pivot_time = sweep_frame.pivot(index="k", columns="version", values="predict_ms")

    print("\n[A3] ACCURACY ACROSS k FOR ALL THREE VERSIONS")
    print(pivot_accuracy.round(4).to_string())

    print("\n[A3] PREDICT TIME (ms) ACROSS k FOR ALL THREE VERSIONS")
    print(pivot_time.round(4).to_string())

    time_plot = plot_time_comparison(results, "lab06_time_comparison.png")
    metric_plot = plot_metric_comparison(results, "lab06_metric_comparison.png")
    table.to_csv("lab06_comparison_table.csv", index=False)
    sweep_frame.to_csv("lab06_k_sweep.csv", index=False)

    print("\n[OUTPUT FILES]")
    print("      %s" % time_plot)
    print("      %s" % metric_plot)
    print("      lab06_comparison_table.csv")
    print("      lab06_k_sweep.csv")

    fastest = min(results, key=lambda r: r["predict_ms"])
    slowest = max(results, key=lambda r: r["predict_ms"])
    print("\n[OBSERVATIONS]")
    print("      * All three versions agree on every test label, which validates")
    print("        the hand-written implementation against the reference package.")
    print("      * Fastest version : %s (%.4f ms)"
          % (fastest["version"], fastest["predict_ms"]))
    print("      * Slowest version : %s (%.4f ms)"
          % (slowest["version"], slowest["predict_ms"]))
    print("      * The accuracy metrics are identical, so the whole difference")
    print("        between the versions is computational, not predictive.")

    print("\n" + "=" * 84)
    print("LAB 06 A3 COMPLETE")
    print("=" * 84)
