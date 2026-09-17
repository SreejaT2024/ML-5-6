

import ast
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend, safe for servers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


CSV_PATH      = "ToP_250_movies_on_imdb_in_2026.csv"
RANDOM_STATE  = 42
TEST_SIZE     = 0.30

NUMERIC_FEATURES = [
    "startYear", "runtimeMinutes", "averageRating", "numVotes",
    "metascore", "budget", "grossWorldwide",
    "num_countries", "num_languages",
]
CATEGORICAL_FEATURES = ["contentRating"]


def _count_list_items(cell):
    """Count elements in a stringified python list such as "['US', 'GB']".

    Returns 0 when the cell is missing or cannot be parsed.
    """
    if pd.isna(cell):
        return 0
    try:
        parsed = ast.literal_eval(cell)
        return len(parsed) if isinstance(parsed, (list, tuple)) else 0
    except (ValueError, SyntaxError):
        return 0


def _contains_genre(cell, genre):
    """Return True when the stringified genre list contains `genre`."""
    if pd.isna(cell):
        return False
    try:
        parsed = ast.literal_eval(cell)
        return genre in parsed if isinstance(parsed, (list, tuple)) else False
    except (ValueError, SyntaxError):
        return False


def load_dataset(csv_path):
    """Read the project CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def build_feature_target(dataframe, positive_genre="Drama"):
    """Derive the feature matrix (DataFrame) and the binary target (Series).

    Two classes only, as mandated by assignment A3.
    """
    frame = dataframe.copy()

    # engineered features from the stringified list columns
    frame["num_countries"] = frame["countriesOfOrigin"].apply(_count_list_items)
    frame["num_languages"] = frame["spokenLanguages"].apply(_count_list_items)

    # binary class label
    target = frame["genres"].apply(lambda c: _contains_genre(c, positive_genre))
    target = target.map({True: 1, False: 0}).astype(int)

    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return features, target


def build_label_mapping(series):
    """Build a deterministic {category: integer} mapping for one column.

    Missing values are deliberately left out of the map so that the
    imputation module can deal with them afterwards.
    """
    categories = sorted(series.dropna().astype(str).unique())
    return {category: code for code, category in enumerate(categories)}


def apply_label_mapping(series, mapping):
    """Apply a category->code mapping, keeping NaN as NaN."""
    return series.astype(str).map(
        lambda value: mapping.get(value, np.nan) if value != "nan" else np.nan
    )


def encode_categorical(frame, categorical_columns, mappings=None):
    """Label-encode every categorical column of a DataFrame.

    When `mappings` is supplied the existing encoding is reused (this is what
    keeps train and test encodings consistent); otherwise a new one is learnt.
    Returns (encoded_frame, mappings).
    """
    encoded = frame.copy()
    learnt = {} if mappings is None else mappings

    for column in categorical_columns:
        if mappings is None:
            learnt[column] = build_label_mapping(encoded[column])
        encoded[column] = apply_label_mapping(encoded[column], learnt[column])

    return encoded, learnt

def compute_central_tendency(series, strategy="mean"):
    """Return the requested central tendency of a Series.

    strategy: 'mean' | 'median' | 'mode'
    """
    clean = series.dropna()
    if len(clean) == 0:
        return 0.0

    if strategy == "mean":
        return float(clean.mean())
    if strategy == "median":
        return float(clean.median())
    if strategy == "mode":
        modes = clean.mode()
        return float(modes.iloc[0]) if len(modes) else float(clean.iloc[0])

    raise ValueError("strategy must be one of 'mean', 'median', 'mode'")


def fit_imputer(frame, strategy_map, default_strategy="median"):
    """Learn the fill value for every column of the training frame.

    strategy_map : {column_name: 'mean'|'median'|'mode'} for specific columns.
    Any column not listed falls back to `default_strategy`.
    Returns {column: fill_value}.
    """
    fill_values = {}
    for column in frame.columns:
        strategy = strategy_map.get(column, default_strategy)
        fill_values[column] = compute_central_tendency(frame[column], strategy)
    return fill_values


def apply_imputer(frame, fill_values):
    """Fill missing entries using the values learnt by `fit_imputer`."""
    imputed = frame.copy()
    for column, value in fill_values.items():
        if column in imputed.columns:
            imputed[column] = imputed[column].fillna(value)
    return imputed


def count_missing(frame):
    """Return a Series with the number of NaNs per column (diagnostic helper)."""
    return frame.isna().sum()


def fit_standard_scaler(matrix):
    """Learn per-column mean and standard deviation from the training matrix."""
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0                 # guard against constant columns
    return means, stds


def apply_standard_scaler(matrix, means, stds):
    """Z-score normalise a matrix using previously learnt statistics."""
    return (matrix - means) / stds

def euclidean_distance(vector_a, vector_b):
    """L2 distance."""
    difference = vector_a - vector_b
    return float(np.sqrt(np.dot(difference, difference)))


def manhattan_distance(vector_a, vector_b):
    """L1 / city-block distance."""
    return float(np.sum(np.abs(vector_a - vector_b)))


def chebyshev_distance(vector_a, vector_b):
    """L-infinity distance."""
    return float(np.max(np.abs(vector_a - vector_b)))


def minkowski_distance(vector_a, vector_b, order=3):
    """Generalised Lp distance; order=1 -> Manhattan, order=2 -> Euclidean."""
    return float(np.sum(np.abs(vector_a - vector_b) ** order) ** (1.0 / order))


def cosine_distance(vector_a, vector_b):
    """1 - cosine similarity. Bounded in [0, 2]."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(vector_a, vector_b) / (norm_a * norm_b))

DISTANCE_METRICS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "minkowski": minkowski_distance,
    "cosine":    cosine_distance,
}


def compute_distance(vector_a, vector_b, metric="euclidean", order=3):
    """Dispatch to the distance function selected by `metric`."""
    if metric not in DISTANCE_METRICS:
        raise ValueError("unknown metric: %s" % metric)
    if metric == "minkowski":
        return minkowski_distance(vector_a, vector_b, order)
    return DISTANCE_METRICS[metric](vector_a, vector_b)


def compute_distance_vector(train_matrix, test_vector, metric="euclidean", order=3):
    """Distance from one test pattern to every training pattern.

    Returns a python list of floats of length len(train_matrix).
    """
    return [
        compute_distance(train_matrix[row], test_vector, metric, order)
        for row in range(train_matrix.shape[0])
    ]

def bubble_sort(items):
    """Bubble sort - O(n^2), with early exit when a pass makes no swap."""
    data = list(items)
    length = len(data)
    for outer in range(length - 1):
        swapped = False
        for inner in range(length - 1 - outer):
            if data[inner] > data[inner + 1]:
                data[inner], data[inner + 1] = data[inner + 1], data[inner]
                swapped = True
        if not swapped:
            break
    return data


def insertion_sort(items):
    """Insertion sort - O(n^2) worst case, O(n) on nearly-sorted input."""
    data = list(items)
    for index in range(1, len(data)):
        current = data[index]
        position = index - 1
        while position >= 0 and data[position] > current:
            data[position + 1] = data[position]
            position -= 1
        data[position + 1] = current
    return data


def selection_sort(items):
    """Selection sort - O(n^2), minimum number of writes."""
    data = list(items)
    length = len(data)
    for outer in range(length - 1):
        smallest = outer
        for inner in range(outer + 1, length):
            if data[inner] < data[smallest]:
                smallest = inner
        if smallest != outer:
            data[outer], data[smallest] = data[smallest], data[outer]
    return data


def merge_sort(items):
    """Merge sort - O(n log n), stable, divide and conquer."""
    data = list(items)
    if len(data) <= 1:
        return data

    middle = len(data) // 2
    left = merge_sort(data[:middle])
    right = merge_sort(data[middle:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(items):
    """Quick sort - O(n log n) average, median-of-three style middle pivot."""
    data = list(items)
    if len(data) <= 1:
        return data

    pivot = data[len(data) // 2]
    smaller = [x for x in data if x < pivot]
    equal = [x for x in data if x == pivot]
    larger = [x for x in data if x > pivot]
    return quick_sort(smaller) + equal + quick_sort(larger)


def builtin_sort(items):
    """Python's Timsort - reference implementation used for benchmarking."""
    return sorted(items)


SORTING_ALGORITHMS = {
    "bubble":    bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "merge":     merge_sort,
    "quick":     quick_sort,
    "builtin":   builtin_sort,
}


def sort_distances(distance_index_pairs, algorithm="merge"):
    """Sort (distance, index) pairs ascending using the chosen algorithm."""
    if algorithm not in SORTING_ALGORITHMS:
        raise ValueError("unknown sorting algorithm: %s" % algorithm)
    return SORTING_ALGORITHMS[algorithm](distance_index_pairs)

def identify_neighbors(distances, k, sort_algorithm="merge"):
    """Return the k nearest neighbours as a list of (distance, index).

    Tie-breaking rule 1: when two training patterns sit at exactly the same
    distance, the pattern with the smaller training index is preferred. This is
    achieved by sorting (distance, index) tuples, which compare
    lexicographically, and it makes the classifier fully deterministic.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    pairs = [(float(distance), index) for index, distance in enumerate(distances)]
    ordered = sort_distances(pairs, sort_algorithm)
    effective_k = min(k, len(ordered))
    return ordered[:effective_k]


def majority_vote(neighbor_labels, neighbor_distances=None):
    """Assign a class label by simple majority voting.

    Tie-breaking rule 2: if two or more classes receive the same number of
    votes, the class owning the single closest neighbour among the tied classes
    wins. If distances are unavailable, the numerically smallest label wins.
    """
    counts = {}
    for label in neighbor_labels:
        counts[label] = counts.get(label, 0) + 1

    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]

    if len(winners) == 1:
        return winners[0]

    if neighbor_distances is None:
        return min(winners)

    # closest-neighbour tie break
    best_label, best_distance = None, float("inf")
    for label, distance in zip(neighbor_labels, neighbor_distances):
        if label in winners and distance < best_distance:
            best_label, best_distance = label, distance
    return best_label


def distance_weight(distance, epsilon=1e-9):
    """Inverse-square-distance weight. Closer neighbours count for more."""
    return 1.0 / (distance ** 2 + epsilon)


def weighted_vote(neighbor_labels, neighbor_distances, epsilon=1e-9):
    """Assign a class label by weighted voting (weight = 1 / d^2).

    Tie-breaking rule: exact ties in accumulated weight are broken by the
    closest neighbour belonging to one of the tied classes.
    """
    scores = {}
    for label, distance in zip(neighbor_labels, neighbor_distances):
        scores[label] = scores.get(label, 0.0) + distance_weight(distance, epsilon)

    highest = max(scores.values())
    winners = [label for label, score in scores.items() if score == highest]

    if len(winners) == 1:
        return winners[0]

    best_label, best_distance = None, float("inf")
    for label, distance in zip(neighbor_labels, neighbor_distances):
        if label in winners and distance < best_distance:
            best_label, best_distance = label, distance
    return best_label

class CustomKNNClassifier:
    """Hand-written k-Nearest Neighbours classifier.

    Parameters
    ----------
    n_neighbors     : int   - the k of kNN
    metric          : str   - key into DISTANCE_METRICS
    order           : int   - p of the Minkowski metric
    sort_algorithm  : str   - key into SORTING_ALGORITHMS
    weighted        : bool  - False -> majority vote (A1), True -> weighted (A2)
    """

    def __init__(self, n_neighbors=3, metric="euclidean", order=3,
                 sort_algorithm="merge", weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.order = order
        self.sort_algorithm = sort_algorithm
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    # ---------------------------------------------------------------- fit
    def fit(self, X_train, y_train):
        """kNN is a lazy learner: 'training' simply stores the patterns."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if self.n_neighbors > self.X_train.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of samples")
        return self

    # ------------------------------------------------------- predict_one
    def predict_one(self, test_vector):
        """Classify a single test pattern."""
        distances = compute_distance_vector(
            self.X_train, test_vector, self.metric, self.order
        )
        neighbors = identify_neighbors(
            distances, self.n_neighbors, self.sort_algorithm
        )
        neighbor_distances = [distance for distance, _ in neighbors]
        neighbor_labels = [self.y_train[index] for _, index in neighbors]

        if self.weighted:
            return weighted_vote(neighbor_labels, neighbor_distances)
        return majority_vote(neighbor_labels, neighbor_distances)

    # ------------------------------------------------------------ predict
    def predict(self, X_test):
        """Classify every row of X_test; returns a numpy array of labels."""
        if self.X_train is None:
            raise RuntimeError("call fit() before predict()")
        matrix = np.asarray(X_test, dtype=float)
        return np.array([self.predict_one(matrix[row])
                         for row in range(matrix.shape[0])])

    # -------------------------------------------------------------- score
    def score(self, X_test, y_test):
        """Mean accuracy on the supplied test set."""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == np.asarray(y_test)))

    # ------------------------------------------------------- kneighbors
    def kneighbors(self, test_vector):
        """Expose the neighbour list of one pattern (useful for debugging)."""
        distances = compute_distance_vector(
            self.X_train, test_vector, self.metric, self.order
        )
        return identify_neighbors(distances, self.n_neighbors, self.sort_algorithm)


def confusion_matrix_binary(y_true, y_predicted, positive_label=1):
    """Return (true_pos, false_pos, true_neg, false_neg)."""
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_predicted)

    true_positive = int(np.sum((true_array == positive_label) &
                               (pred_array == positive_label)))
    false_positive = int(np.sum((true_array != positive_label) &
                                (pred_array == positive_label)))
    true_negative = int(np.sum((true_array != positive_label) &
                               (pred_array != positive_label)))
    false_negative = int(np.sum((true_array == positive_label) &
                                (pred_array != positive_label)))
    return true_positive, false_positive, true_negative, false_negative


def accuracy_score_custom(y_true, y_predicted):
    """Fraction of correctly classified patterns."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_predicted)))


def precision_score_custom(y_true, y_predicted, positive_label=1):
    """TP / (TP + FP); returns 0.0 when nothing was predicted positive."""
    tp, fp, _, _ = confusion_matrix_binary(y_true, y_predicted, positive_label)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score_custom(y_true, y_predicted, positive_label=1):
    """TP / (TP + FN); returns 0.0 when there is no positive ground truth."""
    tp, _, _, fn = confusion_matrix_binary(y_true, y_predicted, positive_label)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score_custom(y_true, y_predicted, positive_label=1):
    """Harmonic mean of precision and recall."""
    precision = precision_score_custom(y_true, y_predicted, positive_label)
    recall = recall_score_custom(y_true, y_predicted, positive_label)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_classifier(y_true, y_predicted, positive_label=1):
    """Bundle all four metrics into a dictionary."""
    return {
        "accuracy":  accuracy_score_custom(y_true, y_predicted),
        "precision": precision_score_custom(y_true, y_predicted, positive_label),
        "recall":    recall_score_custom(y_true, y_predicted, positive_label),
        "f1":        f1_score_custom(y_true, y_predicted, positive_label),
    }


def preprocess_train_test(X_train_raw, X_test_raw, imputation_strategies=None):
    """Encode -> impute -> scale, learning every statistic on the TRAIN split
    only and then applying it to the test split (prevents data leakage).

    Returns (X_train_matrix, X_test_matrix, artefacts_dictionary).
    """
    if imputation_strategies is None:
        imputation_strategies = {
            "budget":         "median",   # heavily right-skewed -> median
            "grossWorldwide": "median",   # heavily right-skewed -> median
            "metascore":      "mean",     # roughly symmetric    -> mean
            "contentRating":  "mode",     # categorical          -> mode
        }

    # ---- A1(a) encoding
    train_encoded, mappings = encode_categorical(X_train_raw, CATEGORICAL_FEATURES)
    test_encoded, _ = encode_categorical(X_test_raw, CATEGORICAL_FEATURES, mappings)

    # ---- A1(b) imputation
    fill_values = fit_imputer(train_encoded, imputation_strategies)
    train_imputed = apply_imputer(train_encoded, fill_values)
    test_imputed = apply_imputer(test_encoded, fill_values)

    # ---- scaling
    train_matrix = train_imputed.to_numpy(dtype=float)
    test_matrix = test_imputed.to_numpy(dtype=float)
    means, stds = fit_standard_scaler(train_matrix)

    artefacts = {
        "mappings": mappings,
        "fill_values": fill_values,
        "means": means,
        "stds": stds,
        "missing_before_train": count_missing(train_encoded),
    }
    return (apply_standard_scaler(train_matrix, means, stds),
            apply_standard_scaler(test_matrix, means, stds),
            artefacts)

def sweep_k_values(X_train, y_train, X_test, y_test, k_values,
                   metric="euclidean", sort_algorithm="merge"):
    """Accuracy of custom-plain, custom-weighted and sklearn kNN over a k range.

    Returns a dict of lists, all aligned with `k_values`.
    """
    results = {"custom": [], "custom_weighted": [], "sklearn": [],
               "sklearn_weighted": [], "train_custom": []}

    for k in k_values:
        plain = CustomKNNClassifier(n_neighbors=k, metric=metric,
                                    sort_algorithm=sort_algorithm, weighted=False)
        plain.fit(X_train, y_train)
        results["custom"].append(plain.score(X_test, y_test))
        results["train_custom"].append(plain.score(X_train, y_train))

        weighted = CustomKNNClassifier(n_neighbors=k, metric=metric,
                                       sort_algorithm=sort_algorithm, weighted=True)
        weighted.fit(X_train, y_train)
        results["custom_weighted"].append(weighted.score(X_test, y_test))

        package = KNeighborsClassifier(n_neighbors=k, metric=metric)
        package.fit(X_train, y_train)
        results["sklearn"].append(float(package.score(X_test, y_test)))

        package_weighted = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                                weights="distance")
        package_weighted.fit(X_train, y_train)
        results["sklearn_weighted"].append(float(package_weighted.score(X_test, y_test)))

    return results


def plot_accuracy_comparison(k_values, results, output_path):
    """Draw the A8/A9 accuracy-versus-k comparison plot and save it."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(k_values, results["custom"], "o-", label="Custom kNN (majority vote)")
    axis.plot(k_values, results["sklearn"], "s--", label="Scikit-Learn kNN (uniform)")
    axis.plot(k_values, results["custom_weighted"], "^-", label="Custom weighted kNN")
    axis.plot(k_values, results["sklearn_weighted"], "v--", label="Scikit-Learn kNN (distance)")

    axis.set_xlabel("k (number of neighbours)")
    axis.set_ylabel("Test-set accuracy")
    axis.set_title("A8 / A9 : Accuracy vs k -- custom implementation vs Scikit-Learn")
    axis.set_xticks(list(k_values))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_fit_diagnosis(k_values, results, output_path):
    """Train vs test accuracy -- used to reason about over/under-fitting."""
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(k_values, results["train_custom"], "o-", label="Train accuracy")
    axis.plot(k_values, results["custom"], "s-", label="Test accuracy")
    axis.set_xlabel("k (number of neighbours)")
    axis.set_ylabel("Accuracy")
    axis.set_title("Fit diagnosis : train vs test accuracy of the custom kNN")
    axis.set_xticks(list(k_values))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def benchmark_sorting_algorithms(distances, k, algorithms=None, repeats=5):
    """Time every sorting algorithm on the same distance list.

    Returns {algorithm_name: (average_seconds, top_k_result)}.
    """
    if algorithms is None:
        algorithms = ["bubble", "insertion", "selection", "merge", "quick", "builtin"]

    timings = {}
    for name in algorithms:
        elapsed_total = 0.0
        neighbours = None
        for _ in range(repeats):
            start = time.perf_counter()
            neighbours = identify_neighbors(distances, k, name)
            elapsed_total += time.perf_counter() - start
        timings[name] = (elapsed_total / repeats, neighbours)
    return timings


def compare_distance_metrics(X_train, y_train, X_test, y_test, k=3):
    """Accuracy of the custom classifier under each available distance metric."""
    scores = {}
    for metric in DISTANCE_METRICS:
        model = CustomKNNClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)
        scores[metric] = model.score(X_test, y_test)
    return scores

if __name__ == "__main__":

    print("=" * 78)
    print("23CSE301  |  LAB SESSION 05  |  k-NEAREST NEIGHBOURS CLASSIFIER")
    print("Dataset : IMDb Top-250 movies (2026)")
    print("=" * 78)

    # ------------------------------------------------ data loading
    raw_dataframe = load_dataset(CSV_PATH)
    X_raw, y = build_feature_target(raw_dataframe, positive_genre="Drama")

    print("\n[DATA] rows x columns in raw file : %s" % (raw_dataframe.shape,))
    print("[DATA] feature matrix shape       : %s" % (X_raw.shape,))
    print("[DATA] features used              : %s"
          % (NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    print("[DATA] class distribution         : Drama=%d  Non-Drama=%d"
          % (int((y == 1).sum()), int((y == 0).sum())))

    print("\n[MISSING VALUES BEFORE IMPUTATION]")
    for column, number in count_missing(X_raw).items():
        if number > 0:
            print("    %-16s : %d missing" % (column, number))

    # ------------------------------------------------ A3 : train / test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("\n[A3] train-test split with test_size = %.2f" % TEST_SIZE)
    print("     training patterns : %d" % len(X_train_raw))
    print("     test patterns     : %d" % len(X_test_raw))

    # ------------------------------------------------ A1(a),(b) preprocessing
    X_train, X_test, artefacts = preprocess_train_test(X_train_raw, X_test_raw)
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    print("\n[A1a] ENCODING -- contentRating label map (first 6 entries):")
    for category, code in list(artefacts["mappings"]["contentRating"].items())[:6]:
        print("      %-12s -> %d" % (category, code))

    print("\n[A1b] IMPUTATION -- fill values learnt on the training split:")
    for column in ["metascore", "budget", "grossWorldwide", "contentRating"]:
        print("      %-16s -> %.4f" % (column, artefacts["fill_values"][column]))
    print("      remaining NaNs after imputation : %d"
          % int(np.isnan(X_train).sum() + np.isnan(X_test).sum()))

    # ------------------------------------------------ A1(c) distance metrics
    print("\n[A1c] DISTANCE CALCULATION -- test pattern 0 vs training pattern 0")
    for metric_name in DISTANCE_METRICS:
        value = compute_distance(X_train[0], X_test[0], metric_name)
        print("      %-10s : %.6f" % (metric_name, value))

    # ------------------------------------------------ A1(d) sorting
    sample_distances = compute_distance_vector(X_train, X_test[0], "euclidean")
    sorting_timings = benchmark_sorting_algorithms(sample_distances, k=3)

    print("\n[A1d] SORTING ALGORITHMS -- averaged over 5 runs on %d distances"
          % len(sample_distances))
    print("      %-12s %-14s %s" % ("algorithm", "avg time (ms)", "top-3 (dist, idx)"))
    reference_top3 = None
    for name, (elapsed, top_k) in sorting_timings.items():
        if reference_top3 is None:
            reference_top3 = top_k
        formatted = ", ".join("(%.3f, %d)" % (d, i) for d, i in top_k)
        print("      %-12s %-14.4f %s" % (name, elapsed * 1000, formatted))
    identical = all(top_k == reference_top3 for _, top_k in sorting_timings.values())
    print("      all algorithms returned identical neighbours : %s" % identical)

    # ------------------------------------------------ A1(e) neighbours
    print("\n[A1e] NEAREST NEIGHBOURS of test pattern 0 (k=5, merge sort)")
    for rank, (distance, index) in enumerate(
            identify_neighbors(sample_distances, 5, "merge"), start=1):
        print("      rank %d : train index %-4d distance %.4f  true class %d"
              % (rank, index, distance, y_train_array[index]))

    # ------------------------------------------------ A1(f) voting
    neighbours_5 = identify_neighbors(sample_distances, 5, "merge")
    labels_5 = [y_train_array[index] for _, index in neighbours_5]
    distances_5 = [distance for distance, _ in neighbours_5]
    print("\n[A1f] CLASS ASSIGNMENT for test pattern 0")
    print("      neighbour labels   : %s" % [int(label) for label in labels_5])
    print("      majority vote      : %d" % majority_vote(labels_5, distances_5))
    print("      weighted vote (A2) : %d" % weighted_vote(labels_5, distances_5))
    print("      ground truth       : %d" % y_test_array[0])
    print("      tie-break demo, labels [0,0,1,1] -> majority vote gives %d"
          % majority_vote([0, 0, 1, 1], [0.9, 1.0, 0.2, 1.5]))

    # ------------------------------------------------ A4, A5, A6 : sklearn
    package_model = KNeighborsClassifier(n_neighbors=3)
    package_model.fit(X_train, y_train_array)
    package_accuracy = package_model.score(X_test, y_test_array)
    package_predictions = package_model.predict(X_test)

    print("\n[A4] Scikit-Learn KNeighborsClassifier(n_neighbors=3) trained.")
    print("[A5] package test accuracy  : %.4f" % package_accuracy)
    print("[A6] first 20 predictions   : %s" % package_predictions[:20])
    print("     first 20 ground truth  : %s" % y_test_array[:20])

    # ------------------------------------------------ A7 : own package
    custom_model = CustomKNNClassifier(n_neighbors=3, metric="euclidean",
                                       sort_algorithm="merge", weighted=False)
    custom_model.fit(X_train, y_train_array)
    custom_accuracy = custom_model.score(X_test, y_test_array)
    custom_predictions = custom_model.predict(X_test)

    print("\n[A7] CustomKNNClassifier(k=3) -- fit(), predict(), score()")
    print("     custom test accuracy   : %.4f" % custom_accuracy)
    print("     first 20 predictions   : %s" % custom_predictions[:20])
    print("     agreement with sklearn : %.2f%% of test patterns"
          % (100.0 * np.mean(custom_predictions == package_predictions)))

    custom_metrics = evaluate_classifier(y_test_array, custom_predictions)
    package_metrics = evaluate_classifier(y_test_array, package_predictions)
    print("\n     %-22s %-10s %-10s %-10s %-10s"
          % ("implementation", "accuracy", "precision", "recall", "f1"))
    for name, metrics in [("custom (k=3)", custom_metrics),
                          ("scikit-learn (k=3)", package_metrics)]:
        print("     %-22s %-10.4f %-10.4f %-10.4f %-10.4f"
              % (name, metrics["accuracy"], metrics["precision"],
                 metrics["recall"], metrics["f1"]))

    tp, fp, tn, fn = confusion_matrix_binary(y_test_array, custom_predictions)
    print("\n     confusion matrix (custom) : TP=%d  FP=%d  TN=%d  FN=%d"
          % (tp, fp, tn, fn))

    # ------------------------------------------------ distance metric study
    print("\n[EXTRA] custom kNN accuracy (k=3) under different distance metrics")
    for metric_name, value in compare_distance_metrics(
            X_train, y_train_array, X_test, y_test_array, k=3).items():
        print("      %-10s : %.4f" % (metric_name, value))

    # ------------------------------------------------ A8 and A9 : k sweep
    k_range = list(range(1, 22, 2))
    sweep = sweep_k_values(X_train, y_train_array, X_test, y_test_array, k_range)

    print("\n[A8/A9] ACCURACY OVER A RANGE OF k")
    print("      %-4s %-12s %-12s %-14s %-14s %-12s"
          % ("k", "custom", "sklearn", "custom-wtd", "sklearn-wtd", "train(custom)"))
    for position, k in enumerate(k_range):
        print("      %-4d %-12.4f %-12.4f %-14.4f %-14.4f %-12.4f"
              % (k, sweep["custom"][position], sweep["sklearn"][position],
                 sweep["custom_weighted"][position],
                 sweep["sklearn_weighted"][position],
                 sweep["train_custom"][position]))

    max_gap = max(abs(a - b) for a, b in zip(sweep["custom"], sweep["sklearn"]))
    print("\n      largest custom-vs-sklearn accuracy gap : %.4f" % max_gap)

    best_index = int(np.argmax(sweep["custom"]))
    best_weighted_index = int(np.argmax(sweep["custom_weighted"]))
    print("      best plain    k = %d  accuracy %.4f"
          % (k_range[best_index], sweep["custom"][best_index]))
    print("      best weighted k = %d  accuracy %.4f"
          % (k_range[best_weighted_index], sweep["custom_weighted"][best_weighted_index]))

    accuracy_plot = plot_accuracy_comparison(k_range, sweep, "lab05_accuracy_vs_k.png")
    fit_plot = plot_fit_diagnosis(k_range, sweep, "lab05_fit_diagnosis.png")
    print("\n      plot saved : %s" % accuracy_plot)
    print("      plot saved : %s" % fit_plot)

    # ------------------------------------------------ fit diagnosis
    train_at_1 = sweep["train_custom"][0]
    gap_at_best = (sweep["train_custom"][best_index] - sweep["custom"][best_index])
    print("\n[FIT DIAGNOSIS]")
    print("      train accuracy at k=1  : %.4f  (memorises the training set)"
          % train_at_1)
    print("      train-test gap at k=%-2d : %.4f" % (k_range[best_index], gap_at_best))
    print("      a large positive gap indicates over-fitting (small k);")
    print("      both accuracies falling together indicates under-fitting (large k).")

    print("\n" + "=" * 78)
    print("LAB 05 COMPLETE")
    print("=" * 78)
