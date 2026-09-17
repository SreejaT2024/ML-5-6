
import numpy as np

def euclidean_distance_matrix(X_test, X_train):
    """Pairwise L2 distances, shape (n_test, n_train).

    [GenAI: Claude] Uses the expansion ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2 so
    that the whole matrix falls out of a single BLAS matrix product.
    """
    test_sq = np.sum(X_test ** 2, axis=1).reshape(-1, 1)
    train_sq = np.sum(X_train ** 2, axis=1).reshape(1, -1)
    cross = X_test @ X_train.T
    squared = np.maximum(test_sq - 2.0 * cross + train_sq, 0.0)
    return np.sqrt(squared)


def manhattan_distance_matrix(X_test, X_train):
    """Pairwise L1 distances via broadcasting.

    [GenAI: Claude] generated the broadcasting expression.
    """
    return np.sum(np.abs(X_test[:, None, :] - X_train[None, :, :]), axis=2)


def chebyshev_distance_matrix(X_test, X_train):
    """Pairwise L-infinity distances.

    [GenAI: Claude] generated the broadcasting expression.
    """
    return np.max(np.abs(X_test[:, None, :] - X_train[None, :, :]), axis=2)


def minkowski_distance_matrix(X_test, X_train, order=3):
    """Pairwise Lp distances for arbitrary p.

    [GenAI: Claude] generated the broadcasting expression.
    """
    differences = np.abs(X_test[:, None, :] - X_train[None, :, :])
    return np.sum(differences ** order, axis=2) ** (1.0 / order)


def cosine_distance_matrix(X_test, X_train):
    """Pairwise cosine distances (1 - cosine similarity).

    [GenAI: Claude] generated the norm-guarding logic for zero vectors.
    """
    test_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
    train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
    test_norms[test_norms == 0] = 1.0
    train_norms[train_norms == 0] = 1.0
    similarity = (X_test / test_norms) @ (X_train / train_norms).T
    return 1.0 - similarity


DISTANCE_MATRIX_KERNELS = {
    "euclidean": euclidean_distance_matrix,
    "manhattan": manhattan_distance_matrix,
    "chebyshev": chebyshev_distance_matrix,
    "minkowski": minkowski_distance_matrix,
    "cosine":    cosine_distance_matrix,
}


def compute_distance_matrix(X_test, X_train, metric="euclidean", order=3):
    """Dispatch to the requested vectorised distance kernel."""
    if metric not in DISTANCE_MATRIX_KERNELS:
        raise ValueError("unknown metric: %s" % metric)
    if metric == "minkowski":
        return minkowski_distance_matrix(X_test, X_train, order)
    return DISTANCE_MATRIX_KERNELS[metric](X_test, X_train)


def fit_column_fill_values(matrix, strategies):
    """Learn a fill value per column of a numeric matrix that may contain NaN.

    strategies : list of 'mean' | 'median' | 'mode', one entry per column.
    [GenAI: Claude] generated the nan-aware reductions and the mode routine.
    """
    fill_values = np.zeros(matrix.shape[1], dtype=float)

    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = values[~np.isnan(values)]

        if finite.size == 0:
            fill_values[column] = 0.0
            continue

        strategy = strategies[column]
        if strategy == "mean":
            fill_values[column] = float(np.mean(finite))
        elif strategy == "median":
            fill_values[column] = float(np.median(finite))
        elif strategy == "mode":
            uniques, counts = np.unique(finite, return_counts=True)
            fill_values[column] = float(uniques[np.argmax(counts)])
        else:
            raise ValueError("unknown strategy: %s" % strategy)

    return fill_values


def apply_column_fill_values(matrix, fill_values):
    """Replace every NaN with the learnt per-column fill value.

    [GenAI: Claude] generated the np.where broadcast.
    """
    filled = matrix.copy()
    indices = np.where(np.isnan(filled))
    filled[indices] = np.take(fill_values, indices[1])
    return filled


def encode_labels_vectorised(values):
    """Integer-encode an array of categorical labels.

    Returns (encoded_array, {category: code}).
    [GenAI: Claude] generated the np.unique based encoder.
    """
    categories = np.unique(values.astype(str))
    mapping = {category: code for code, category in enumerate(categories)}
    encoded = np.array([mapping[str(value)] for value in values], dtype=float)
    return encoded, mapping


def standardise(matrix, means=None, stds=None):
    """Z-score normalise; learns the statistics when they are not supplied.

    [GenAI: Claude] generated the zero-variance guard.
    """
    if means is None:
        means = matrix.mean(axis=0)
    if stds is None:
        stds = matrix.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
    return (matrix - means) / stds, means, stds


def select_k_nearest(distance_row, k):
    """Indices of the k smallest distances, ordered nearest first.

    [GenAI: Claude] np.argpartition gives an O(n) partial selection; the small
    slice is then sorted by (distance, index) so that equal distances are broken
    by the smaller training index -- identical to the Lab-05 tie-break rule.
    """
    k = min(k, distance_row.shape[0])
    candidate_indices = np.argpartition(distance_row, k - 1)[:k]
    ordering = np.lexsort((candidate_indices, distance_row[candidate_indices]))
    return candidate_indices[ordering]


def vote_majority(labels, distances):
    """Unweighted majority vote with closest-neighbour tie breaking.

    [GenAI: Claude] generated the bincount tally and the tie-break branch.
    """
    classes, counts = np.unique(labels, return_counts=True)
    winners = classes[counts == counts.max()]

    if winners.size == 1:
        return winners[0]

    for position in np.argsort(distances, kind="stable"):
        if labels[position] in winners:
            return labels[position]
    return winners[0]


def vote_weighted(labels, distances, epsilon=1e-9):
    """Inverse-square-distance weighted vote with the same tie-break rule.

    [GenAI: Claude] generated the weight accumulation loop.
    """
    weights = 1.0 / (distances ** 2 + epsilon)
    classes = np.unique(labels)
    scores = np.array([weights[labels == cls].sum() for cls in classes])
    winners = classes[scores == scores.max()]

    if winners.size == 1:
        return winners[0]

    for position in np.argsort(distances, kind="stable"):
        if labels[position] in winners:
            return labels[position]
    return winners[0]

class GenAIKNNClassifier:
    """Vectorised k-Nearest Neighbours classifier (GenAI-generated internals).

    Exposes exactly the same API as the Lab-05 CustomKNNClassifier and as
    sklearn's KNeighborsClassifier, so all three can be benchmarked with one
    harness.
    """

    def __init__(self, n_neighbors=3, metric="euclidean", order=3, weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.order = order
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store the training patterns (kNN is a lazy learner)."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if self.n_neighbors > self.X_train.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of samples")
        return self

    def predict(self, X_test):
        """Classify every row of X_test.

        [GenAI: Claude] one distance-matrix call for the whole test set, then a
        per-row partial selection and vote.
        """
        if self.X_train is None:
            raise RuntimeError("call fit() before predict()")

        matrix = np.asarray(X_test, dtype=float)
        distance_matrix = compute_distance_matrix(
            matrix, self.X_train, self.metric, self.order
        )

        predictions = np.empty(matrix.shape[0], dtype=self.y_train.dtype)
        for row in range(matrix.shape[0]):
            neighbour_indices = select_k_nearest(distance_matrix[row], self.n_neighbors)
            neighbour_labels = self.y_train[neighbour_indices]
            neighbour_distances = distance_matrix[row][neighbour_indices]

            if self.weighted:
                predictions[row] = vote_weighted(neighbour_labels, neighbour_distances)
            else:
                predictions[row] = vote_majority(neighbour_labels, neighbour_distances)

        return predictions

    def score(self, X_test, y_test):
        """Mean accuracy on the supplied test set."""
        return float(np.mean(self.predict(X_test) == np.asarray(y_test)))


def binary_confusion_counts(y_true, y_predicted, positive_label=1):
    """Return (TP, FP, TN, FN) for a binary problem."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_predicted)

    true_positive = int(np.sum((truth == positive_label) & (prediction == positive_label)))
    false_positive = int(np.sum((truth != positive_label) & (prediction == positive_label)))
    true_negative = int(np.sum((truth != positive_label) & (prediction != positive_label)))
    false_negative = int(np.sum((truth == positive_label) & (prediction != positive_label)))
    return true_positive, false_positive, true_negative, false_negative


def compute_all_metrics(y_true, y_predicted, positive_label=1):
    """Accuracy, precision, recall and F-score in a single dictionary.

    [GenAI: Claude] generated the zero-division guards.
    """
    tp, fp, tn, fn = binary_confusion_counts(y_true, y_predicted, positive_label)

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_score = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f_score),
        "confusion": (tp, fp, tn, fn),
    }


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from lab05_knn import (
        CSV_PATH, RANDOM_STATE, TEST_SIZE,
        load_dataset, build_feature_target, preprocess_train_test,
    )

    print("=" * 78)
    print("23CSE301  |  LAB SESSION 06  |  A1 : GenAI-GENERATED kNN")
    print("GenAI tool used : Claude (Anthropic)")
    print("=" * 78)

    dataframe = load_dataset(CSV_PATH)
    X_raw, y = build_feature_target(dataframe, positive_genre="Drama")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, _ = preprocess_train_test(X_train_raw, X_test_raw)
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    print("\n[DATA] train patterns %d | test patterns %d | features %d"
          % (X_train.shape[0], X_test.shape[0], X_train.shape[1]))

    distance_matrix = compute_distance_matrix(X_test, X_train, "euclidean")
    print("\n[VECTORISED DISTANCES] matrix shape : %s" % (distance_matrix.shape,))
    print("      min %.4f | max %.4f | mean %.4f"
          % (distance_matrix.min(), distance_matrix.max(), distance_matrix.mean()))

    nearest = select_k_nearest(distance_matrix[0], 5)
    print("\n[NEIGHBOURS] test pattern 0, k=5")
    for rank, index in enumerate(nearest, start=1):
        print("      rank %d : train index %-4d distance %.4f  class %d"
              % (rank, index, distance_matrix[0][index], y_train_array[index]))

    genai_model = GenAIKNNClassifier(n_neighbors=3, metric="euclidean", weighted=False)
    genai_model.fit(X_train, y_train_array)
    genai_predictions = genai_model.predict(X_test)
    genai_metrics = compute_all_metrics(y_test_array, genai_predictions)

    genai_weighted = GenAIKNNClassifier(n_neighbors=3, weighted=True)
    genai_weighted.fit(X_train, y_train_array)
    weighted_metrics = compute_all_metrics(y_test_array, genai_weighted.predict(X_test))

    print("\n[A1] GenAI kNN (k=3) results")
    print("      %-24s %-10s %-10s %-10s %-10s"
          % ("variant", "accuracy", "precision", "recall", "f1"))
    print("      %-24s %-10.4f %-10.4f %-10.4f %-10.4f"
          % ("majority vote", genai_metrics["accuracy"], genai_metrics["precision"],
             genai_metrics["recall"], genai_metrics["f1"]))
    print("      %-24s %-10.4f %-10.4f %-10.4f %-10.4f"
          % ("distance weighted", weighted_metrics["accuracy"],
             weighted_metrics["precision"], weighted_metrics["recall"],
             weighted_metrics["f1"]))

    tp, fp, tn, fn = genai_metrics["confusion"]
    print("\n      confusion matrix : TP=%d  FP=%d  TN=%d  FN=%d" % (tp, fp, tn, fn))

    from lab05_knn import CustomKNNClassifier

    lab05_model = CustomKNNClassifier(n_neighbors=3, metric="euclidean")
    lab05_model.fit(X_train, y_train_array)
    lab05_predictions = lab05_model.predict(X_test)

    agreement = float(np.mean(lab05_predictions == genai_predictions))
    print("\n[CROSS-CHECK] GenAI version vs Lab-05 hand-written version")
    print("      identical predictions on %.2f%% of the test set" % (100.0 * agreement))

    k_values = list(range(1, 22, 2))
    print("\n[k SWEEP] GenAI implementation")
    print("      %-4s %-12s %-12s" % ("k", "plain", "weighted"))
    for k in k_values:
        plain = GenAIKNNClassifier(n_neighbors=k).fit(X_train, y_train_array)
        weighted = GenAIKNNClassifier(n_neighbors=k, weighted=True).fit(X_train, y_train_array)
        print("      %-4d %-12.4f %-12.4f"
              % (k, plain.score(X_test, y_test_array),
                 weighted.score(X_test, y_test_array)))

    print("\n" + "=" * 78)
    print("LAB 06 A1 COMPLETE -- run lab06_comparison.py for A3, "
          "test_knn_units.py for A2")
    print("=" * 78)
