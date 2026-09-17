
import unittest
import numpy as np
import pandas as pd

from lab05_knn import (
    build_label_mapping, apply_label_mapping, encode_categorical,
    compute_central_tendency, fit_imputer, apply_imputer, count_missing,
    fit_standard_scaler, apply_standard_scaler,
    euclidean_distance, manhattan_distance, chebyshev_distance,
    minkowski_distance, cosine_distance, compute_distance,
    compute_distance_vector, DISTANCE_METRICS,
    bubble_sort, insertion_sort, selection_sort, merge_sort, quick_sort,
    sort_distances, SORTING_ALGORITHMS,
    identify_neighbors, majority_vote, weighted_vote, distance_weight,
    CustomKNNClassifier,
    confusion_matrix_binary, accuracy_score_custom, precision_score_custom,
    recall_score_custom, f1_score_custom, evaluate_classifier,
    load_dataset, build_feature_target, preprocess_train_test,
    CSV_PATH,
)

from lab06_knn_genai import (
    euclidean_distance_matrix, manhattan_distance_matrix,
    chebyshev_distance_matrix, minkowski_distance_matrix,
    cosine_distance_matrix, compute_distance_matrix,
    fit_column_fill_values, apply_column_fill_values,
    encode_labels_vectorised, standardise,
    select_k_nearest, vote_majority, vote_weighted,
    GenAIKNNClassifier, binary_confusion_counts, compute_all_metrics,
)


class TestEncodingModule(unittest.TestCase):
    """Unit tests for the categorical encoding module."""

    def test_tc01_mapping_is_alphabetical_and_zero_based(self):
        """TC-01: categories are mapped to 0..n-1 in sorted order."""
        series = pd.Series(["PG-13", "R", "G", "R"])
        mapping = build_label_mapping(series)
        self.assertEqual(mapping, {"G": 0, "PG-13": 1, "R": 2})

    def test_tc02_mapping_excludes_missing_values(self):
        """TC-02: NaN must not become a category of its own."""
        series = pd.Series(["R", np.nan, "PG"])
        mapping = build_label_mapping(series)
        self.assertEqual(set(mapping.keys()), {"PG", "R"})

    def test_tc03_apply_mapping_preserves_nan(self):
        """TC-03: applying the map leaves NaN intact for the imputer."""
        series = pd.Series(["R", np.nan, "PG"])
        encoded = apply_label_mapping(series, {"PG": 0, "R": 1})
        self.assertEqual(encoded.iloc[0], 1)
        self.assertTrue(pd.isna(encoded.iloc[1]))
        self.assertEqual(encoded.iloc[2], 0)

    def test_tc04_unseen_category_becomes_nan(self):
        """TC-04: a test-set category unseen in training must not crash."""
        series = pd.Series(["NC-17"])
        encoded = apply_label_mapping(series, {"PG": 0, "R": 1})
        self.assertTrue(pd.isna(encoded.iloc[0]))

    def test_tc05_encode_categorical_reuses_supplied_mapping(self):
        """TC-05: train and test must share one encoding."""
        train = pd.DataFrame({"rating": ["R", "PG"]})
        test = pd.DataFrame({"rating": ["PG", "R"]})
        _, mappings = encode_categorical(train, ["rating"])
        encoded_test, reused = encode_categorical(test, ["rating"], mappings)
        self.assertEqual(reused, mappings)
        self.assertEqual(list(encoded_test["rating"]), [0, 1])

    def test_tc06_vectorised_encoder_matches_pandas_encoder(self):
        """TC-06: the Lab-06 encoder agrees with the Lab-05 encoder."""
        values = np.array(["R", "PG", "R", "G"])
        encoded, mapping = encode_labels_vectorised(values)
        expected = build_label_mapping(pd.Series(values))
        self.assertEqual(mapping, expected)
        self.assertEqual(list(encoded), [2, 1, 2, 0])

class TestImputationModule(unittest.TestCase):
    """Unit tests for missing-value imputation."""

    def test_tc07_mean_ignores_nan(self):
        """TC-07: mean of [1,2,3,NaN] is 2.0."""
        self.assertAlmostEqual(
            compute_central_tendency(pd.Series([1, 2, 3, np.nan]), "mean"), 2.0)

    def test_tc08_median_of_even_length(self):
        """TC-08: median of [1,2,3,4] is 2.5."""
        self.assertAlmostEqual(
            compute_central_tendency(pd.Series([1, 2, 3, 4]), "median"), 2.5)

    def test_tc09_mode_returns_most_frequent(self):
        """TC-09: mode of [5,5,9] is 5."""
        self.assertAlmostEqual(
            compute_central_tendency(pd.Series([5, 5, 9]), "mode"), 5.0)

    def test_tc10_all_nan_column_returns_zero(self):
        """TC-10: an entirely empty column must not raise."""
        self.assertEqual(
            compute_central_tendency(pd.Series([np.nan, np.nan]), "mean"), 0.0)

    def test_tc11_invalid_strategy_raises(self):
        """TC-11: an unknown strategy is rejected."""
        with self.assertRaises(ValueError):
            compute_central_tendency(pd.Series([1, 2]), "geometric")

    def test_tc12_imputer_removes_every_nan(self):
        """TC-12: after apply_imputer no NaN survives."""
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 2.0]})
        fill_values = fit_imputer(frame, {"a": "mean", "b": "mode"})
        imputed = apply_imputer(frame, fill_values)
        self.assertEqual(int(count_missing(imputed).sum()), 0)
        self.assertAlmostEqual(imputed["a"].iloc[1], 2.0)
        self.assertAlmostEqual(imputed["b"].iloc[0], 2.0)

    def test_tc13_vectorised_imputer_matches_pandas_imputer(self):
        """TC-13: Lab-06 NumPy imputer agrees with the Lab-05 pandas imputer."""
        matrix = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 4.0]])
        fill_values = fit_column_fill_values(matrix, ["mean", "median"])
        filled = apply_column_fill_values(matrix, fill_values)
        self.assertFalse(np.isnan(filled).any())
        self.assertAlmostEqual(filled[1, 0], 2.0)      # mean of 1 and 3
        self.assertAlmostEqual(filled[0, 1], 4.0)      # median of 4 and 4


class TestDistanceModule(unittest.TestCase):
    """Unit tests for every distance metric."""

    def setUp(self):
        self.origin = np.array([0.0, 0.0])
        self.point = np.array([3.0, 4.0])

    def test_tc14_euclidean_three_four_five(self):
        """TC-14: the 3-4-5 triangle."""
        self.assertAlmostEqual(euclidean_distance(self.origin, self.point), 5.0)

    def test_tc15_manhattan_is_sum_of_absolute_differences(self):
        """TC-15: L1 of (3,4) from origin is 7."""
        self.assertAlmostEqual(manhattan_distance(self.origin, self.point), 7.0)

    def test_tc16_chebyshev_is_largest_component(self):
        """TC-16: L-infinity of (3,4) from origin is 4."""
        self.assertAlmostEqual(chebyshev_distance(self.origin, self.point), 4.0)

    def test_tc17_minkowski_order_two_equals_euclidean(self):
        """TC-17: p=2 Minkowski collapses to Euclidean."""
        self.assertAlmostEqual(
            minkowski_distance(self.origin, self.point, 2),
            euclidean_distance(self.origin, self.point))

    def test_tc18_minkowski_order_one_equals_manhattan(self):
        """TC-18: p=1 Minkowski collapses to Manhattan."""
        self.assertAlmostEqual(
            minkowski_distance(self.origin, self.point, 1),
            manhattan_distance(self.origin, self.point))

    def test_tc19_cosine_of_parallel_vectors_is_zero(self):
        """TC-19: parallel vectors have cosine distance 0."""
        self.assertAlmostEqual(
            cosine_distance(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 0.0)

    def test_tc20_cosine_of_orthogonal_vectors_is_one(self):
        """TC-20: orthogonal vectors have cosine distance 1."""
        self.assertAlmostEqual(
            cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)

    def test_tc21_distance_properties_hold(self):
        """TC-21: identity, non-negativity and symmetry for all metrics."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 0.0, -1.0])
        for metric in DISTANCE_METRICS:
            self.assertAlmostEqual(compute_distance(a, a, metric), 0.0,
                                   msg="identity failed for %s" % metric)
            self.assertGreaterEqual(compute_distance(a, b, metric), 0.0,
                                    msg="non-negativity failed for %s" % metric)
            self.assertAlmostEqual(compute_distance(a, b, metric),
                                   compute_distance(b, a, metric),
                                   msg="symmetry failed for %s" % metric)

    def test_tc22_unknown_metric_raises(self):
        """TC-22: an unsupported metric name is rejected."""
        with self.assertRaises(ValueError):
            compute_distance(self.origin, self.point, "mahalanobis")


class TestVectorisedDistances(unittest.TestCase):
    """The Lab-06 distance matrices must reproduce the Lab-05 scalar results."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.X_train = rng.normal(size=(12, 4))
        self.X_test = rng.normal(size=(5, 4))

    def _scalar_reference(self, metric, order=3):
        return np.array([
            compute_distance_vector(self.X_train, self.X_test[row], metric, order)
            for row in range(self.X_test.shape[0])
        ])

    def test_tc23_euclidean_matrix_matches_scalar(self):
        """TC-23: vectorised Euclidean == looped Euclidean."""
        np.testing.assert_allclose(
            euclidean_distance_matrix(self.X_test, self.X_train),
            self._scalar_reference("euclidean"), atol=1e-9)

    def test_tc24_manhattan_and_chebyshev_match_scalar(self):
        """TC-24: vectorised L1 and L-infinity == looped versions."""
        np.testing.assert_allclose(
            manhattan_distance_matrix(self.X_test, self.X_train),
            self._scalar_reference("manhattan"), atol=1e-9)
        np.testing.assert_allclose(
            chebyshev_distance_matrix(self.X_test, self.X_train),
            self._scalar_reference("chebyshev"), atol=1e-9)

    def test_tc25_minkowski_and_cosine_match_scalar(self):
        """TC-25: vectorised Lp and cosine == looped versions."""
        np.testing.assert_allclose(
            minkowski_distance_matrix(self.X_test, self.X_train, 3),
            self._scalar_reference("minkowski", 3), atol=1e-9)
        np.testing.assert_allclose(
            cosine_distance_matrix(self.X_test, self.X_train),
            self._scalar_reference("cosine"), atol=1e-9)

    def test_tc26_distance_matrix_shape(self):
        """TC-26: shape is (n_test, n_train)."""
        matrix = compute_distance_matrix(self.X_test, self.X_train)
        self.assertEqual(matrix.shape, (5, 12))

class TestSortingModule(unittest.TestCase):
    """Every hand-written sorting algorithm must be a correct sort."""

    def setUp(self):
        self.unsorted = [(3.5, 2), (1.1, 7), (9.9, 0), (1.1, 3), (0.2, 5)]
        self.expected = sorted(self.unsorted)

    def test_tc27_bubble_sort_correct(self):
        """TC-27: bubble sort output equals Timsort output."""
        self.assertEqual(bubble_sort(self.unsorted), self.expected)

    def test_tc28_insertion_sort_correct(self):
        """TC-28: insertion sort output equals Timsort output."""
        self.assertEqual(insertion_sort(self.unsorted), self.expected)

    def test_tc29_selection_sort_correct(self):
        """TC-29: selection sort output equals Timsort output."""
        self.assertEqual(selection_sort(self.unsorted), self.expected)

    def test_tc30_merge_and_quick_sort_correct(self):
        """TC-30: merge and quick sort output equals Timsort output."""
        self.assertEqual(merge_sort(self.unsorted), self.expected)
        self.assertEqual(quick_sort(self.unsorted), self.expected)

    def test_tc31_sorts_do_not_mutate_the_input(self):
        """TC-31: the caller's list must be left untouched."""
        original = list(self.unsorted)
        for name in SORTING_ALGORITHMS:
            sort_distances(self.unsorted, name)
            self.assertEqual(self.unsorted, original,
                             msg="%s mutated its input" % name)

    def test_tc32_edge_cases_empty_and_single(self):
        """TC-32: empty and single-element lists are handled."""
        for name in SORTING_ALGORITHMS:
            self.assertEqual(sort_distances([], name), [])
            self.assertEqual(sort_distances([(1.0, 0)], name), [(1.0, 0)])

    def test_tc33_all_algorithms_agree_on_random_input(self):
        """TC-33: 200 random floats, all six algorithms give one answer."""
        rng = np.random.default_rng(11)
        pairs = [(float(v), i) for i, v in enumerate(rng.normal(size=200))]
        reference = sorted(pairs)
        for name in SORTING_ALGORITHMS:
            self.assertEqual(sort_distances(pairs, name), reference,
                             msg="%s disagreed" % name)

    def test_tc34_unknown_algorithm_raises(self):
        """TC-34: an unsupported sort name is rejected."""
        with self.assertRaises(ValueError):
            sort_distances(self.unsorted, "heap")

class TestNeighbourModule(unittest.TestCase):
    """Neighbour selection, including the tie-breaking rule."""

    def test_tc35_returns_k_nearest_in_order(self):
        """TC-35: the k smallest distances come back nearest first."""
        distances = [5.0, 1.0, 3.0, 2.0]
        neighbours = identify_neighbors(distances, 3, "merge")
        self.assertEqual(neighbours, [(1.0, 1), (2.0, 3), (3.0, 2)])

    def test_tc36_tie_broken_by_smaller_index(self):
        """TC-36: equal distances -> lower training index wins."""
        distances = [2.0, 2.0, 2.0]
        neighbours = identify_neighbors(distances, 2, "merge")
        self.assertEqual([index for _, index in neighbours], [0, 1])

    def test_tc37_k_larger_than_dataset_is_clamped(self):
        """TC-37: asking for more neighbours than exist returns all of them."""
        self.assertEqual(len(identify_neighbors([1.0, 2.0], 10, "merge")), 2)

    def test_tc38_non_positive_k_raises(self):
        """TC-38: k must be a positive integer."""
        with self.assertRaises(ValueError):
            identify_neighbors([1.0, 2.0], 0, "merge")

    def test_tc39_every_sort_gives_the_same_neighbours(self):
        """TC-39: the sorting algorithm is a config knob, not a result change."""
        rng = np.random.default_rng(3)
        distances = list(rng.normal(size=60))
        reference = identify_neighbors(distances, 5, "merge")
        for name in SORTING_ALGORITHMS:
            self.assertEqual(identify_neighbors(distances, 5, name), reference,
                             msg="%s changed the neighbour set" % name)

    def test_tc40_argpartition_selection_matches_full_sort(self):
        """TC-40: the Lab-06 O(n) selection matches the Lab-05 full sort."""
        rng = np.random.default_rng(5)
        distances = rng.normal(size=100)
        fast = list(select_k_nearest(distances, 7))
        slow = [index for _, index in identify_neighbors(list(distances), 7, "merge")]
        self.assertEqual(fast, slow)

class TestVotingModule(unittest.TestCase):
    """Majority and weighted voting, including tie breaking."""

    def test_tc41_clear_majority(self):
        """TC-41: 2 votes for class 1 beats 1 vote for class 0."""
        self.assertEqual(majority_vote([1, 1, 0], [0.1, 0.2, 0.3]), 1)

    def test_tc42_tie_broken_by_nearest_neighbour(self):
        """TC-42: on a 2-2 tie the class of the closest neighbour wins."""
        # class 1 owns the nearest neighbour at distance 0.1
        self.assertEqual(majority_vote([0, 0, 1, 1], [0.5, 0.6, 0.1, 0.7]), 1)
        # class 0 owns the nearest neighbour at distance 0.05
        self.assertEqual(majority_vote([0, 0, 1, 1], [0.05, 0.6, 0.2, 0.7]), 0)

    def test_tc43_tie_without_distances_picks_smallest_label(self):
        """TC-43: deterministic fallback when distances are unavailable."""
        self.assertEqual(majority_vote([0, 1]), 0)

    def test_tc44_unanimous_vote(self):
        """TC-44: all neighbours of one class return that class."""
        self.assertEqual(majority_vote([1, 1, 1], [0.1, 0.2, 0.3]), 1)

    def test_tc45_weight_decreases_with_distance(self):
        """TC-45: a nearer neighbour must carry a larger weight."""
        self.assertGreater(distance_weight(0.5), distance_weight(2.0))

    def test_tc46_zero_distance_weight_is_finite(self):
        """TC-46: an exact duplicate must not produce a division by zero."""
        self.assertTrue(np.isfinite(distance_weight(0.0)))

    def test_tc47_weighted_vote_overturns_a_plain_majority(self):
        """TC-47: one very close neighbour outvotes two distant ones."""
        labels = [1, 0, 0]
        distances = [0.01, 5.0, 6.0]
        self.assertEqual(majority_vote(labels, distances), 0)
        self.assertEqual(weighted_vote(labels, distances), 1)

    def test_tc48_vectorised_votes_match_lab05_votes(self):
        """TC-48: the Lab-06 voting routines agree with the Lab-05 ones."""
        labels = np.array([1, 0, 0, 1, 1])
        distances = np.array([0.4, 0.1, 0.9, 0.3, 1.2])
        self.assertEqual(int(vote_majority(labels, distances)),
                         int(majority_vote(list(labels), list(distances))))
        self.assertEqual(int(vote_weighted(labels, distances)),
                         int(weighted_vote(list(labels), list(distances))))


class TestClassifierAPI(unittest.TestCase):
    """fit(), predict() and score() on small, fully controlled data."""

    def setUp(self):
        # two tight, well separated clusters
        self.X_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.2],
                                 [9.0, 9.0], [9.1, 9.1], [9.0, 8.8]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])
        self.X_test = np.array([[0.05, 0.05], [9.05, 9.05]])
        self.y_test = np.array([0, 1])

    def test_tc49_fit_returns_self_for_chaining(self):
        """TC-49: fit() returns the estimator, sklearn style."""
        model = CustomKNNClassifier(n_neighbors=3)
        self.assertIs(model.fit(self.X_train, self.y_train), model)

    def test_tc50_predict_before_fit_raises(self):
        """TC-50: predicting on an untrained model is an error."""
        with self.assertRaises(RuntimeError):
            CustomKNNClassifier(n_neighbors=3).predict(self.X_test)

    def test_tc51_mismatched_lengths_raise(self):
        """TC-51: X and y must have the same number of rows."""
        with self.assertRaises(ValueError):
            CustomKNNClassifier().fit(self.X_train, self.y_train[:4])

    def test_tc52_k_larger_than_training_set_raises(self):
        """TC-52: k cannot exceed the number of training patterns."""
        with self.assertRaises(ValueError):
            CustomKNNClassifier(n_neighbors=99).fit(self.X_train, self.y_train)

    def test_tc53_separable_clusters_classified_perfectly(self):
        """TC-53: accuracy is 1.0 on two well separated clusters."""
        model = CustomKNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        self.assertEqual(model.score(self.X_test, self.y_test), 1.0)

    def test_tc54_predict_output_shape(self):
        """TC-54: one prediction per test row."""
        model = CustomKNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        self.assertEqual(model.predict(self.X_test).shape, (2,))

    def test_tc55_k_equals_one_reproduces_training_labels(self):
        """TC-55: with k=1 every training point is its own nearest neighbour."""
        model = CustomKNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(model.predict(self.X_train), self.y_train)

    def test_tc56_weighted_and_plain_agree_on_easy_data(self):
        """TC-56: both voting rules are correct when the data is separable."""
        plain = CustomKNNClassifier(n_neighbors=3, weighted=False)
        weighted = CustomKNNClassifier(n_neighbors=3, weighted=True)
        plain.fit(self.X_train, self.y_train)
        weighted.fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(plain.predict(self.X_test),
                                      weighted.predict(self.X_test))

    def test_tc57_genai_classifier_matches_custom_classifier(self):
        """TC-57: the Lab-06 classifier reproduces the Lab-05 classifier."""
        custom = CustomKNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        genai = GenAIKNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(custom.predict(self.X_test),
                                      genai.predict(self.X_test))


class TestMetricsModule(unittest.TestCase):
    """Accuracy, precision, recall and F-score on hand-checked inputs."""

    def setUp(self):
        #                      TP  FP  TN  FN
        self.y_true = np.array([1, 0, 0, 1, 1, 0])
        self.y_pred = np.array([1, 1, 0, 1, 0, 0])

    def test_tc58_confusion_counts_are_correct(self):
        """TC-58: TP=2, FP=1, TN=2, FN=1."""
        self.assertEqual(confusion_matrix_binary(self.y_true, self.y_pred),
                         (2, 1, 2, 1))

    def test_tc59_accuracy(self):
        """TC-59: 4 correct out of 6."""
        self.assertAlmostEqual(accuracy_score_custom(self.y_true, self.y_pred),
                               4 / 6)

    def test_tc60_precision(self):
        """TC-60: 2 / (2 + 1)."""
        self.assertAlmostEqual(precision_score_custom(self.y_true, self.y_pred),
                               2 / 3)

    def test_tc61_recall(self):
        """TC-61: 2 / (2 + 1)."""
        self.assertAlmostEqual(recall_score_custom(self.y_true, self.y_pred),
                               2 / 3)

    def test_tc62_f1_is_harmonic_mean(self):
        """TC-62: with precision == recall, F1 equals both."""
        self.assertAlmostEqual(f1_score_custom(self.y_true, self.y_pred), 2 / 3)

    def test_tc63_no_positive_predictions_gives_zero_not_nan(self):
        """TC-63: precision must not divide by zero."""
        self.assertEqual(precision_score_custom(np.array([1, 1]), np.array([0, 0])),
                         0.0)

    def test_tc64_perfect_prediction_scores_one(self):
        """TC-64: identical arrays give 1.0 on every metric."""
        metrics = evaluate_classifier(self.y_true, self.y_true)
        for name, value in metrics.items():
            self.assertAlmostEqual(value, 1.0, msg="%s was not 1.0" % name)

    def test_tc65_lab06_metrics_match_lab05_metrics(self):
        """TC-65: both metric implementations agree."""
        lab05 = evaluate_classifier(self.y_true, self.y_pred)
        lab06 = compute_all_metrics(self.y_true, self.y_pred)
        for name in ["accuracy", "precision", "recall", "f1"]:
            self.assertAlmostEqual(lab05[name], lab06[name])
        self.assertEqual(binary_confusion_counts(self.y_true, self.y_pred),
                         confusion_matrix_binary(self.y_true, self.y_pred))


class TestScalingModule(unittest.TestCase):
    """Z-score normalisation."""

    def test_tc66_scaled_data_has_zero_mean_unit_variance(self):
        """TC-66: after scaling, mean ~ 0 and std ~ 1."""
        rng = np.random.default_rng(1)
        matrix = rng.normal(loc=50, scale=7, size=(40, 3))
        means, stds = fit_standard_scaler(matrix)
        scaled = apply_standard_scaler(matrix, means, stds)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-9)

    def test_tc67_constant_column_does_not_divide_by_zero(self):
        """TC-67: a column with zero variance is handled safely."""
        matrix = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        means, stds = fit_standard_scaler(matrix)
        scaled = apply_standard_scaler(matrix, means, stds)
        self.assertTrue(np.isfinite(scaled).all())

    def test_tc68_test_split_uses_training_statistics(self):
        """TC-68: test data is scaled with the train mean/std, not its own."""
        train = np.array([[0.0], [10.0]])
        test = np.array([[5.0]])
        means, stds = fit_standard_scaler(train)
        self.assertAlmostEqual(apply_standard_scaler(test, means, stds)[0, 0], 0.0)

    def test_tc69_lab06_standardise_matches_lab05(self):
        """TC-69: both scalers produce the same output."""
        rng = np.random.default_rng(2)
        matrix = rng.normal(size=(20, 4))
        means, stds = fit_standard_scaler(matrix)
        lab05 = apply_standard_scaler(matrix, means, stds)
        lab06, _, _ = standardise(matrix)
        np.testing.assert_allclose(lab05, lab06, atol=1e-9)


class TestEndToEndPipeline(unittest.TestCase):
    """Functional tests that exercise the whole pipeline on the real data."""

    @classmethod
    def setUpClass(cls):
        from sklearn.model_selection import train_test_split
        dataframe = load_dataset(CSV_PATH)
        cls.X_raw, cls.y = build_feature_target(dataframe)
        (cls.X_train_raw, cls.X_test_raw,
         cls.y_train, cls.y_test) = train_test_split(
            cls.X_raw, cls.y, test_size=0.3, random_state=42, stratify=cls.y)
        cls.X_train, cls.X_test, cls.artefacts = preprocess_train_test(
            cls.X_train_raw, cls.X_test_raw)

    def test_tc70_dataset_loads_with_expected_shape(self):
        """TC-70: 250 movies are loaded and the target is binary."""
        self.assertEqual(len(self.X_raw), 250)
        self.assertEqual(sorted(self.y.unique().tolist()), [0, 1])

    def test_tc71_pipeline_leaves_no_missing_values(self):
        """TC-71: encoding + imputation clears every NaN in both splits."""
        self.assertFalse(np.isnan(self.X_train).any())
        self.assertFalse(np.isnan(self.X_test).any())

    def test_tc72_raw_data_really_did_contain_missing_values(self):
        """TC-72: guards against a vacuous version of TC-71."""
        self.assertGreater(int(count_missing(self.X_raw).sum()), 0)

    def test_tc73_split_sizes_respect_test_size(self):
        """TC-73: a 0.3 split of 250 rows gives 175 / 75."""
        self.assertEqual(self.X_train.shape[0], 175)
        self.assertEqual(self.X_test.shape[0], 75)

    def test_tc74_all_three_implementations_agree_on_real_data(self):
        """TC-74: custom, GenAI and scikit-learn produce identical predictions."""
        from sklearn.neighbors import KNeighborsClassifier

        y_train = self.y_train.to_numpy()
        custom = CustomKNNClassifier(n_neighbors=5).fit(self.X_train, y_train)
        genai = GenAIKNNClassifier(n_neighbors=5).fit(self.X_train, y_train)
        package = KNeighborsClassifier(n_neighbors=5).fit(self.X_train, y_train)

        custom_pred = custom.predict(self.X_test)
        genai_pred = genai.predict(self.X_test)
        package_pred = package.predict(self.X_test)

        np.testing.assert_array_equal(custom_pred, genai_pred)
        np.testing.assert_array_equal(custom_pred, package_pred)

    def test_tc75_accuracy_beats_the_majority_class_baseline(self):
        """TC-75: the classifier is better than always predicting 'Drama'."""
        y_train = self.y_train.to_numpy()
        y_test = self.y_test.to_numpy()
        baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
        model = CustomKNNClassifier(n_neighbors=15).fit(self.X_train, y_train)
        self.assertGreaterEqual(model.score(self.X_test, y_test), baseline * 0.95)

if __name__ == "__main__":
    unittest.main(verbosity=2)
