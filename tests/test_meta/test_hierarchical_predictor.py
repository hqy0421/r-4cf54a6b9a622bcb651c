from contextlib import nullcontext as does_not_raise
from random import randint

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sklearn import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_compat.utils.estimator_checks import parametrize_with_checks

from sklego.meta import HierarchicalClassifier, HierarchicalRegressor
from sklego.meta.hierarchical_predictor import _ConstantClassifier
from tests.conftest import DATAFRAME_ARRAY_API_REASON, expect_array_api_failure

frame_funcs = [pd.DataFrame, pl.DataFrame, pa.table]


@parametrize_with_checks(
    [HierarchicalRegressor(estimator=LinearRegression(), groups=0)],
    expected_failed_checks=expect_array_api_failure(DATAFRAME_ARRAY_API_REASON),
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_no_attributes_set_in_init",  # Setting **shrinkage_kwargs in init
        "check_estimators_pickle",  # Fails when input contains NaN
        "check_regressor_data_not_an_array",  # DataFrame constructor not properly called!  TODO: This should work
        "check_dtype_object",  # custom message
        "check_fit2d_1feature",  # custom message
        "check_supervised_y_2d",  # TODO: Is it possible to support multioutput?
        "check_estimators_empty_data_messages",  # custom message
        "check_requires_y_none",
        "check_n_features_in_after_fitting",  # custom check
    }:
        pytest.skip()

    check(estimator)


def make_hierarchical_dataset(task, frame_func=pd.DataFrame):
    n_samples, n_features, n_informative, random_state = 1000, 10, 3, 42
    if task == "binary-classification":
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
        )

    elif task == "multiclass-classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=random_state,
            n_classes=4,
        )

    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
        )

    else:
        raise ValueError("Invalid task")

    X_ = frame_func(
        pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
        .assign(
            g_0=1,
            g_1=["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
            g_2=["X"] * (n_samples // 4) + ["Y"] * (n_samples // 2) + ["Z"] * (n_samples // 4),
        )
        .to_dict(orient="list")
    )
    groups = ["g_0", "g_1", "g_2"]

    return X_, y, groups


def make_hierarchical_dummy(frame_func):
    df_train = frame_func(
        {
            "x": np.ones(1000),
            "g_1": ["A"] * 500 + ["B"] * 500,
            "g_2": ["X"] * 250 + ["Y"] * 500 + ["Z"] * 250,
            "target": [0] * 250 + [1] * 500 + [0] * 250,
        }
    )
    # -> will fit the following values: (g_1, g_2) in {(A,X), (A, Y), (B, Y), (B, Z)} and g_1 in {A, B}

    df_pred = frame_func({"x": [1] * 7, "g_1": ["A"] * 3 + ["B"] * 3 + ["C"], "g_2": list("XYZ") * 2 + ["X"]})

    # The following fallbacks are expected:
    # ("A", "Z") -> to estimator for g_1 = A
    # ("B", "X") -> to estimator for g_1 = B
    # ("C", "X") -> to global estimator

    return nw.from_native(df_train), nw.from_native(df_pred)


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task",
    [
        (HierarchicalRegressor, LinearRegression(), "regression"),
        (HierarchicalRegressor, Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), "regression"),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification"),
        (
            HierarchicalClassifier,
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
            "binary-classification",
        ),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification"),
        (
            HierarchicalClassifier,
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
            "multiclass-classification",
        ),
    ],
)
@pytest.mark.parametrize("fallback_method", ["raise", "parent"])
@pytest.mark.parametrize(
    ("shrinkage", "kwargs"),
    [
        (None, None),
        ("equal", None),
        ("relative", None),
        ("min_n_obs", {"min_n_obs": 10}),
        ("constant", {"alpha": 0.5}),
    ],
)
def test_fit_predict(meta_cls, base_estimator, task, fallback_method, shrinkage, kwargs):
    """Tests that the model can be fit and predict with different configurations of fallback and shrinkage methods if
    X to predict contains same groups as X used to fit.
    """
    X, y, groups = make_hierarchical_dataset(task, frame_func=frame_funcs[randint(0, 2)])

    meta_model = meta_cls(
        estimator=base_estimator,
        groups=groups,
        fallback_method=fallback_method,
        shrinkage=shrinkage,
        shrinkage_kwargs=kwargs,
    ).fit(X, y)

    assert meta_model.estimators_ is not None
    assert meta_model.predict(X) is not None

    if task in {"binary-classification", "multiclass-classification"}:
        assert meta_model.predict_proba(X) is not None


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task",
    [
        (HierarchicalRegressor, LinearRegression(), "regression"),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification"),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification"),
    ],
)
@pytest.mark.parametrize("fallback_method,context", [("raise", pytest.raises(KeyError)), ("parent", does_not_raise())])
def test_fallback(meta_cls, base_estimator, task, fallback_method, context):
    """Tests that the model fails or not when predicting with different fallback methods if X to predict contains
    unseen group values.
    """
    X, y, groups = make_hierarchical_dataset(task, frame_func=frame_funcs[randint(0, 2)])

    meta_model = meta_cls(estimator=base_estimator, groups=groups, fallback_method=fallback_method).fit(X, y)
    X = nw.to_native(nw.from_native(X).with_columns(**{g: nw.lit(-1) for g in groups}))

    with context:
        meta_model.predict(X)


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task,metric",
    [
        (HierarchicalRegressor, LinearRegression(), "regression", r2_score),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification", accuracy_score),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification", accuracy_score),
    ],
)
@pytest.mark.parametrize(
    ("shrinkage", "kwargs"),
    [
        (None, None),
        ("equal", None),
        ("relative", None),
        ("min_n_obs", {"min_n_obs": 10}),
        ("constant", {"alpha": 0.5}),
    ],
)
def test_shrinkage(meta_cls, base_estimator, task, metric, shrinkage, kwargs):
    """Tests that the model performance is better than the base estimator when predicting with different shrinkage
    methods.
    """
    X, y, groups = make_hierarchical_dataset(task, frame_func=frame_funcs[randint(0, 2)])

    X_ = nw.from_native(X).drop(groups).pipe(nw.to_native)
    meta_model = meta_cls(
        estimator=clone(base_estimator), groups=groups, shrinkage=shrinkage, shrinkage_kwargs=kwargs
    ).fit(X, y)
    base_model = clone(base_estimator).fit(X_, y)

    assert metric(y, base_model.predict(X_)) <= metric(y, meta_model.predict(X))


@pytest.mark.parametrize(
    "meta_model,method",
    [
        (HierarchicalRegressor(DummyRegressor(strategy="mean"), groups=["g_1", "g_2"], check_X=False), "predict"),
        (
            HierarchicalClassifier(DummyClassifier(strategy="prior"), groups=["g_1", "g_2"], check_X=False),
            "predict_proba",
        ),
    ],
)
@pytest.mark.parametrize(
    "shrinkage,expected",
    [
        # most granual prediction is the class/target itself, fallback to parent or global is the average
        (None, [0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5]),
        # prediction is same as global, which is always 0.5
        (lambda x: np.array([1, 0, 0]), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        # prediction is average of global and most granual i.e. the previous two cases
        (lambda x: np.array([1, 0, 1]), [0.25, 0.75, 0.5, 0.5, 0.75, 0.25, 0.5]),
    ],
)
@pytest.mark.parametrize("frame_func", frame_funcs)
def test_expected_output(meta_model, method, shrinkage, expected, frame_func):
    df_train, df_test = make_hierarchical_dummy(frame_func)

    X_train, y_train = df_train.select("x", "g_1", "g_2"), df_train["target"]
    X_test = df_test.select("x", "g_1", "g_2")
    meta_model.set_params(shrinkage=shrinkage).fit(nw.to_native(X_train), nw.to_native(y_train))

    select_pred = lambda x: x[:, 1] if x.ndim > 1 else x
    y_pred = select_pred(getattr(meta_model, method)(nw.to_native(X_test)))

    assert np.allclose(expected, y_pred)


def make_partial_labels_dataset(frame_func=pd.DataFrame, labels=("a", "b", "c")):
    """Dataset where the most granular groups do not all see the same labels.

    - store S1 only ever sees `labels[0]` (single-class group)
    - store S2 sees `labels[0]` and `labels[1]`
    - store S3 sees `labels[0]` and `labels[1]`
    - store S4 only ever sees `labels[2]` (single-class group)
    """
    n_per_store = 30
    rng = np.random.default_rng(42)
    target = (
        [labels[0]] * n_per_store
        + [labels[0]] * (n_per_store // 2)
        + [labels[1]] * (n_per_store // 2)
        + [labels[0]] * (n_per_store // 2)
        + [labels[1]] * (n_per_store // 2)
        + [labels[2]] * n_per_store
    )
    n = len(target)
    data = {
        "x_0": rng.normal(size=n),
        "x_1": rng.normal(size=n),
        "region": ["R1"] * (2 * n_per_store) + ["R2"] * (2 * n_per_store),
        "store": ["S1"] * n_per_store + ["S2"] * n_per_store + ["S3"] * n_per_store + ["S4"] * n_per_store,
    }
    return frame_func(data), np.array(target), ["region", "store"]


@pytest.mark.parametrize("frame_func", frame_funcs)
@pytest.mark.parametrize(
    "labels",
    [
        ("a", "b", "c"),  # string labels
        (3, 8, 15),  # discontinuous integer labels, none of them a valid positional index
    ],
)
@pytest.mark.parametrize("shrinkage", [None, "equal", "relative", "constant"])
def test_classifier_partial_and_single_class_groups(frame_func, labels, shrinkage):
    """Local models participate with their real class meanings in fallback/shrinkage, for any label type."""
    X, y, groups = make_partial_labels_dataset(frame_func, labels)
    model = HierarchicalClassifier(
        estimator=LogisticRegression(max_iter=300),
        groups=groups,
        shrinkage=shrinkage,
        shrinkage_kwargs={"alpha": 0.5} if shrinkage == "constant" else None,
    ).fit(X, y)

    global_classes = np.unique(y)
    assert np.array_equal(model.classes_, global_classes)

    proba = model.predict_proba(X)

    # predict_proba always returns one column per global class, in the global `classes_` order
    assert proba.shape == (len(y), len(global_classes))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()

    # single-class groups hold a deterministic estimator on the real label, not fabricated samples
    s1_estimator = model.estimators_[(1, "R1", "S1")]
    s4_estimator = model.estimators_[(1, "R2", "S4")]
    assert isinstance(s1_estimator, _ConstantClassifier)
    assert isinstance(s4_estimator, _ConstantClassifier)
    assert np.array_equal(s1_estimator.classes_, np.array([labels[0]]))
    assert np.array_equal(s4_estimator.classes_, np.array([labels[2]]))

    # without shrinkage, single-class leaves give a degenerate distribution on their single class
    if shrinkage is None:
        s1_rows = np.arange(len(y)) < 30
        s4_rows = np.arange(len(y)) >= 90
        assert np.allclose(proba[np.ix_(s1_rows, [0])], 1.0)
        assert np.allclose(proba[np.ix_(s1_rows, [1, 2])], 0.0)
        assert np.allclose(proba[np.ix_(s4_rows, [2])], 1.0)
        assert np.allclose(proba[np.ix_(s4_rows, [0, 1])], 0.0)

    # `predict` is the argmax of the blended probabilities and returns labels from the global `classes_`
    preds = model.predict(X)
    assert np.array_equal(preds, global_classes[np.argmax(proba, axis=1)])
    assert set(np.unique(preds)).issubset(set(global_classes))


@pytest.mark.parametrize("frame_func", frame_funcs)
def test_predict_proba_unseen_local_classes_are_zero(frame_func):
    """A class never seen by a local group contributes zero to that group's probability row."""
    X, y, groups = make_partial_labels_dataset(frame_func)
    model = HierarchicalClassifier(estimator=LogisticRegression(max_iter=300), groups=groups, shrinkage=None).fit(X, y)

    proba = model.predict_proba(X)
    class_ix = {label: ix for ix, label in enumerate(model.classes_)}

    # S2 never sees 'c': its leaf-level prediction must put zero mass on the 'c' column
    s2_rows = np.arange(30, 60)
    assert np.allclose(proba[s2_rows, class_ix["c"]], 0.0)
    # the remaining mass is distributed over the classes S2 actually saw ('a', 'b')
    assert np.allclose(proba[np.ix_(s2_rows, [class_ix["a"], class_ix["b"]])].sum(axis=1), 1.0)


@pytest.mark.parametrize("frame_func", frame_funcs)
def test_stable_output_shape_and_row_order(frame_func):
    """Querying one or many rows keeps the same number of columns and the input row order."""
    X, y, groups = make_partial_labels_dataset(frame_func)
    model = HierarchicalClassifier(LogisticRegression(max_iter=300), groups=groups).fit(X, y)

    full_proba = model.predict_proba(X)
    full_pred = model.predict(X)
    n_classes = len(model.classes_)

    # single-row query keeps a 2D probability matrix and 1D label array
    single_row = nw.to_native(nw.from_native(X).head(1))
    single_proba = model.predict_proba(single_row)
    assert single_proba.shape == (1, n_classes)
    assert model.predict(single_row).shape == (1,)

    # shuffled inputs return outputs aligned to the shuffled row order
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    native_frame = nw.to_native(nw.from_native(X))
    if hasattr(native_frame, "iloc"):
        shuffled = native_frame.iloc[perm]
    elif hasattr(native_frame, "take"):  # pyarrow Table
        shuffled = native_frame.take(perm)
    else:  # polars DataFrame
        shuffled = native_frame[perm]
    assert np.allclose(model.predict_proba(shuffled), full_proba[perm])
    assert np.array_equal(model.predict(shuffled), full_pred[perm])


def test_refit_does_not_keep_stale_classes():
    """After a successful refit, attributes reflect only the new target's classes."""
    X, y, groups = make_partial_labels_dataset(pd.DataFrame, labels=("a", "b", "c"))
    model = HierarchicalClassifier(LogisticRegression(max_iter=300), groups=groups).fit(X, y)
    assert np.array_equal(model.classes_, ["a", "b", "c"])

    y_new = np.array([10, 20] * (len(y) // 2))
    model.fit(X, y_new)

    assert np.array_equal(model.classes_, [10, 20])
    assert model.n_classes_ == 2
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(X))).issubset({10, 20})
    # every stored estimator only knows classes of the new target
    for estimator in model.estimators_.values():
        assert set(np.asarray(estimator.classes_).tolist()).issubset({10, 20})


@pytest.mark.parametrize("frame_func", frame_funcs)
def test_parent_fallback_returns_full_global_probabilities(frame_func):
    """Unseen groups fall back to parents and still get a complete, normalized global probability row."""
    X, y, groups = make_partial_labels_dataset(frame_func)
    model = HierarchicalClassifier(LogisticRegression(max_iter=300), groups=groups, fallback_method="parent").fit(X, y)

    X_unseen = nw.to_native(
        nw.from_native(X).with_columns(region=nw.lit("unknown_region"), store=nw.lit("unknown_store"))
    )
    proba = model.predict_proba(X_unseen)
    assert proba.shape == (len(y), len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)

    with pytest.raises(KeyError):
        HierarchicalClassifier(LogisticRegression(max_iter=300), groups=groups, fallback_method="raise").fit(
            X, y
        ).predict_proba(X_unseen)
