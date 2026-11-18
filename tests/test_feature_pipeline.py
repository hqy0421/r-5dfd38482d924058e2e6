import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pytest

from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.feature_engineering import fourier, pipeline, trend
from utilsforecast.feature_pipeline import (
    FeatureMatrix,
    _feature_matrix,
    FeatureNode,
    FeaturePipeline,
    compute_features,
    feature_node,
)
from utilsforecast.losses import mae


@pytest.fixture
def series():
    df = generate_series(3, min_length=20, max_length=20, equal_ends=True)
    return df.reset_index(drop=True)


@pytest.fixture
def series_pl(series):
    return pl.from_pandas(series.astype({"unique_id": "int64"}))


def _fourier_node(name="fourier", season_length=7, k=1, dependencies=()):
    def fn(df, freq, h, id_col, time_col):
        return fourier(
            df,
            freq=freq,
            season_length=season_length,
            k=k,
            h=h,
            id_col=id_col,
            time_col=time_col,
        )

    outputs = [f"{op}{i + 1}_{season_length}" for op in ("sin", "cos") for i in range(k)]
    return FeatureNode(name, fn, outputs, dependencies=dependencies)


def _trend_node(name="trend", dependencies=()):
    def fn(df, freq, h, id_col, time_col):
        return trend(df, freq=freq, h=h, id_col=id_col, time_col=time_col)

    return FeatureNode(name, fn, ["trend"], dependencies=dependencies)


class _SumModel:
    def __init__(self):
        self.fit_columns = None
        self.predict_columns = None
        self.fit_target = None
        self.fit_X = None
        self.predict_X = None

    def fit(self, X, y):
        self.fit_columns = list(X.columns)
        self.fit_target = np.asarray(y)
        self.fit_X = np.asarray(X)

    def predict(self, X):
        self.predict_columns = list(X.columns)
        self.predict_X = np.asarray(X)
        return self.predict_X.sum(axis=1)


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------
def test_dependencies_run_before_dependents_and_shared_nodes_once(series):
    calls = []

    def wrap(name, fn):
        def inner(df, freq, h, id_col, time_col):
            calls.append(name)
            return fn(df, freq=freq, h=h, id_col=id_col, time_col=time_col)

        return inner

    seasonal = _fourier_node()
    linear = _trend_node()
    seasonal.fn = wrap("fourier", seasonal.fn)
    linear.fn = wrap("trend", linear.fn)

    def combine(df, id_col, time_col):
        calls.append("combine")
        assert set(df.columns) == {
            "unique_id",
            "ds",
            "y",
            "sin1_7",
            "cos1_7",
            "trend",
        }, df.columns
        out = df[["unique_id", "ds"]].copy()
        out["combined"] = df["sin1_7"] + df["trend"]
        return out

    nodes = [
        FeatureNode("combine", combine, ["combined"], ["fourier", "trend"]),
        seasonal,
        linear,
    ]
    fs = compute_features(series, nodes, freq="D")
    assert calls == ["fourier", "trend", "combine"]
    # the columns follow the dependency order, so dependencies come first
    assert fs.features.columns.tolist() == [
        "unique_id",
        "ds",
        "sin1_7",
        "cos1_7",
        "trend",
        "combined",
    ]

    # a shared node used by two dependents still runs once
    calls.clear()

    def left(df, id_col, time_col):
        return df.assign(left=df["trend"] * 2)[["unique_id", "ds", "left"]]

    def right(df, id_col, time_col):
        return df.assign(right=df["trend"] * 3)[["unique_id", "ds", "right"]]

    nodes = [
        _trend_node(),
        FeatureNode("left", left, ["left"], ["trend"]),
        FeatureNode("right", right, ["right"], ["trend"]),
    ]
    calls.clear()
    nodes[0].fn = wrap("trend", nodes[0].fn)
    fs = compute_features(series, nodes, freq="D")
    assert calls == ["trend"]


def test_node_only_sees_business_columns_and_declared_dependencies(series):
    seen = {}

    def needs_business(df, id_col, time_col):
        seen["business"] = list(df.columns)
        return df[["unique_id", "ds"]].assign(first=df["y"] * 0 + 1.0)

    def needs_first(df, id_col, time_col):
        seen["dependent"] = list(df.columns)
        return df[["unique_id", "ds"]].assign(second=df["first"] * 2)

    def independent(df, id_col, time_col):
        seen["independent"] = list(df.columns)
        return df[["unique_id", "ds"]].assign(third=df["y"] * 0 + 3.0)

    nodes = [
        FeatureNode("first", needs_business, ["first"]),
        FeatureNode("second", needs_first, ["second"], ["first"]),
        FeatureNode("third", independent, ["third"]),
    ]
    compute_features(series, nodes, freq="D")
    assert seen["business"] == ["unique_id", "ds", "y"]
    assert seen["dependent"] == ["unique_id", "ds", "y", "first"]
    assert seen["independent"] == ["unique_id", "ds", "y"]


def test_cycle_is_rejected(series):
    def identity(df, id_col, time_col):
        return df.assign(**{f"v{len(df) % 2}": 1.0})[["unique_id", "ds"]]

    node_a = FeatureNode("a", identity, ["fa"], ["b"])
    node_b = FeatureNode("b", identity, ["fb"], ["a"])
    with pytest.raises(ValueError, match="cycle"):
        compute_features(series, [node_a, node_b], freq="D")
    with pytest.raises(ValueError, match="cycle"):
        FeaturePipeline([node_a, node_b], freq="D", h=2)


def test_missing_dependency_is_rejected(series):
    node = _trend_node("a")
    node.dependencies = ("missing",)
    with pytest.raises(ValueError, match="missing"):
        compute_features(series, [node], freq="D")


def test_output_name_conflicts_are_rejected(series):
    with pytest.raises(ValueError, match="Output names must be unique"):
        compute_features(
            series, [_trend_node("a"), _trend_node("b")], freq="D"
        )

    def collide(df, id_col, time_col):
        return df[["unique_id", "ds"]].assign(y=1.0)

    with pytest.raises(ValueError, match="target column"):
        compute_features(
            series, [FeatureNode("bad", collide, ["y"])], freq="D"
        )

    def key_collide(df, id_col, time_col):
        return df[["unique_id", "ds"]].assign(ds=1)

    with pytest.raises(ValueError, match="key column"):
        compute_features(
            series, [FeatureNode("bad", key_collide, ["ds"])], freq="D"
        )

    with pytest.raises(ValueError, match="Duplicated feature node name"):
        compute_features(
            series, [_trend_node("a"), _trend_node("a")], freq="D"
        )

    with pytest.raises(ValueError, match="at least one node"):
        compute_features(series, [], freq="D")


def test_node_validation_helpers():
    def fn(df):
        return df

    node = FeatureNode("a", fn, "out")
    assert node.outputs == ("out",)
    assert node.dependencies == ()
    assert "a" in repr(node)
    assert node == FeatureNode("a", fn, "out")
    assert hash(node) == hash(FeatureNode("a", fn, "out"))

    with pytest.raises(ValueError, match="non-empty"):
        FeatureNode("", fn, ["a"])
    with pytest.raises(TypeError, match="callable"):
        FeatureNode("a", None, ["a"])
    with pytest.raises(ValueError, match="at least one output"):
        FeatureNode("a", fn, [])
    with pytest.raises(ValueError, match="duplicated outputs"):
        FeatureNode("a", fn, ["x", "x"])
    with pytest.raises(ValueError, match="duplicated dependencies"):
        FeatureNode("a", fn, ["x"], ["d", "d"])
    with pytest.raises(ValueError, match="depends on itself"):
        FeatureNode("a", fn, ["x"], ["a"])
    with pytest.raises(TypeError, match="only strings"):
        FeatureNode("a", fn, [1])

    decorated = feature_node("weekly", outputs=["sin1_7"])(fn)
    assert isinstance(decorated, FeatureNode)
    assert decorated.name == "weekly"
    assert decorated.outputs == ("sin1_7",)

    with pytest.raises(TypeError, match="FeatureNode"):
        compute_features(
            pd.DataFrame({"unique_id": ["a"], "ds": [0], "y": [1.0]}),
            [fn],
            freq=1,
        )


# ---------------------------------------------------------------------------
# key alignment
# ---------------------------------------------------------------------------
def test_results_are_aligned_by_keys_not_position(series):
    shuffled = series.sample(frac=1.0, random_state=0).reset_index(drop=True)

    def positional(df, id_col, time_col):
        # value encodes the position of the row in the *original* frame
        ordered = df.sort_values([id_col, time_col]).reset_index(drop=True)
        ordered["row"] = np.arange(ordered.shape[0], dtype=float)
        return ordered.sample(frac=1.0, random_state=1).reset_index(drop=True)

    fs = compute_features(shuffled, [FeatureNode("p", positional, ["row"])], freq="D")
    merged = fs.features.merge(
        series.sort_values(["unique_id", "ds"]).assign(
            expected=np.arange(series.shape[0], dtype=float)
        ),
        on=["unique_id", "ds"],
    )
    np.testing.assert_array_equal(merged["row"], merged["expected"])
    assert fs.features[["unique_id", "ds"]].equals(
        ufp_reset(shuffled[["unique_id", "ds"]])
    )


def ufp_reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def test_duplicated_keys_are_rejected(series):
    dup = pd.concat([series, series.iloc[[0]]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="duplicated"):
        compute_features(dup, [_trend_node()], freq="D")

    def duplicating(df, id_col, time_col):
        out = df[["unique_id", "ds"]].iloc[[0, 0]].copy()
        out["feat"] = 1.0
        return out

    with pytest.raises(ValueError, match="duplicated"):
        compute_features(series, [FeatureNode("d", duplicating, ["feat"])], freq="D")


def test_node_with_wrong_keys_is_rejected(series):
    def missing_row(df, id_col, time_col):
        out = df[["unique_id", "ds"]].iloc[:-1].copy()
        out["feat"] = 1.0
        return out

    with pytest.raises(ValueError, match="missing keys"):
        compute_features(series, [FeatureNode("m", missing_row, ["feat"])], freq="D")

    def extra_row(df, id_col, time_col):
        out = df[["unique_id", "ds"]].copy()
        out["ds"] = out["ds"] + pd.Timedelta(days=1)
        out["feat"] = 1.0
        return out

    with pytest.raises(ValueError, match="missing keys"):
        compute_features(series, [FeatureNode("e", extra_row, ["feat"])], freq="D")

    def shifted_keys(df, id_col, time_col):
        out = df[["unique_id", "ds"]].iloc[:-1].copy()
        out["ds"] = out["ds"] + pd.Timedelta(days=1)
        out["feat"] = 1.0
        return out

    with pytest.raises(ValueError, match="unexpected keys"):
        compute_features(
            series, [FeatureNode("s", shifted_keys, ["feat"])], freq="D"
        )

    def missing_col(df, id_col, time_col):
        return df[["unique_id", "ds"]]

    with pytest.raises(ValueError, match="missing the following columns"):
        compute_features(series, [FeatureNode("c", missing_col, ["feat"])], freq="D")


# ---------------------------------------------------------------------------
# future features and forecasts
# ---------------------------------------------------------------------------
def test_future_features_cover_future_keys(series):
    nodes = [_fourier_node(), _trend_node()]
    fs = compute_features(series, nodes, freq="D", h=4)
    expected = series.groupby("unique_id", observed=True)["ds"].max() + pd.Timedelta(days=1)
    expected = expected.reset_index()
    expected["ds"] = expected["ds"].dt.normalize()
    expected = pd.concat(
        [
            expected.assign(ds=expected["ds"] + pd.Timedelta(days=i))
            for i in range(4)
        ]
    ).reset_index(drop=True)
    got = fs.future_features[["unique_id", "ds"]].sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)
    exp = expected.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, exp)
    assert fs.future_features.columns.tolist() == [
        "unique_id",
        "ds",
        "sin1_7",
        "cos1_7",
        "trend",
    ]


def test_fit_sees_history_and_target_predict_does_not(series):
    nodes = [_fourier_node(), _trend_node()]
    model = _SumModel()
    pipe = FeaturePipeline(nodes, freq="D", h=3)
    pipe.fit(series, model=model)
    fs = compute_features(series, nodes, freq="D", h=3)
    # the model inputs are exactly the graph outputs
    assert model.fit_columns == fs.features.columns.tolist()[2:]
    np.testing.assert_allclose(model.fit_X, fs.features[model.fit_columns].to_numpy())
    np.testing.assert_array_equal(model.fit_target, series["y"].to_numpy())
    assert "y" not in model.fit_columns

    result = pipe.predict()
    assert model.predict_columns == fs.future_features.columns.tolist()[2:]
    np.testing.assert_allclose(
        model.predict_X, fs.future_features[model.predict_columns].to_numpy()
    )
    assert "y" not in model.predict_columns
    np.testing.assert_allclose(
        result.forecast["model"].to_numpy(),
        result.future_features[model.predict_columns].to_numpy().sum(axis=1),
    )
    pd.testing.assert_frame_equal(result.features, fs.features)
    pd.testing.assert_frame_equal(result.future_features, fs.future_features)


def test_predict_requires_fit(series):
    pipe = FeaturePipeline([_trend_node()], freq="D", h=2)
    with pytest.raises(ValueError, match="before `predict`"):
        pipe.predict()
    with pytest.raises(ValueError, match="requires a `model`"):
        pipe.fit(series)

    class NoPredict:
        def fit(self, X, y):
            pass

    with pytest.raises(TypeError, match="predict"):
        pipe.fit(series, NoPredict())

    with pytest.raises(ValueError, match="target column"):
        FeaturePipeline([_trend_node()], freq="D", h=2).fit(
            series.drop(columns="y"), _SumModel()
        )


def test_model_errors_never_return_a_partial_forecast(series):
    class WrongLength:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0] - 1)

    class WithNaN:
        def fit(self, X, y):
            pass

        def predict(self, X):
            out = np.zeros(X.shape[0])
            out[0] = np.nan
            return out

    class BadShape:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros((X.shape[0], 1, 1))

    for model, match in [
        (WrongLength(), "predictions but there are"),
        (WithNaN(), "null values"),
        (BadShape(), "dimensions"),
    ]:
        pipe = FeaturePipeline([_trend_node()], freq="D", h=2).fit(series, model)
        with pytest.raises(ValueError, match=match):
            pipe.predict()


def test_node_errors_never_return_a_partial_forecast(series):
    def half_future(df, freq, h, id_col, time_col):
        historic = df[["unique_id", "ds", "y"]].assign(feat=1.0)
        future = df[["unique_id", "ds"]].copy()
        future["ds"] = future["ds"] + pd.Timedelta(days=h)
        future["feat"] = 1.0
        return historic, future.iloc[:-1]

    pipe = FeaturePipeline(
        [FeatureNode("f", half_future, ["feat"])], freq="D", h=2
    ).fit(series, _SumModel())
    with pytest.raises(ValueError, match="does not cover the expected"):
        pipe.predict()


def test_x_df_is_validated(series):
    nodes = [_trend_node()]
    future = series.groupby("unique_id", observed=True)["ds"].max().reset_index()
    future["ds"] = future["ds"] + pd.Timedelta(days=1)
    future["ds"] = future["ds"].dt.normalize()

    with pytest.raises(ValueError, match="target column"):
        compute_features(series, nodes, freq="D", h=1, X_df=future.assign(y=0.0))
    with pytest.raises(ValueError, match="business columns"):
        compute_features(series, nodes, freq="D", h=1, X_df=future.assign(other=1.0))
    with pytest.raises(ValueError, match="does not cover the expected"):
        compute_features(series, nodes, freq="D", h=2, X_df=future)


def test_extra_future_business_columns_reach_the_nodes(series):
    series = series.assign(price=np.linspace(1.0, 2.0, series.shape[0]))
    seen = {}

    def scaled(df, id_col, time_col):
        seen.setdefault("cols", []).append(list(df.columns))
        out = df[["unique_id", "ds"]].copy()
        out["scaled"] = df["trend"] * df["price"]
        return out.sort_values(time_col, ascending=False)

    nodes = [
        _trend_node(),
        FeatureNode("scaled", scaled, ["scaled"], ["trend"]),
    ]
    future = series.groupby("unique_id", observed=True)["ds"].max().reset_index()
    future["ds"] = (future["ds"] + pd.Timedelta(days=1)).dt.normalize()
    future["price"] = 2.0
    fs = compute_features(series, nodes, freq="D", h=1, X_df=future)
    assert seen["cols"][0] == ["unique_id", "ds", "y", "price", "trend"]
    assert seen["cols"][1] == ["unique_id", "ds", "price", "trend"]
    assert fs.future_features.columns.tolist() == [
        "unique_id",
        "ds",
        "trend",
        "scaled",
    ]
    np.testing.assert_allclose(
        fs.future_features["scaled"].to_numpy(),
        fs.future_features["trend"].to_numpy() * 2.0,
    )

    # without the future values of the business column the graph can't run
    with pytest.raises(ValueError, match="Provide the future values"):
        compute_features(series, nodes, freq="D", h=1)


def test_models_can_output_multiple_columns(series):
    class TwoOutputs:
        def fit(self, X, y):
            pass

        def predict(self, X):
            values = np.asarray(X).sum(axis=1)
            return np.column_stack([values, values * 2])

    pipe = FeaturePipeline(
        [_trend_node()], freq="D", h=2, model_names=["lo", "hi"]
    ).fit(series, TwoOutputs())
    result = pipe.predict()
    assert result.forecast.columns.tolist() == ["unique_id", "ds", "lo", "hi"]
    np.testing.assert_allclose(
        result.forecast["hi"].to_numpy(), 2 * result.forecast["lo"].to_numpy()
    )

    pipe = FeaturePipeline([_trend_node()], freq="D", h=2).fit(
        series, TwoOutputs()
    )
    result = pipe.predict()
    assert result.forecast.columns.tolist() == ["unique_id", "ds", "model_0", "model_1"]

    with pytest.raises(ValueError, match="`model_names` has"):
        FeaturePipeline(
            [_trend_node()], freq="D", h=2, model_names=["only"]
        ).fit(series, TwoOutputs()).predict()

    with pytest.raises(ValueError, match="key columns"):
        FeaturePipeline(
            [_trend_node()], freq="D", h=2, model_name="ds"
        ).fit(series, _SumModel()).predict()


def test_features_can_be_used_with_native_evaluation_and_plotting(series):
    try:
        import matplotlib

        matplotlib.use("Agg")
        from utilsforecast.plotting import plot_series
    except ImportError:  # pragma: no cover
        plot_series = None

    nodes = [_fourier_node(), _trend_node()]
    pipe = FeaturePipeline(nodes, freq="D", h=2).fit(series, _SumModel())
    result = pipe.predict()
    cutoff = series["ds"].max() - pd.Timedelta(days=1)
    test = series[series["ds"] > cutoff].merge(
        result.forecast.rename(columns={"model": "fourier_trend"}),
        on=["unique_id", "ds"],
    )
    evaluation = evaluate(test, metrics=[mae], train_df=series)
    assert evaluation.columns.tolist() == ["unique_id", "metric", "fourier_trend"]
    assert np.isfinite(evaluation["fourier_trend"]).all()

    if plot_series is not None:
        ax = plot_series(series, forecasts_df=result.forecast, max_ids=1)
        assert ax is not None

    # the features are separately usable, e.g. for a native evaluation
    train_features = result.features.merge(series, on=["unique_id", "ds"])
    assert set(nodes[0].outputs).issubset(train_features.columns)
    assert set(nodes[1].outputs).issubset(train_features.columns)


def test_feature_matrix():
    values = np.arange(6.0).reshape(3, 2)
    X = FeatureMatrix(values, ["a", "b"])
    assert X.columns == ["a", "b"]
    np.testing.assert_array_equal(X["a"], values[:, 0])
    np.testing.assert_array_equal(X[["b", "a"]], values[:, ::-1])
    assert X[["b", "a"]].columns == ["b", "a"]
    np.testing.assert_array_equal(X[:, 1], values[:, 1])
    np.testing.assert_array_equal(X.to_numpy(), values)
    pd.testing.assert_frame_equal(X.to_pandas(), pd.DataFrame(values, columns=["a", "b"]))


# ---------------------------------------------------------------------------
# backends and dtypes
# ---------------------------------------------------------------------------
def test_pandas_dtypes_are_preserved(series):
    series = series.copy()
    series["unique_id"] = series["unique_id"].astype("category")
    nodes = [_trend_node()]
    fs = compute_features(series, nodes, freq="D", h=2)
    assert fs.features["unique_id"].dtype == series["unique_id"].dtype
    assert fs.features["ds"].dtype == series["ds"].dtype
    assert fs.future_features["unique_id"].dtype == series["unique_id"].dtype
    assert fs.future_features["ds"].dtype == series["ds"].dtype

    with_tz = series.copy()
    with_tz["ds"] = with_tz["ds"].dt.tz_localize("UTC")
    fs_tz = compute_features(with_tz, nodes, freq="D", h=2)
    assert str(fs_tz.features["ds"].dtype) == "datetime64[ns, UTC]"
    assert str(fs_tz.future_features["ds"].dtype) == "datetime64[ns, UTC]"


def test_polars_roundtrip(series_pl):
    nodes = [_fourier_node(), _trend_node()]
    fs = compute_features(series_pl, nodes, freq="1d", h=2)
    assert isinstance(fs.features, pl.DataFrame)
    assert isinstance(fs.future_features, pl.DataFrame)
    assert fs.features.schema["unique_id"] == series_pl.schema["unique_id"]
    assert fs.future_features.schema["unique_id"] == series_pl.schema["unique_id"]
    assert fs.future_features.schema["ds"] == series_pl.schema["ds"]
    assert fs.features.columns == ["unique_id", "ds", "sin1_7", "cos1_7", "trend"]

    model = _SumModel()
    pipe = FeaturePipeline(nodes, freq="1d", h=2).fit(series_pl, model)
    result = pipe.predict()
    assert isinstance(result.forecast, pl.DataFrame)
    assert result.forecast.columns == ["unique_id", "ds", "model"]
    assert result.forecast.schema["unique_id"] == series_pl.schema["unique_id"]
    assert result.forecast.schema["ds"] == series_pl.schema["ds"]
    pd_result = result.forecast.to_pandas()
    pd.testing.assert_frame_equal(
        pd_result[["unique_id", "ds"]],
        result.future_features.to_pandas()[["unique_id", "ds"]],
    )


def test_mixing_backends_is_rejected(series, series_pl):
    with pytest.raises(ValueError, match="same backend"):
        compute_features(series, [_trend_node()], freq="D", h=1, X_df=series_pl)


# ---------------------------------------------------------------------------
# backwards compatibility
# ---------------------------------------------------------------------------
def test_legacy_pipeline_is_unchanged(series):
    features = [
        lambda df, freq, h, id_col, time_col: fourier(
            df,
            freq=freq,
            season_length=7,
            k=1,
            h=h,
            id_col=id_col,
            time_col=time_col,
        ),
        lambda df, freq, h, id_col, time_col: trend(
            df, freq=freq, h=h, id_col=id_col, time_col=time_col
        ),
    ]
    transformed, future = pipeline(series, features=features, freq="D", h=2)
    fs = compute_features(
        series,
        [_fourier_node(), _trend_node()],
        freq="D",
        h=2,
    )
    np.testing.assert_allclose(
        transformed[["sin1_7", "cos1_7", "trend"]].to_numpy(),
        fs.features[["sin1_7", "cos1_7", "trend"]].to_numpy(),
    )
    np.testing.assert_allclose(
        future[["sin1_7", "cos1_7", "trend"]].to_numpy(),
        fs.future_features[["sin1_7", "cos1_7", "trend"]].to_numpy(),
    )


def test_integer_times_and_non_numeric_features(series_pl):
    int_series = series_pl.with_columns(
        (pl.col("ds").rank("dense").over("unique_id") - 1)
        .cast(pl.Int32)
        .alias("ds")
    )
    nodes = [_trend_node()]
    fs = compute_features(int_series, nodes, freq=1, h=2)
    assert fs.features.schema["ds"] == pl.Int32
    assert fs.future_features.schema["ds"] == pl.Int32
    assert fs.future_features["ds"].to_list() == [20, 21] * 3

    # pandas integer times as well
    pd_int = int_series.to_pandas()
    fs = compute_features(pd_int, nodes, freq=1, h=2)
    assert fs.future_features["ds"].dtype == pd_int["ds"].dtype

    def non_numeric(df, id_col, time_col):
        return df[["unique_id", "ds"]].assign(label="a")

    pipe = FeaturePipeline(
        [FeatureNode("label", non_numeric, ["label"])], freq=1, h=1
    )
    with pytest.raises(ValueError, match="must be numeric"):
        pipe.fit(pd_int, _SumModel())


def test_transform_and_helpers(series):
    pipe = FeaturePipeline([_fourier_node(), _trend_node()], freq="D", h=2)
    fs = pipe.transform(series)
    assert fs.features.columns.tolist() == ["unique_id", "ds", *pipe.feature_names]
    assert "FeaturePipeline" in repr(pipe)
    assert pipe.feature_names == ["sin1_7", "cos1_7", "trend"]

    with pytest.raises(ValueError, match="positive integer"):
        FeaturePipeline([_trend_node()], freq="D", h=0)
    with pytest.raises(ValueError, match="non-negative integer"):
        compute_features(series, [_trend_node()], freq="D", h=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        compute_features(series, [_trend_node()], freq="D", h=True)
    with pytest.raises(ValueError, match="didn't produce any model inputs"):
        _feature_matrix(series[["unique_id", "ds"]], [])


def test_horizon_zero_has_no_future(series):
    fs = compute_features(series, [_trend_node()], freq="D")
    assert fs.future_features.shape[0] == 0
    assert fs.future_features.columns.tolist() == ["unique_id", "ds", "trend"]


def test_node_equality_and_bad_return_types(series):
    def fn(df):
        return df

    assert (FeatureNode("a", fn, ["x"]) == 1) is not True
    assert FeatureNode("a", fn, ["x"]) != FeatureNode("b", fn, ["x"])

    def triple(df, id_col, time_col):
        out = df[["unique_id", "ds"]].assign(feat=1.0)
        return out, out, out

    with pytest.raises(ValueError, match="tuple of length"):
        compute_features(series, [FeatureNode("t", triple, ["feat"])], freq="D")

    def duplicate_columns(df, id_col, time_col):
        out = df[["unique_id", "ds"]].assign(feat=1.0)
        return pd.concat([out, out[["feat"]]], axis=1)

    with pytest.raises(ValueError, match="duplicated columns"):
        compute_features(
            series, [FeatureNode("d", duplicate_columns, ["feat"])], freq="D"
        )


def test_predict_with_x_df(series):
    series = series.assign(price=1.0)
    future = series.groupby("unique_id", observed=True)["ds"].max().reset_index()
    future["ds"] = (future["ds"] + pd.Timedelta(days=1)).dt.normalize()
    future["price"] = 3.0

    def scaled(df, id_col, time_col):
        return df[["unique_id", "ds"]].assign(p=df["trend"] * df["price"])

    pipe = FeaturePipeline(
        [_trend_node(), FeatureNode("scaled", scaled, ["p"], ["trend"])],
        freq="D",
        h=1,
    ).fit(series, _SumModel())
    result = pipe.predict(X_df=future)
    np.testing.assert_allclose(
        result.future_features["p"].to_numpy(),
        result.future_features["trend"].to_numpy() * 3.0,
    )
    assert pipe.future_features_ is result.future_features

    with pytest.raises(ValueError, match="same backend"):
        pipe.predict(X_df=pl.from_pandas(future))


def test_null_keys_and_small_frames(series):
    single = series[series["unique_id"] == 0].iloc[[0]]
    fs = compute_features(single, [_trend_node()], freq="D", h=1)
    assert fs.features.shape[0] == 1

    with_nulls = series.copy()
    with_nulls.loc[with_nulls.index[0], "unique_id"] = None
    with pytest.raises(ValueError, match="contains null values"):
        compute_features(with_nulls, [_trend_node()], freq="D")

    with_nulls = series.copy()
    with_nulls.loc[with_nulls.index[0], "ds"] = pd.NaT
    with pytest.raises(ValueError, match="contains null values"):
        compute_features(with_nulls, [_trend_node()], freq="D")

    # a node that returns null keys is rejected too
    def null_keys(df, id_col, time_col):
        out = df[["unique_id", "ds"]].copy()
        out["feat"] = 1.0
        out.loc[out.index[0], "ds"] = pd.NaT
        return out

    with pytest.raises(ValueError, match="null values in the key columns"):
        compute_features(series, [FeatureNode("n", null_keys, ["feat"])], freq="D")


def test_output_can_not_shadow_existing_business_column(series):
    series = series.assign(aux=1.0)

    def shadow(df, id_col, time_col):
        return df[["unique_id", "ds"]].assign(aux=2.0)

    with pytest.raises(ValueError, match="already exists in the input dataframe"):
        compute_features(series, [FeatureNode("s", shadow, ["aux"])], freq="D")


def test_node_exceptions_propagate(series):
    class Boom(Exception):
        pass

    def explode(df, freq, h, id_col, time_col):
        raise Boom("boom")

    with pytest.raises(Boom, match="boom"):
        compute_features(series, [FeatureNode("b", explode, ["feat"])], freq="D")


def test_restore_key_dtypes():
    from utilsforecast.feature_pipeline import _restore_key_dtypes

    base = pd.DataFrame(
        {
            "unique_id": pd.Series(["a", "b"], dtype="category"),
            "ds": pd.Series([1, 2], dtype="int32"),
        }
    )
    future = pd.DataFrame({"unique_id": ["a", "b"], "ds": pd.Series([3, 4])})
    out = _restore_key_dtypes(future, base, "unique_id", "ds")
    assert out["unique_id"].dtype == base["unique_id"].dtype
    assert out["ds"].dtype == base["ds"].dtype

    tz_base = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02"]).tz_localize("UTC"),
        }
    )
    naive = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": pd.to_datetime(["2020-01-03", "2020-01-04"])}
    )
    out = _restore_key_dtypes(naive, tz_base, "unique_id", "ds")
    assert str(out["ds"].dtype) == "datetime64[ns, UTC]"

    pl_base = pl.DataFrame(
        {
            "unique_id": pl.Series(["a", "b"], dtype=pl.Categorical),
            "ds": pl.Series([1, 2], dtype=pl.Int32),
        }
    )
    pl_future = pl.DataFrame(
        {
            "unique_id": pl.Series(["a", "b"], dtype=pl.Categorical),
            "ds": pl.Series([3, 4], dtype=pl.Int64),
        }
    )
    out = _restore_key_dtypes(pl_future, pl_base, "unique_id", "ds")
    assert out.schema["ds"] == pl.Int32
