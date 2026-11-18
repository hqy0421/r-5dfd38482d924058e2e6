"""Forecasting pipelines whose features declare their dependencies"""

__all__ = [
    "FeatureNode",
    "FeaturePipeline",
    "FeatureSet",
    "ForecastResult",
    "FeatureMatrix",
    "feature_node",
    "compute_features",
]


import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import DataFrame, DFType, pl
from .validation import _has_nulls, _is_dt_dtype, validate_format, validate_freq

_POS_COL = "__utilsforecast_pos__"
_ROW_COL = "__utilsforecast_row__"


# ---------------------------------------------------------------------------
# backend helpers
# ---------------------------------------------------------------------------
def _is_pandas(df: Any) -> bool:
    return isinstance(df, pd.DataFrame)


def _check_backend(df: Any, ref: DataFrame, ctx: str) -> None:
    if _is_pandas(df) != _is_pandas(ref):
        raise ValueError(
            f"{ctx} must be a {type(ref).__name__}, got {type(df).__name__}. "
            "All the dataframes in a feature graph must use the same backend."
        )


def _as_str_tuple(
    value: Union[str, Iterable[str]], arg: str, name: str
) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    out = tuple(value)
    bad = [v for v in out if not isinstance(v, str)]
    if bad:
        raise TypeError(
            f"`{arg}` of feature node {name!r} must contain only strings, got {bad!r}."
        )
    return out


def _take(df: DFType, idxs: np.ndarray) -> DFType:
    return ufp.drop_index_if_pandas(ufp.take_rows(df, idxs))


def _empty_rows(df: DFType) -> DFType:
    return _take(df, np.array([], dtype=np.int64))


# ---------------------------------------------------------------------------
# key handling
# ---------------------------------------------------------------------------
def _key_arrays(
    df: DataFrame, id_col: str, time_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return backend independent representations of the (id, time) keys."""
    ids = df[id_col]
    times = df[time_col]
    if _is_pandas(df):
        ids_arr = ids.astype(str).to_numpy(dtype=object)
        if _is_dt_dtype(times):
            if getattr(times.dtype, "tz", None) is not None:
                times = times.dt.tz_convert("UTC").dt.tz_localize(None)
            times_arr = times.astype("datetime64[ns]").astype("int64").to_numpy()
        else:
            times_arr = times.to_numpy().astype(np.int64)
    else:
        ids_arr = ids.cast(pl.Utf8).to_numpy()
        if _is_dt_dtype(times):
            times_arr = times.cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
        else:
            times_arr = times.cast(pl.Int64).to_numpy()
    return ids_arr, times_arr


def _has_duplicate_keys(ids: np.ndarray, times: np.ndarray) -> bool:
    if ids.size < 2:
        return False
    order = np.lexsort((times, ids))
    sorted_ids = ids[order]
    sorted_times = times[order]
    same_id = sorted_ids[1:] == sorted_ids[:-1]
    same_time = sorted_times[1:] == sorted_times[:-1]
    return bool((same_id & same_time).any())


def _sorted_keys(
    ids: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=object)
    times = np.asarray(times, dtype=np.int64)
    order = np.lexsort((times, ids))
    return ids[order], times[order]


def _keys_to_str(keys: Iterable[Tuple[str, int]], limit: int = 5) -> str:
    ordered = sorted(keys)
    shown = ", ".join(f"({i!r}, {t!r})" for i, t in ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... ({len(ordered)} total)"
    return shown


def _check_unique_keys(df: DataFrame, id_col: str, time_col: str, ctx: str) -> None:
    if _has_nulls(df[id_col]) or _has_nulls(df[time_col]):
        raise ValueError(
            f"{ctx} contains null values in the key columns ({id_col!r}, {time_col!r})."
        )
    ids, times = _key_arrays(df, id_col, time_col)
    if _has_duplicate_keys(ids, times):
        raise ValueError(
            f"{ctx} contains duplicated ({id_col!r}, {time_col!r}) keys. "
            "Keys must uniquely identify each row."
        )


def _align_rows(
    ref: DataFrame, other: DataFrame, id_col: str, time_col: str
) -> np.ndarray:
    """Return the positions of `other`'s rows that correspond to `ref`'s rows."""
    ref_ids, ref_times = _key_arrays(ref, id_col, time_col)
    other_ids, other_times = _key_arrays(other, id_col, time_col)
    if _is_pandas(ref):
        left = pd.DataFrame(
            {_ROW_COL: np.arange(ref_ids.size), id_col: ref_ids, time_col: ref_times}
        )
        right = pd.DataFrame(
            {
                _POS_COL: np.arange(other_ids.size),
                id_col: other_ids,
                time_col: other_times,
            }
        )
        merged = left.merge(right, on=[id_col, time_col], how="left", sort=False)
        idxs = merged.sort_values(_ROW_COL, kind="stable")[_POS_COL].to_numpy()
    else:
        left = pl.DataFrame(
            {
                _ROW_COL: np.arange(ref_ids.size, dtype=np.int64),
                id_col: pl.Series(ref_ids.tolist(), dtype=pl.Utf8),
                time_col: pl.Series(ref_times, dtype=pl.Int64),
            }
        )
        right = pl.DataFrame(
            {
                _POS_COL: np.arange(other_ids.size, dtype=np.int64),
                id_col: pl.Series(other_ids.tolist(), dtype=pl.Utf8),
                time_col: pl.Series(other_times, dtype=pl.Int64),
            }
        )
        merged = left.join(right, on=[id_col, time_col], how="left")
        idxs = merged.sort(_ROW_COL, maintain_order=True)[_POS_COL].to_numpy()
    return idxs.astype(np.int64)


def _aligned_frame(
    ref: DFType,
    other: DataFrame,
    id_col: str,
    time_col: str,
    columns: Sequence[str],
    ctx: str,
) -> DFType:
    """Align `other` to the keys of `ref`, keeping only `columns` from `other`.

    `other` rows may come in any order, but its keys have to match `ref`'s keys
    exactly (no missing or extra keys and no duplicates).
    """
    _check_backend(other, ref, ctx)
    columns = list(columns)
    other_cols = list(other.columns)
    missing = [c for c in [id_col, time_col, *columns] if c not in other_cols]
    if missing:
        raise ValueError(f"{ctx} is missing the following columns: {missing}.")
    duplicated = [c for c in columns if other_cols.count(c) > 1]
    if duplicated:
        raise ValueError(f"{ctx} contains duplicated columns: {duplicated}.")
    _check_unique_keys(other, id_col, time_col, ctx)
    ref_ids, ref_times = _key_arrays(ref, id_col, time_col)
    other_ids, other_times = _key_arrays(other, id_col, time_col)
    ref_keys = _sorted_keys(ref_ids, ref_times)
    other_keys = _sorted_keys(other_ids, other_times)
    same_keys = np.array_equal(
        ref_keys[0].astype(str), other_keys[0].astype(str)
    ) and np.array_equal(ref_keys[1], other_keys[1])
    if ref_ids.size != other_ids.size or not same_keys:
        ref_set = set(zip(ref_ids.tolist(), ref_times.tolist()))
        other_set = set(zip(other_ids.tolist(), other_times.tolist()))
        parts = []
        if ref_set - other_set:
            parts.append("missing keys: " + _keys_to_str(ref_set - other_set))
        if other_set - ref_set:
            parts.append("unexpected keys: " + _keys_to_str(other_set - ref_set))
        raise ValueError(
            f"{ctx} does not cover the expected ({id_col!r}, {time_col!r}) keys. "
            + "; ".join(parts)
        )
    idxs = _align_rows(ref, other, id_col, time_col)
    keep = [c for c in columns if c not in (id_col, time_col)]
    taken = _take(other[keep] if keep else other[[id_col]], idxs)
    out = _take(ref[[id_col, time_col]], np.arange(ref_ids.size))
    for col in keep:
        out = ufp.assign_columns(out, col, taken[col])
    return out


# ---------------------------------------------------------------------------
# feature nodes
# ---------------------------------------------------------------------------
class FeatureNode:
    """A feature computation with explicit dependencies and outputs.

    Args:
        name (str): Unique name that identifies this node in the graph.
        fn (callable): Function that computes the features. It receives a
            dataframe with the original business columns plus the outputs of
            its declared dependencies (and nothing else) as first positional
            argument, and whichever of ``freq``, ``h``, ``id_col`` and
            ``time_col`` its signature accepts as keyword arguments. Rows can
            come in any order, they're aligned by the keys.

            The return value defines how the future values are obtained:

            - a dataframe with the key columns and the declared outputs. It's
              evaluated twice, once on the history and once on the future keys,
              so it sees the future values of the business columns (from
              ``X_df``) and of its dependencies.
            - a ``(historic, future)`` pair of such dataframes. It's evaluated
              once on the history and the future part is reused, which is what
              features that project the history forward (e.g. ``fourier`` or
              ``trend``) need.
        outputs (sequence of str): Columns produced by this node. They have to
            be unique across the graph and can't shadow the keys, the target or
            the original columns of the input dataframe.
        dependencies (sequence of str, optional): Names of the nodes whose
            outputs this node needs.

    Since a node only gets what it declared, two nodes that compute the same
    kind of feature can depend on different inputs without leaking columns into
    each other.
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        outputs: Union[str, Iterable[str]],
        dependencies: Union[str, Iterable[str]] = (),
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Feature node names must be non-empty strings.")
        if not callable(fn):
            raise TypeError(f"The function of feature node {name!r} must be callable.")
        outputs = _as_str_tuple(outputs, "outputs", name)
        dependencies = _as_str_tuple(dependencies, "dependencies", name)
        if not outputs:
            raise ValueError(
                f"Feature node {name!r} must declare at least one output column."
            )
        duplicated_outputs = sorted({c for c in outputs if outputs.count(c) > 1})
        if duplicated_outputs:
            raise ValueError(
                f"Feature node {name!r} declares duplicated outputs: "
                f"{duplicated_outputs}."
            )
        duplicated_deps = sorted({d for d in dependencies if dependencies.count(d) > 1})
        if duplicated_deps:
            raise ValueError(
                f"Feature node {name!r} declares duplicated dependencies: "
                f"{duplicated_deps}."
            )
        if name in dependencies:
            raise ValueError(
                f"Feature node {name!r} depends on itself, which creates a cycle."
            )
        self.name = name
        self.fn = fn
        self.outputs = outputs
        self.dependencies = dependencies

    def __repr__(self) -> str:
        return (
            f"FeatureNode(name={self.name!r}, outputs={list(self.outputs)!r}, "
            f"dependencies={list(self.dependencies)!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FeatureNode):
            return NotImplemented
        return (
            self.name == other.name
            and self.outputs == other.outputs
            and self.dependencies == other.dependencies
            and self.fn == other.fn
        )

    def __hash__(self) -> int:
        return hash((self.name, self.outputs, self.dependencies, self.fn))


def feature_node(
    name: str,
    outputs: Union[str, Iterable[str]],
    dependencies: Union[str, Iterable[str]] = (),
) -> Callable[[Callable], FeatureNode]:
    """Decorator that turns a function into a :class:`FeatureNode`.

    ```python
    @feature_node("weekly", outputs=["sin1_7", "cos1_7"])
    def weekly(df, freq, h, id_col, time_col):
        ...
    ```
    """

    def decorator(fn: Callable) -> FeatureNode:
        return FeatureNode(name=name, fn=fn, outputs=outputs, dependencies=dependencies)

    return decorator


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
class FeatureSet(NamedTuple):
    """Aligned historic and future features.

    Attributes:
        features: Dataframe with the key columns and the features computed for
            the historic data, in the row order of the input dataframe.
        future_features: Dataframe with the future keys and their features. It's
            empty when no horizon was requested.
    """

    features: DataFrame
    future_features: DataFrame


class ForecastResult(NamedTuple):
    """Features and forecasts produced by a :class:`FeaturePipeline`.

    Attributes:
        features: Historic features, aligned by the key columns.
        future_features: Future features, aligned by the future keys. These are
            the exact values that were passed to the model.
        forecast: Dataframe with the key columns and the predictions.
    """

    features: DataFrame
    future_features: DataFrame
    forecast: DataFrame


# ---------------------------------------------------------------------------
# graph validation
# ---------------------------------------------------------------------------
def _topological_order(nodes: Sequence[FeatureNode]) -> List[str]:
    indegree = {node.name: len(node.dependencies) for node in nodes}
    children: Dict[str, List[str]] = {node.name: [] for node in nodes}
    for node in nodes:
        for dep in node.dependencies:
            children[dep].append(node.name)
    queue = [node.name for node in nodes if indegree[node.name] == 0]
    order: List[str] = []
    while queue:
        name = queue.pop(0)
        order.append(name)
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        done = set(order)
        cyclic = sorted(node.name for node in nodes if node.name not in done)
        raise ValueError(
            f"The feature graph contains a cycle. The nodes involved are {cyclic}."
        )
    return order


def _validate_graph(
    nodes: Sequence[FeatureNode], id_col: str, time_col: str
) -> List[str]:
    if not nodes:
        raise ValueError("The feature graph must contain at least one node.")
    by_name: Dict[str, FeatureNode] = {}
    for node in nodes:
        if not isinstance(node, FeatureNode):
            raise TypeError(
                "Nodes must be `FeatureNode` instances (see `FeatureNode` and "
                f"`feature_node`), got {type(node).__name__}."
            )
        if node.name in by_name:
            raise ValueError(f"Duplicated feature node name: {node.name!r}.")
        by_name[node.name] = node
    owner: Dict[str, str] = {}
    for node in nodes:
        for dep in node.dependencies:
            if dep not in by_name:
                raise ValueError(
                    f"Feature node {node.name!r} depends on {dep!r}, which isn't "
                    f"part of the graph. Available nodes: {sorted(by_name)}."
                )
        for out in node.outputs:
            if out in (id_col, time_col):
                raise ValueError(
                    f"Feature node {node.name!r} can't produce the key column {out!r}."
                )
            if out in owner:
                raise ValueError(
                    f"Column {out!r} is produced by both {owner[out]!r} and "
                    f"{node.name!r}. Output names must be unique."
                )
            owner[out] = node.name
    return _topological_order(nodes)


def _check_output_collisions(
    nodes: Sequence[FeatureNode],
    columns: Sequence[str],
    id_col: str,
    time_col: str,
    target_col: Optional[str],
) -> None:
    for node in nodes:
        for out in node.outputs:
            if target_col is not None and out == target_col:
                raise ValueError(
                    f"Feature node {node.name!r} can't produce the target column "
                    f"{target_col!r}."
                )
            if out in columns and out not in (id_col, time_col):
                raise ValueError(
                    f"Feature node {node.name!r} produces column {out!r}, which "
                    "already exists in the input dataframe."
                )


# ---------------------------------------------------------------------------
# node execution
# ---------------------------------------------------------------------------
def _missing_column_error(exc: Exception) -> bool:
    # pandas raises a KeyError, polars a ColumnNotFoundError
    return isinstance(exc, KeyError) or (
        exc.__class__.__name__ == "ColumnNotFoundError"
    )


def _call_node(
    node: FeatureNode,
    df: DFType,
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    is_future: bool = False,
) -> Tuple[DataFrame, Optional[DataFrame]]:
    all_kwargs: Dict[str, Any] = {
        "freq": freq,
        "h": h,
        "id_col": id_col,
        "time_col": time_col,
    }
    try:
        params = inspect.signature(node.fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - no signature available
        params = None
    if params is not None and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        all_kwargs = {k: v for k, v in all_kwargs.items() if k in params}
    try:
        out = node.fn(df, **all_kwargs)
    except Exception as exc:
        if is_future and _missing_column_error(exc):
            raise ValueError(
                f"The future part of feature node {node.name!r} needs a column "
                f"that isn't available for the future keys ({exc}). Provide the "
                "future values of the business columns it uses through `X_df`."
            ) from exc
        raise
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(
                f"Feature node {node.name!r} returned a tuple of length {len(out)}. "
                "It must return either a dataframe or a (historic, future) pair."
            )
        return out[0], out[1]
    return out, None


def _node_input(
    current: DFType,
    node: FeatureNode,
    by_name: Dict[str, FeatureNode],
    business_cols: Sequence[str],
    id_col: str,
    time_col: str,
) -> DFType:
    dep_cols = [c for dep in node.dependencies for c in by_name[dep].outputs]
    return current[[id_col, time_col, *business_cols, *dep_cols]]


def _historic_features(
    df: DFType,
    nodes: Sequence[FeatureNode],
    order: Sequence[str],
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    business_cols: Sequence[str],
) -> Tuple[DFType, Dict[str, Optional[DataFrame]]]:
    by_name = {node.name: node for node in nodes}
    current = _take(df, np.arange(df.shape[0]))
    future_contribs: Dict[str, Optional[DataFrame]] = {}
    for name in order:
        node = by_name[name]
        node_input = _node_input(
            current, node, by_name, business_cols, id_col, time_col
        )
        historic, future = _call_node(
            node, node_input, freq=freq, h=h, id_col=id_col, time_col=time_col
        )
        aligned = _aligned_frame(
            node_input,
            historic,
            id_col,
            time_col,
            node.outputs,
            ctx=f"Feature node {name!r}",
        )
        for out in node.outputs:
            current = ufp.assign_columns(current, out, aligned[out])
        future_contribs[name] = future
    return current, future_contribs


def _future_features(
    future: DFType,
    nodes: Sequence[FeatureNode],
    order: Sequence[str],
    contribs: Optional[Dict[str, Optional[DataFrame]]],
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    business_cols: Sequence[str],
) -> DFType:
    by_name = {node.name: node for node in nodes}
    current = _take(future, np.arange(future.shape[0]))
    for name in order:
        node = by_name[name]
        node_input = _node_input(
            current, node, by_name, business_cols, id_col, time_col
        )
        fut = None if contribs is None else contribs.get(name)
        if fut is None:
            single, fut = _call_node(
                node,
                node_input,
                freq=freq,
                h=h,
                id_col=id_col,
                time_col=time_col,
                is_future=True,
            )
            if fut is None:
                # the node was evaluated on the future keys, so its only output
                # is the future contribution
                fut = single
        aligned = _aligned_frame(
            node_input,
            fut,
            id_col,
            time_col,
            node.outputs,
            ctx=f"The future part of feature node {name!r}",
        )
        for out in node.outputs:
            current = ufp.assign_columns(current, out, aligned[out])
    return current


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _restore_key_dtypes(
    future: DFType, df: DFType, id_col: str, time_col: str
) -> DFType:
    """Cast the keys of `future` back to the dtypes of the input dataframe."""
    if _is_pandas(future):
        for col in (id_col, time_col):
            dtype = df[col].dtype
            if future[col].dtype == dtype:
                continue
            if isinstance(dtype, pd.CategoricalDtype):
                values = pd.Categorical(future[col], dtype=dtype)
            elif getattr(dtype, "tz", None) is not None:
                values = future[col].dt.tz_localize(dtype.tz)
            else:
                values = future[col].astype(dtype)
            future = ufp.assign_columns(future, col, values)
    else:
        for col in (id_col, time_col):
            dtype = df.schema[col]
            if future.schema[col] == dtype:
                continue
            future = future.with_columns(pl.col(col).cast(dtype))
    return future


def _future_keys(
    df: DFType, freq: Union[str, int], h: int, id_col: str, time_col: str
) -> DFType:
    sorted_df = ufp.ensure_sorted(df, id_col, time_col)
    last_times = ufp.group_by_agg(
        sorted_df, id_col, {time_col: "max"}, maintain_order=True
    )
    last_times = ufp.sort(last_times, id_col)
    keys = ufp.make_future_dataframe(
        uids=last_times[id_col],
        last_times=last_times[time_col],
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    keys = ufp.drop_index_if_pandas(keys)
    return _restore_key_dtypes(keys, df, id_col, time_col)


def _prepare(
    df: DFType,
    nodes: Sequence[FeatureNode],
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    target_col: Optional[str],
) -> List[str]:
    if not isinstance(h, int) or isinstance(h, bool) or h < 0:
        raise ValueError("`h` must be a non-negative integer.")
    has_target = target_col is not None and target_col in df.columns
    validate_format(df, id_col, time_col, target_col if has_target else None)
    validate_freq(df[time_col], freq)
    _check_unique_keys(df, id_col, time_col, ctx="The input dataframe")
    # historic nodes get every original business column (including the target),
    # the future chain only gets the ones that aren't the target
    business_cols = [c for c in df.columns if c not in (id_col, time_col)]
    _check_output_collisions(
        nodes,
        [c for c in df.columns],
        id_col,
        time_col,
        target_col if has_target else None,
    )
    return business_cols


def _feature_columns(
    nodes: Sequence[FeatureNode], order: Sequence[str]
) -> List[str]:
    by_name = {node.name: node for node in nodes}
    return [c for name in order for c in by_name[name].outputs]


def _build_future(
    future: DFType,
    df: DFType,
    nodes: Sequence[FeatureNode],
    order: Sequence[str],
    contribs: Optional[Dict[str, Optional[DataFrame]]],
    freq: Union[str, int],
    h: int,
    id_col: str,
    time_col: str,
    target_col: Optional[str],
    business_cols: Sequence[str],
    X_df: Optional[DataFrame],
) -> DFType:
    feature_cols = _feature_columns(nodes, order)
    future_business: Sequence[str] = ()
    if X_df is not None:
        _check_backend(X_df, df, "`X_df`")
        validate_format(X_df, id_col, time_col, None)
        _check_unique_keys(X_df, id_col, time_col, ctx="`X_df`")
        if target_col is not None and target_col in X_df.columns:
            raise ValueError(
                f"`X_df` can't contain the target column ({target_col!r}). The "
                "future features and the predictions never see the target."
            )
        extra_cols = [c for c in X_df.columns if c not in (id_col, time_col)]
        unknown = [c for c in extra_cols if c not in business_cols]
        if unknown:
            raise ValueError(
                "`X_df` contains columns that aren't business columns of the "
                f"input dataframe: {unknown}."
            )
        future = _aligned_frame(
            future, X_df, id_col, time_col, extra_cols, ctx="`X_df`"
        )
        future_business = extra_cols
    future = ufp.sort(future, [id_col, time_col])
    future_features = _future_features(
        future=future,
        nodes=nodes,
        order=order,
        contribs=contribs,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        business_cols=future_business,
    )
    return future_features[[id_col, time_col, *feature_cols]]


def compute_features(
    df: DFType,
    nodes: Sequence[FeatureNode],
    freq: Union[str, int],
    h: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: Optional[str] = "y",
    X_df: Optional[DataFrame] = None,
) -> FeatureSet:
    """Compute the features of a dependency graph for history and forecast.

    Args:
        df (pandas or polars DataFrame): Dataframe with ids, times and the
            original business columns.
        nodes (sequence of FeatureNode): Nodes of the graph. Dependencies run
            before their dependents and every node runs at most once per pass,
            no matter how many other nodes use it.
        freq (str or int): Frequency of the data. Must be a valid pandas or
            polars offset alias, or an integer.
        h (int, optional): Forecast horizon. Defaults to 0, which skips the
            future features.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its
            values can be timestamps or integers. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target. It's never
            passed to the future features. Defaults to 'y'.
        X_df (pandas or polars DataFrame, optional): Future values of the
            business columns. Its keys must cover the future keys exactly.

    Returns:
        FeatureSet: The historic and future features aligned by the key columns.

    Raises:
        ValueError: If the graph has cycles, missing dependencies or conflicting
            output names, or if a node returns data whose keys don't match the
            expected ones.
    """
    nodes = list(nodes)
    order = _validate_graph(nodes, id_col, time_col)
    business_cols = _prepare(
        df=df,
        nodes=nodes,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    feature_cols = _feature_columns(nodes, order)
    historic, contribs = _historic_features(
        df=df,
        nodes=nodes,
        order=order,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        business_cols=business_cols,
    )
    features = historic[[id_col, time_col, *feature_cols]]
    if h == 0:
        return FeatureSet(
            features=features, future_features=_empty_rows(features)
        )
    future = _future_keys(df, freq=freq, h=h, id_col=id_col, time_col=time_col)
    future_features = _build_future(
        future=future,
        df=df,
        nodes=nodes,
        order=order,
        contribs=contribs,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        business_cols=business_cols,
        X_df=X_df,
    )
    return FeatureSet(features=features, future_features=future_features)


# ---------------------------------------------------------------------------
# model inputs
# ---------------------------------------------------------------------------
class FeatureMatrix(np.ndarray):
    """A numpy array that keeps track of the feature columns it holds.

    Models receive this instead of a plain array, so the exact columns that the
    graph produced are available through ``columns``. It also supports indexing
    by column name.

    Example:
        >>> X = pipe.predict().future_features
        >>> X.columns
        ['sin1_7', 'cos1_7']
        >>> X["sin1_7"]
    """

    columns: Optional[List[str]]

    def __new__(cls, values: Any, columns: Sequence[str]) -> "FeatureMatrix":
        obj = np.asarray(values).view(cls)
        obj.columns = list(columns)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is not None:
            self.columns = getattr(obj, "columns", None)

    def __getitem__(self, key: Any) -> Any:
        columns = getattr(self, "columns", None)
        if columns is not None and isinstance(key, str):
            return np.asarray(self)[:, columns.index(key)]
        if (
            columns is not None
            and isinstance(key, (list, tuple))
            and len(key) > 0
            and all(isinstance(k, str) for k in key)
        ):
            idxs = [columns.index(k) for k in key]
            return FeatureMatrix(np.asarray(self)[:, idxs], list(key))
        return super().__getitem__(key)

    def to_numpy(self) -> np.ndarray:
        """Return the values as a plain numpy array."""
        return np.asarray(self)

    def to_pandas(self) -> pd.DataFrame:
        """Return the values as a pandas dataframe with the feature names."""
        columns = getattr(self, "columns", None) or []
        return pd.DataFrame(np.asarray(self), columns=columns)


def _feature_matrix(df: DataFrame, columns: Sequence[str]) -> FeatureMatrix:
    columns = list(columns)
    if not columns:
        raise ValueError("The feature graph didn't produce any model inputs.")
    if _is_pandas(df):
        non_numeric = [
            c for c in columns if not pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
    else:
        non_numeric = [c for c in columns if not df[c].dtype.is_numeric()]
    if non_numeric:
        raise ValueError(
            "The model inputs must be numeric, but the following feature columns "
            f"aren't: {non_numeric}."
        )
    return FeatureMatrix(ufp.to_numpy(df[columns]), columns)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
class FeaturePipeline:
    """A forecasting pipeline built from a graph of feature nodes.

    Each node declares its dependencies and the columns it produces. Nodes only
    see the original business columns plus the outputs they declared, they run
    after their dependencies and a shared node runs only once per pass. Both the
    historic and the future results are aligned by the key columns, so nodes can
    return their rows in any order.

    The features produced by the graph are the exact inputs of the model:
    ``fit`` receives the historic features together with the target, while the
    future features and ``predict`` never see it.

    Args:
        nodes (sequence of FeatureNode): Nodes of the graph.
        freq (str or int): Frequency of the data. Must be a valid pandas or
            polars offset alias, or an integer.
        h (int, optional): Forecast horizon. Defaults to 1.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its
            values can be timestamps or integers. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target.
            Defaults to 'y'.
        model_name (str, optional): Name of the column with the predictions.
            Defaults to 'model'.
        model_names (sequence of str, optional): Names of the columns when the
            model returns more than one prediction per row.

    Example:
        >>> pipe = FeaturePipeline(nodes=[weekly, linear], freq="D", h=7)
        >>> pipe.fit(train_df, model=model)
        >>> result = pipe.predict()
        >>> result.future_features.head()
        >>> result.forecast.head()
    """

    def __init__(
        self,
        nodes: Sequence[FeatureNode],
        freq: Union[str, int],
        h: int = 1,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        model_name: str = "model",
        model_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(h, int) or isinstance(h, bool) or h < 1:
            raise ValueError("`h` must be a positive integer for a forecast pipeline.")
        self.nodes = list(nodes)
        self._order = _validate_graph(self.nodes, id_col, time_col)
        self.freq = freq
        self.h = h
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.model_name = model_name
        self.model_names = None if model_names is None else list(model_names)
        self._train_df: Optional[DataFrame] = None
        self.model_: Any = None
        self.features_: Optional[DataFrame] = None
        self.future_features_: Optional[DataFrame] = None

    @property
    def feature_names(self) -> List[str]:
        """Columns produced by the graph, in graph order."""
        return _feature_columns(self.nodes, self._order)

    def __repr__(self) -> str:
        return (
            f"FeaturePipeline(nodes={[n.name for n in self.nodes]!r}, "
            f"freq={self.freq!r}, h={self.h!r})"
        )

    def transform(
        self, df: DFType, X_df: Optional[DataFrame] = None
    ) -> FeatureSet:
        """Compute the features of the graph without fitting a model.

        Args:
            df (pandas or polars DataFrame): Historic data with the business
                columns.
            X_df (pandas or polars DataFrame, optional): Future values of the
                business columns.

        Returns:
            FeatureSet: Historic and future features aligned by the key columns.
        """
        return compute_features(
            df=df,
            nodes=self.nodes,
            freq=self.freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            X_df=X_df,
        )

    def _historic(self, df: DFType) -> Tuple[DataFrame, Dict[str, Optional[DataFrame]]]:
        business_cols = _prepare(
            df=df,
            nodes=self.nodes,
            freq=self.freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        historic, contribs = _historic_features(
            df=df,
            nodes=self.nodes,
            order=self._order,
            freq=self.freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
            business_cols=business_cols,
        )
        features = historic[[self.id_col, self.time_col, *self.feature_names]]
        return features, contribs

    def fit(self, df: DFType, model: Any = None) -> "FeaturePipeline":
        """Fit the model on the features produced by the graph.

        Args:
            df (pandas or polars DataFrame): Historic data. It must contain the
                key columns, the target and the business columns used by the
                nodes.
            model (object, optional): Object with ``fit(X, y)`` and
                ``predict(X)`` methods. ``X`` is a :class:`FeatureMatrix` whose
                columns and values are exactly the outputs of the graph, and
                ``y`` is a numpy array with the target.

        Returns:
            FeaturePipeline: The fitted pipeline.
        """
        if model is None:
            raise ValueError(
                "`fit` requires a `model` with `fit` and `predict` methods."
            )
        for method in ("fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"The model must implement a `{method}` method.")
        if self.target_col not in df.columns:
            raise ValueError(
                f"The target column ({self.target_col!r}) is missing from the "
                "input dataframe."
            )
        features, contribs = self._historic(df)
        y_frame = _aligned_frame(
            features,
            df,
            self.id_col,
            self.time_col,
            [self.target_col],
            ctx=f"The target column ({self.target_col!r})",
        )
        X = _feature_matrix(features, self.feature_names)
        y = ufp.to_numpy(y_frame[[self.target_col]])[:, 0]
        model.fit(X, y)
        self.model_ = model
        self.features_ = features
        self.future_features_ = None
        self._train_df = df
        self._contribs = contribs
        return self

    def predict(self, X_df: Optional[DataFrame] = None) -> ForecastResult:
        """Forecast the future keys with the fitted model.

        Args:
            X_df (pandas or polars DataFrame, optional): Future values of the
                business columns. Its keys must cover the future keys exactly.
                It can't contain the target column.

        Returns:
            ForecastResult: The historic features, the future features that were
                handed to the model and the predictions.

        Raises:
            ValueError: If the pipeline isn't fitted, or if the features or the
                predictions don't cover the future keys exactly. In those cases
                no partial forecast is returned.
        """
        if self.model_ is None or self._train_df is None:
            raise ValueError("You must call `fit` before `predict`.")
        df = self._train_df
        business_cols = _prepare(
            df=df,
            nodes=self.nodes,
            freq=self.freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        if X_df is not None:
            _check_backend(X_df, df, "`X_df`")
        future = _future_keys(
            df, freq=self.freq, h=self.h, id_col=self.id_col, time_col=self.time_col
        )
        future_features = _build_future(
            future=future,
            df=df,
            nodes=self.nodes,
            order=self._order,
            contribs=self._contribs,
            freq=self.freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            business_cols=business_cols,
            X_df=X_df,
        )
        self.future_features_ = future_features
        X = _feature_matrix(future_features, self.feature_names)
        preds = np.asarray(self.model_.predict(X))
        n_future = future_features.shape[0]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        elif preds.ndim != 2:
            raise ValueError(
                "The model must return a one or two dimensional array, got an "
                f"array with {preds.ndim} dimensions."
            )
        if preds.shape[0] != n_future:
            raise ValueError(
                f"The model returned {preds.shape[0]} predictions but there are "
                f"{n_future} future keys."
            )
        names = self._prediction_names(preds.shape[1])
        forecast = _take(
            future_features[[self.id_col, self.time_col]], np.arange(n_future)
        )
        for i, name in enumerate(names):
            forecast = ufp.assign_columns(forecast, name, preds[:, i])
        _check_unique_keys(
            forecast, self.id_col, self.time_col, ctx="The forecast"
        )
        _check_no_missing(forecast, names)
        return ForecastResult(
            features=self.features_,
            future_features=future_features,
            forecast=forecast,
        )

    def _prediction_names(self, n_cols: int) -> List[str]:
        if n_cols == 1:
            names = [self.model_name]
        elif self.model_names is not None:
            if len(self.model_names) != n_cols:
                raise ValueError(
                    f"`model_names` has {len(self.model_names)} names but the model "
                    f"returned {n_cols} columns."
                )
            names = list(self.model_names)
        else:
            names = [f"{self.model_name}_{i}" for i in range(n_cols)]
        clashes = [c for c in names if c in (self.id_col, self.time_col)]
        if clashes:
            raise ValueError(
                f"The prediction columns can't use the key columns' names: {clashes}."
            )
        return names


def _check_no_missing(forecast: DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        missing = ufp.is_nan_or_none(forecast[col])
        if bool(np.asarray(ufp.to_numpy(missing.to_frame())).any()):
            raise ValueError(
                f"The forecast contains null values in column {col!r}. A partial "
                "forecast is not returned."
            )
