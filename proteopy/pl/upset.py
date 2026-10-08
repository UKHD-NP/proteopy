"""UpSet plots of feature membership across annotation categories.

Importing this module also installs compatibility patches for
``upsetplot`` 0.9.0 (the version range pinned in ``pyproject.toml``),
which is otherwise unusable on current pandas/numpy:

1. Per-dot style defaults of the UpSet matrix. upsetplot fills them
   with ``styles["linewidth"].fillna(1, inplace=True)`` and three
   sibling calls; under pandas' Copy-on-Write (mandatory in pandas 3)
   these fills silently do nothing, and every ``UpSet.plot()`` fails in
   ``Axes.scatter`` with ``ValueError: Invalid RGBA argument: nan``.
   The patch restores the defaults where the scatter call consumes
   them and silences the accompanying chained-assignment warning
   (``FutureWarning`` on pandas 2, ``ChainedAssignmentError`` on
   pandas 3). It is harmless on pandas 2.
2. Aggregation of a single-category input, which loses its
   ``MultiIndex`` under pandas' groupby and then fails in
   ``Series.reorder_levels``.
3. Count labels, positioned with a one-element array that numpy >= 2.4
   refuses to convert to a scalar when the figure is drawn.
4. The totals axis of an input whose categories are all empty, which
   triggers matplotlib's "identical xlims" ``UserWarning``; matplotlib
   expands the limits itself, so the warning is silenced.

Each patch is installed at most once. Remove them once ``upsetplot``
ships a release fixing these defects.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import sparse
from upsetplot import UpSet, reformat

from proteopy.utils.anndata import check_proteodata

_NO_CATEGORY_LABEL = "No category"
_COUNT_NAME = "n_features"


# -- upsetplot 0.9.0 compatibility patches (see the module docstring)

_DOT_STYLE_PATCHED_FLAG = "_proteopy_pandas3_dot_style_patch"
_AGG_PATCHED_FLAG = "_proteopy_single_category_agg_patch"
_LABEL_PATCHED_FLAG = "_proteopy_numpy_label_position_patch"
_TOTALS_PATCHED_FLAG = "_proteopy_empty_totals_xlim_patch"

# Defaults that upsetplot intends to fill in, per scatter keyword.
_LITERAL_DOT_DEFAULTS = {
    "linewidths": 1,
    "linestyles": "solid",
}


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _filled(values, defaults) -> list:
    """Replace missing entries of ``values`` using ``defaults``.

    ``defaults`` is either a single value used for every gap, or a
    sequence of the same length supplying a per-entry replacement.
    """
    items = list(values)
    if not any(_is_missing(item) for item in items):
        return items
    if isinstance(defaults, list) and len(defaults) == len(items):
        per_entry = defaults
    else:
        per_entry = [defaults] * len(items)
    return [
        per_entry[i] if _is_missing(item) else item
        for i, item in enumerate(items)
    ]


def _restore_dot_style_defaults(kwargs: dict, facecolor) -> None:
    for keyword, default in _LITERAL_DOT_DEFAULTS.items():
        if keyword in kwargs:
            kwargs[keyword] = _filled(kwargs[keyword], default)

    faces = None
    if "facecolors" in kwargs:
        faces = _filled(kwargs["facecolors"], facecolor)
        kwargs["facecolors"] = faces
    if "edgecolors" in kwargs:
        fallback = faces if faces is not None else facecolor
        kwargs["edgecolors"] = _filled(kwargs["edgecolors"], fallback)


def _patch_dot_styles() -> None:
    """Restore per-dot style defaults in ``UpSet.plot_matrix``, once."""
    if getattr(UpSet, _DOT_STYLE_PATCHED_FLAG, False):
        return

    original_plot_matrix = UpSet.plot_matrix

    def plot_matrix(self, ax):
        real_scatter = matplotlib.axes.Axes.scatter

        def scatter(axes, *args, **kwargs):
            _restore_dot_style_defaults(kwargs, self._facecolor)
            return real_scatter(axes, *args, **kwargs)

        matplotlib.axes.Axes.scatter = scatter
        try:
            with warnings.catch_warnings():
                # pandas 2: FutureWarning; pandas 3: ChainedAssignmentError
                warnings.filterwarnings(
                    "ignore",
                    message=("A value is (trying to be|being) set on a copy"),
                    category=Warning,
                )
                return original_plot_matrix(self, ax)
        finally:
            matplotlib.axes.Axes.scatter = real_scatter

    plot_matrix.__doc__ = original_plot_matrix.__doc__
    UpSet.plot_matrix = plot_matrix
    setattr(UpSet, _DOT_STYLE_PATCHED_FLAG, True)


def _patch_single_category_aggregation() -> None:
    """Keep a one-level ``MultiIndex`` through aggregation, once."""
    if getattr(reformat, _AGG_PATCHED_FLAG, False):
        return

    original_aggregate_data = reformat._aggregate_data

    def _aggregate_data(df, subset_size, sum_over):
        data, aggregated = original_aggregate_data(
            df,
            subset_size,
            sum_over,
        )
        if (
            isinstance(data.index, pd.MultiIndex)
            and data.index.nlevels == 1
            and not isinstance(aggregated.index, pd.MultiIndex)
        ):
            aggregated.index = pd.MultiIndex.from_arrays(
                [aggregated.index],
                names=[aggregated.index.name],
            )
        return data, aggregated

    _aggregate_data.__doc__ = original_aggregate_data.__doc__
    reformat._aggregate_data = _aggregate_data
    setattr(reformat, _AGG_PATCHED_FLAG, True)


def _as_scalar(value):
    """Unwrap a one-element array; return anything else unchanged."""
    array = np.asarray(value)
    if array.size == 1:
        return array.item()
    return value


def _patch_label_positions() -> None:
    """Make count-label positions scalars after ``_label_sizes``, once."""
    if getattr(UpSet, _LABEL_PATCHED_FLAG, False):
        return

    original_label_sizes = UpSet._label_sizes

    def _label_sizes(self, ax, rects, where):
        n_texts_before = len(ax.texts)
        # upsetplot's ``_label_sizes`` returns nothing.
        original_label_sizes(self, ax, rects, where)
        for text in list(ax.texts)[n_texts_before:]:
            x, y = text.get_position()
            text.set_position((_as_scalar(x), _as_scalar(y)))

    _label_sizes.__doc__ = original_label_sizes.__doc__
    UpSet._label_sizes = _label_sizes
    setattr(UpSet, _LABEL_PATCHED_FLAG, True)


def _patch_empty_totals() -> None:
    """Silence the singular-xlim warning of all-zero totals, once."""
    if getattr(UpSet, _TOTALS_PATCHED_FLAG, False):
        return

    original_plot_totals = UpSet.plot_totals

    def plot_totals(self, ax):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Attempting to set identical low and high xlims",
                category=UserWarning,
            )
            return original_plot_totals(self, ax)

    plot_totals.__doc__ = original_plot_totals.__doc__
    UpSet.plot_totals = plot_totals
    setattr(UpSet, _TOTALS_PATCHED_FLAG, True)


_patch_dot_styles()
_patch_single_category_aggregation()
_patch_label_positions()
_patch_empty_totals()


# -- var_detected_by_cat_upset


def _check_flag(value, name: str) -> None:
    """Reject anything that is not an exact Python ``bool``."""
    if not isinstance(value, bool):
        raise TypeError(
            f"`{name}` must be a bool, got " f"{type(value).__name__}."
        )


def _resolve_threshold(
    min_count: int | None,
    min_fraction: float | None,
) -> tuple[str, int | float]:
    """Validate the thresholds and return the active one."""
    if min_count is not None and (
        isinstance(min_count, bool) or not isinstance(min_count, int)
    ):
        raise TypeError(
            "`min_count` must be a non-boolean int or None, got "
            f"{type(min_count).__name__}."
        )
    if min_fraction is not None and (
        isinstance(min_fraction, bool)
        or not isinstance(min_fraction, (int, float))
    ):
        raise TypeError(
            "`min_fraction` must be a non-boolean number or None, "
            f"got {type(min_fraction).__name__}."
        )

    if min_count is not None and min_fraction is not None:
        raise ValueError(
            "`min_count` and `min_fraction` are mutually exclusive. "
            "Provide one or neither."
        )

    if min_count is not None:
        if min_count < 0:
            raise ValueError("`min_count` must be greater than or equal to 0.")
        return "min_count", min_count

    if min_fraction is not None:
        # A chained comparison rather than math.isfinite: it also
        # rejects NaN and infinities, and cannot overflow on huge ints.
        if not 0 <= min_fraction <= 1:
            raise ValueError(
                "`min_fraction` must be a finite number between 0 " "and 1."
            )
        return "min_fraction", min_fraction

    return "min_count", 1


def _validate_upset_args(
    adata: ad.AnnData,
    cat_key,
    min_count,
    min_fraction,
    zero_to_na,
    print_stats,
    verbose,
    show,
    save,
    sort_by,
) -> tuple[str, int | float]:
    """Validate all inputs and return the active threshold."""
    check_proteodata(adata)

    if not isinstance(sort_by, str):
        raise TypeError(
            f"`sort_by` must be a str, got {type(sort_by).__name__}."
        )
    if sort_by not in ("degree", "cardinality", "-degree", "-cardinality"):
        raise ValueError(
            "`sort_by` must be one of 'degree', 'cardinality', "
            "'-degree', '-cardinality'."
        )

    for value, name in (
        (zero_to_na, "zero_to_na"),
        (print_stats, "print_stats"),
        (verbose, "verbose"),
        (show, "show"),
    ):
        _check_flag(value, name)

    if save is not None and not isinstance(save, (str, Path)):
        raise TypeError(
            "`save` must be a str, Path or None, got "
            f"{type(save).__name__}."
        )

    if not isinstance(cat_key, str):
        raise TypeError(
            "`cat_key` must be a str, got " f"{type(cat_key).__name__}."
        )
    if cat_key == "":
        raise ValueError("`cat_key` must not be an empty string.")

    threshold = _resolve_threshold(min_count, min_fraction)

    if cat_key not in adata.obs.columns:
        raise KeyError(f"'{cat_key}' is not a column of `adata.obs`.")

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            "Cannot build an UpSet plot from an AnnData with an "
            "empty observation or variable axis."
        )

    if adata.obs[cat_key].isna().any():
        raise ValueError(
            f"`adata.obs['{cat_key}']` must not contain missing " "values."
        )

    return threshold


def _resolve_categories(
    series: pd.Series,
) -> tuple[list[str], list[np.ndarray]]:
    """Return the category names in spec order and their obs masks.

    Categorical columns keep their category order, including
    categories without observations; any other dtype is ordered by the
    lexicographic order of the ``str``-coerced unique values.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        raw_values = list(series.cat.categories)
    else:
        raw_values = list(pd.unique(series))
        raw_values.sort(key=str)

    names = [str(value) for value in raw_values]
    if len(set(names)) != len(names):
        raise ValueError(
            "Category values collide after coercion to str; make "
            "the categories unique as strings."
        )

    coerced = series.astype(object).map(str).to_numpy()
    masks = [coerced == name for name in names]
    return names, masks


def _detection_matrix(
    adata: ad.AnnData,
    zero_to_na: bool,
) -> np.ndarray:
    """Return the boolean detection matrix of ``adata.X``."""
    matrix = adata.X
    if sparse.issparse(matrix):
        warnings.warn(
            "`adata.X` is sparse and is being densified to build "
            "the UpSet plot.",
            UserWarning,
            stacklevel=2,
        )
        matrix = matrix.toarray()

    values = np.asarray(matrix)
    if values.dtype.kind not in "fiu":
        values = values.astype(float)

    detected = ~np.isnan(values)
    if zero_to_na:
        detected &= values != 0
    return detected


def _membership_matrix(
    detected: np.ndarray,
    masks: list[np.ndarray],
    threshold_name: str,
    threshold_value: int | float,
) -> np.ndarray:
    """Return the (features x categories) boolean membership matrix."""
    n_vars = detected.shape[1]
    membership = np.zeros((n_vars, len(masks)), dtype=bool)

    for index, mask in enumerate(masks):
        n_obs_cat = int(mask.sum())
        if n_obs_cat == 0:
            # An empty category has no members, for any threshold.
            continue
        counts = detected[mask, :].sum(axis=0)
        if threshold_name == "min_count":
            membership[:, index] = counts >= threshold_value
        else:
            membership[:, index] = counts / n_obs_cat >= threshold_value
    return membership


def _intersection_counts(
    membership: np.ndarray,
    names: list[str],
) -> pd.Series:
    """Count features per membership vector, plus ``No category``."""
    counts: dict[tuple[bool, ...], int] = {}
    for row in membership:
        key = tuple(bool(value) for value in row)
        counts[key] = counts.get(key, 0) + 1

    all_false = (False,) * len(names)
    if all_false not in counts:
        counts[all_false] = 0

    index = pd.MultiIndex.from_tuples(
        list(counts.keys()),
        names=names,
    )
    return pd.Series(
        list(counts.values()),
        index=index,
        dtype=int,
        name=_COUNT_NAME,
    )


def _print_stats_df(df: pd.DataFrame) -> None:
    """Print a DataFrame with one-decimal formatting."""
    print(df.to_string(index=False, float_format="%.1f"))


def _global_stats_df(counts: pd.Series) -> pd.DataFrame:
    """One-row summary of the intersection counts."""
    return pd.DataFrame(
        {
            "count": [counts.count()],
            "mean": [counts.mean()],
            "median": [counts.median()],
            "std": [counts.std()],
            "min": [counts.min()],
            "max": [counts.max()],
        }
    )


def _intersection_label(
    vector: tuple[bool, ...],
    names: list[str],
) -> str:
    """Human-readable label of one membership vector."""
    members = [name for name, flag in zip(names, vector) if flag]
    if not members:
        return _NO_CATEGORY_LABEL
    return " & ".join(members)


def _intersections_df(
    counts: pd.Series,
    names: list[str],
) -> pd.DataFrame:
    """One row per intersection, sorted by size then label."""
    rows = []
    for vector, count in counts.items():
        vector = tuple(vector)
        rows.append([*vector, int(count), _intersection_label(vector, names)])

    # Built and sorted by position, then named: a category may itself
    # be named "n_features" or "label", so the headers can repeat.
    count_pos = len(names)
    df = pd.DataFrame(rows).sort_values(
        [count_pos, count_pos + 1],
        ascending=[False, True],
        kind="mergesort",
    )
    df.columns = [*names, _COUNT_NAME, "label"]
    return df


def _per_category_df(
    membership: np.ndarray,
    names: list[str],
    cat_key: str,
    n_vars: int,
) -> pd.DataFrame:
    """One row per category with its member-feature count."""
    n_features = membership.sum(axis=0).astype(int)
    # Built by position: ``cat_key`` may be "n_features" or "percent".
    df = pd.DataFrame(
        {
            0: names,
            1: n_features,
            2: 100 * n_features / n_vars,
        }
    )
    df.columns = [cat_key, _COUNT_NAME, "percent"]
    return df


def _print_upset_stats(
    counts: pd.Series,
    membership: np.ndarray,
    names: list[str],
    cat_key: str,
    n_vars: int,
) -> None:
    """Print the three tables underlying the plot."""
    print("Global:")
    _print_stats_df(_global_stats_df(counts))
    print("Intersections:")
    _print_stats_df(_intersections_df(counts, names))
    print(f"Per {cat_key}:")
    _print_stats_df(_per_category_df(membership, names, cat_key, n_vars))


def _print_verbose_report(
    cat_key: str,
    threshold_name: str,
    threshold_value: int | float,
    n_vars: int,
    n_categories: int,
) -> None:
    """Print a short report about the input of the plot."""
    print(
        f"Using the .X matrix.\n"
        f"Categories from .obs['{cat_key}'].\n"
        f"Threshold: {threshold_name} = {str(threshold_value)}\n"
        f"Features: {n_vars}\n"
        f"Categories: {n_categories}"
    )


def _plot_upset(counts: pd.Series, sort_by: str) -> dict[str, Axes]:
    """Render the UpSet plot of the intersection counts."""
    upset = UpSet(
        counts,
        subset_size="sum",
        sort_by=sort_by,
        sort_categories_by="input",
        show_counts=True,
        include_empty_subsets=False,
    )
    return upset.plot()


def var_detected_by_cat_upset(
    adata: ad.AnnData,
    cat_key: str,
    min_count: int | None = None,
    min_fraction: float | None = None,
    zero_to_na: bool = False,
    print_stats: bool = False,
    verbose: bool = False,
    show: bool = True,
    save: str | Path | None = None,
    *,
    sort_by: str = "degree",
) -> dict[str, Axes]:
    """Plot feature detection overlaps across observation categories.

    A feature belongs to a category when it meets ``min_count`` or
    ``min_fraction``. Missing values do not count as detections; zeros
    count unless ``zero_to_na=True``. Features belonging to no category
    form the ``"No category"`` intersection.

    Parameters
    ----------
    adata : AnnData
        ProteoPy AnnData (peptide- or protein-level). Intensities are
        read from ``adata.X`` only.
    cat_key : str
        Column in ``adata.obs`` defining the categories.
        Categories follow categorical order, otherwise lexicographic
        order after conversion to strings. Use an ordered
        :class:`pandas.Categorical` to customize the order.
    min_count : int | None
        Minimum detections per category, as a nonnegative integer.
        If neither threshold is supplied, one detection suffices.
    min_fraction : float | None
        Minimum fraction of observations with a detection per category,
        between 0 and 1. Supply at most one of ``min_count`` and
        ``min_fraction``.
    zero_to_na : bool
        If True, zeros in ``.X`` count as missing.
    print_stats : bool
        If True, print global, intersection, and per-category
        statistics.
    verbose : bool
        If True, print status messages about the input.
    show : bool
        Call ``plt.show()`` at the end.
    save : str | Path | None
        Path to save the figure to; None skips saving.
    sort_by : str
        Order intersection bars by ``"degree"`` (fewest overlapping
        categories first) or ``"cardinality"`` (largest feature counts
        first). ``"-degree"`` and ``"-cardinality"`` reverse these
        orders. Ties follow the plotting library's ordering.

    Returns
    -------
    dict[str, Axes]
        The axes returned by ``upsetplot.UpSet.plot()``, with keys
        ``"matrix"``, ``"intersections"``, ``"totals"`` and
        ``"shading"``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    KeyError
        If ``cat_key`` is not a column of ``adata.obs``.
    ValueError
        If thresholds are invalid or both supplied, ``cat_key`` is
        empty, either data axis is empty, or category labels are
        missing or collide after conversion to strings, or ``sort_by``
        is unsupported.

    Warns
    -----
    UserWarning
        If ``adata.X`` is sparse and is densified.

    Examples
    --------
    P2 is measured in one liver sample; P3 is never measured.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> samples = ["S1", "S2", "S3", "S4"]
    >>> proteins = ["P1", "P2", "P3"]
    >>> adata = ad.AnnData(
    ...     X=np.array([
    ...         [5.0, 2.0, np.nan], [4.0, np.nan, np.nan],
    ...         [6.0, np.nan, np.nan], [5.0, np.nan, np.nan],
    ...     ]),
    ...     obs=pd.DataFrame(
    ...         {"sample_id": samples,
    ...          "tissue": ["liver", "liver", "lung", "lung"]},
    ...         index=samples,
    ...     ),
    ...     var=pd.DataFrame({"protein_id": proteins}, index=proteins),
    ... )
    >>> axes = pr.pl.var_detected_by_cat_upset(
    ...     adata, cat_key="tissue", show=False,
    ... )
    >>> sorted(axes)
    ['intersections', 'matrix', 'shading', 'totals']

    Require detection in every sample of a tissue:

    >>> axes = pr.pl.var_detected_by_cat_upset(
    ...     adata,
    ...     cat_key="tissue",
    ...     min_fraction=1.0,
    ...     show=False,
    ... )

    Show the largest intersections first:

    >>> axes = pr.pl.var_detected_by_cat_upset(
    ...     adata, cat_key="tissue", sort_by="cardinality", show=False,
    ... )
    """
    # -- Validate inputs
    threshold_name, threshold_value = _validate_upset_args(
        adata,
        cat_key,
        min_count,
        min_fraction,
        zero_to_na,
        print_stats,
        verbose,
        show,
        save,
        sort_by,
    )

    # -- Derive categories and feature memberships
    names, masks = _resolve_categories(adata.obs[cat_key])
    detected = _detection_matrix(adata, zero_to_na)
    membership = _membership_matrix(
        detected,
        masks,
        threshold_name,
        threshold_value,
    )
    counts = _intersection_counts(membership, names)

    # -- Report
    if verbose:
        _print_verbose_report(
            cat_key,
            threshold_name,
            threshold_value,
            adata.n_vars,
            len(names),
        )
    if print_stats:
        _print_upset_stats(
            counts,
            membership,
            names,
            cat_key,
            adata.n_vars,
        )

    # -- Plot
    axes = _plot_upset(counts, sort_by)

    if save is not None:
        axes["matrix"].figure.savefig(save)
    if show:
        plt.show()
    return axes
