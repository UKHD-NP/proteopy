from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from anndata import AnnData

from proteopy.utils.anndata import check_proteodata


def tests(adata: AnnData) -> pd.DataFrame:
    """
    Retrieve a summary of all differential abundance tests stored in ``.varm``.

    Scans the ``.varm`` slots of the AnnData object for statistical test results
    and returns a DataFrame summarizing the tests performed.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data object containing differential abundance results in
        ``.varm``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - ``key``: The ``.varm`` slot name.
        - ``key_group``: String identifier for the test group in format
          ``"<test_type>_<design_mode>"`` or ``"<test_type>_<design_mode>_<layer>"``
          if a layer was used.
        - ``test_type``: The statistical test type (e.g., ``"ttest_two_sample"``).
        - ``design``: Underscore-separated design identifier (e.g., ``"A_vs_rest"``).
        - ``design_label``: Human-readable description of what the test compares.
        - ``design_mode``: Either ``"one_vs_rest"`` or ``"one_vs_one"``.
        - ``layer``: The layer used for the test, or ``None`` if ``.X`` was used.

    Examples
    --------
    >>> import proteopy as pp
    >>> # After running differential abundance tests
    >>> tests_df = pp.get.tests(adata)
    >>> tests_df
              key          key_group  ...  design_mode
    0  welch_A_vs_rest  welch_one_vs_rest  ...  one_vs_rest
    1  welch_B_vs_rest  welch_one_vs_rest  ...  one_vs_rest
    """
    from proteopy.utils.parsers import parse_stat_test_varm_slot

    check_proteodata(adata)

    records = []
    for key in adata.varm.keys():
        try:
            parsed = parse_stat_test_varm_slot(key, adata=adata)
            design = parsed["design"]
            design_mode = "one_vs_rest" if design.endswith("_vs_rest") else "one_vs_one"
            records.append({
                "key": key,
                "test_type": parsed["test_type"],
                "design": design,
                "design_label": parsed["design_label"],
                "design_mode": design_mode,
                "layer": parsed["layer"],
            })
        except ValueError:
            # Not a stat-test slot, skip
            continue

    if not records:
        return pd.DataFrame(
            columns=["key", "key_group", "test_type", "design", "design_label",
                     "design_mode", "layer"]
        )

    df = pd.DataFrame(records)

    # Build key_group string: "<test_type>_<design_mode>" or
    # "<test_type>_<design_mode>_<layer>" if layer is not None
    def build_key_group(row):
        parts = [row["test_type"], row["design_mode"]]
        if row["layer"] is not None:
            parts.append(row["layer"])
        return "_".join(parts)

    df["key_group"] = df.apply(build_key_group, axis=1)
    df = df[["key", "key_group", "test_type", "design", "design_label",
             "design_mode", "layer"]]

    return df
