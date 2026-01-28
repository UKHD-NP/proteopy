import re
import warnings
from typing import Dict, Optional, List

import anndata as ad
import numpy as np
import pandas as pd

from proteopy.utils.string import sanitize_string

STAT_TEST_METHOD_LABELS = {
    "ttest_two_sample": "Two-sample t-test",
    "welch": "Welch's t-test",
}


def parse_tumor_subclass(df: pd.DataFrame, col: str = "tumor_class") -> pd.DataFrame:
    """
    Parse a less-structured tumor_class column into:
      - main_tumor_type
      - genetic_markers
      - subclass
      - subtype
      - rest

    Algorithm:
      - While string has multiple parts (split on commas and the word 'and'):
          - take the last part as the query chunk
          - extract genetic markers, subclass, subtype using regex
          - any leftover in that chunk goes to 'rest'
          - continue with the remaining parts
      - When one part remains:
          - still perform pattern matching on it
          - the left-over (after removing matched parts) is main_tumor_type

    The function leaves the original column intact and appends parsed columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name containing the tumor subclass annotation.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        main_tumor_type, genetic_markers, subclass, subtype, rest.
    """
    df = df.copy()
    df.index.name = None


    # Compile patterns once
    # Genetic markers to capture (exact phrases)
    genetic_marker_patterns = [
        re.compile(r"\bIDH-(?:mutant|wildtype)\b", re.IGNORECASE),
        re.compile(r"\b1p/19q-codeleted\b", re.IGNORECASE),
        re.compile(r"\bPLAGL1-fused\b", re.IGNORECASE),
        re.compile(r"\bZFTA fusion-positive\b", re.IGNORECASE),
    ]

    # subclass and subtype helpers
    subclass_bracket_pat = re.compile(r"\[([^\]]*subclass[^\]]*)\]", re.IGNORECASE)
    subclass_pat = re.compile(r"\bsubclass\b[^\),;\]]*", re.IGNORECASE)

    subtype_bracket_pat = re.compile(r"\[([^\]]*subtype[^\]]*)\]", re.IGNORECASE)
    # 'subtype ...'
    subtype_after_pat = re.compile(r"\bsubtype\b[^\),;\]]*", re.IGNORECASE)
    # '... subtype' (capture up to 3 words before subtype)
    subtype_before_pat = re.compile(r"(?:\b[\w/-]+\s+){1,3}\bsubtype\b", re.IGNORECASE)

    # Splitter on comma or the word 'and'
    splitter = re.compile(r"\s*,\s*|\s+\band\b\s+", re.IGNORECASE)

    def strip_wrappers(s: str) -> str:
        s = s.strip()
        # remove enclosing brackets or parentheses only if they enclose the whole chunk
        if len(s) >= 2 and ((s[0] == "[" and s[-1] == "]") or (s[0] == "(" and s[-1] == ")")):
            s = s[1:-1].strip()
        return s.strip(" ,;")

    def dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            key = x.lower()
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    def normalize_case(val: str) -> str:
        # Return as-is except normalize common capitalization in markers
        # Keep original chunk case for readability
        return val.strip()

    def parse_one(value: Optional[str]) -> Dict[str, Optional[str]]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return {
                "main_tumor_type": None,
                "genetic_markers": None,
                "subclass": None,
                "subtype": None,
                "rest": None,
            }

        remaining = str(value).strip()
        markers: List[str] = []
        subclass_val: Optional[str] = None
        subtype_val: Optional[str] = None
        rest_parts: List[str] = []
        main_tumor_type: Optional[str] = None

        while True:
            # Split into tokens
            tokens = [t for t in splitter.split(remaining) if t != ""]
            if not tokens:
                main_tumor_type = None
                break

            if len(tokens) == 1:
                chunk = tokens[0]
                remaining_next = None
            else:
                chunk = tokens[-1]
                remaining_next = ", ".join(tokens[:-1])

            chunk_work = chunk
            consumed_spans: List[tuple] = []

            def record_span(m):
                if m:
                    consumed_spans.append(m.span())

            # 1) subclass (first bracketed, then unbracketed)
            if subclass_val is None:
                m = subclass_bracket_pat.search(chunk_work)
                if m:
                    subclass_val = normalize_case(m.group(1))
                    record_span(m)
            if subclass_val is None:
                m = subclass_pat.search(chunk_work)
                if m:
                    subclass_val = normalize_case(m.group(0))
                    record_span(m)

            # 2) subtype (first bracketed, then 'subtype ...', then '... subtype')
            if subtype_val is None:
                m = subtype_bracket_pat.search(chunk_work)
                if m:
                    subtype_val = normalize_case(m.group(1))
                    record_span(m)
            if subtype_val is None:
                m = subtype_after_pat.search(chunk_work)
                if m:
                    subtype_val = normalize_case(m.group(0))
                    record_span(m)
                else:
                    m2 = subtype_before_pat.search(chunk_work)
                    if m2:
                        subtype_val = normalize_case(m2.group(0))
                        record_span(m2)

            # 3) genetic markers (can be multiple per chunk)
            for pat in genetic_marker_patterns:
                for m in pat.finditer(chunk_work):
                    val = normalize_case(m.group(0))
                    markers.append(val)
                    record_span(m)

            # Compute residual of this chunk after removing matches
            residual = strip_wrappers(_remove_spans(chunk_work, consumed_spans))

            if residual:
                rest_parts.append(residual)

            if remaining_next is None:
                # Final chunk: this defines main_tumor_type (after removing matched parts)
                # If residual is empty (i.e., the entire chunk was a match), fall back to cleaned chunk
                main_tumor_type = residual if residual else strip_wrappers(chunk_work)
                break
            else:
                remaining = remaining_next

        # Prepare outputs
        markers = dedupe_keep_order(markers)
        genetic_markers = " and ".join(markers) if markers else None
        rest = ", ".join([p for p in rest_parts if p]) if rest_parts else None

        # Clean subclass/subtype to avoid leftover brackets/punct
        if subclass_val:
            subclass_val = strip_wrappers(subclass_val)
        if subtype_val:
            subtype_val = strip_wrappers(subtype_val)

        return {
            "tumor_family": main_tumor_type if main_tumor_type else None,
            "genetic_markers": genetic_markers,
            "subclass": subclass_val,
            "subtype": subtype_val,
            "rest": rest,
        }

    def _remove_spans(text: str, spans: List[tuple]) -> str:
        if not spans:
            return text
        spans_sorted = sorted(spans)
        out = []
        last = 0
        for a, b in spans_sorted:
            if a > last:
                out.append(text[last:a])
            last = max(last, b)
        if last < len(text):
            out.append(text[last:])
        return "".join(out)

    # Apply row-wise
    parsed = df[col].apply(parse_one)
    parsed_df = pd.DataFrame(list(parsed))
    df_list = [
        df.reset_index()[['index', col]],
        parsed_df.reset_index(drop=True)
    ]

    new_df  = pd.concat(df_list, axis=1)
    new_df = new_df.set_index('index')

    # Add original index
    new_df = new_df.loc[df.index,]

    return new_df


def diann_run(s, warn=False):
    match = re.search(r'_(\d+)_T', s)
    if match:
        return 'Run_' + match.group(1)

    match = re.search(r'(?<=_)(?:N?\d{2,5}(?:_[A-Za-z0-9]+)*_[A-Za-z]+|N?\d{5}|N?\d{2}_\d{4}[A-Za-z]?_[A-Za-z]+)(?=_T1_DIA)', s)
    if match:
        return 'Run_' + match.group(0)

    if warn:
        warnings.warn(f'No match for string:\n{s}')
        return 'no_parse_match'

    raise ValueError(f'No match for string:\n{s}')


def _pretty_design_label(label: str) -> str:
    return label.replace("_", " ").strip()


def parse_stat_test_varm_slot(
    varm_slot: str,
    adata: ad.AnnData | None = None,
) -> dict[str, str | None]:
    """
    Parse a stat-test varm slot name into its components.

    The expected format is ``<test_type>;<group_by>;<design>`` when no
    layer is used, or ``<test_type>;<group_by>;<design>;<layer>`` when
    a layer is specified. Components are separated by semicolons.

    Parameters
    ----------
    varm_slot : str
        Slot name produced by ``proteopy.tl.differential_abundance``.
        Format: ``<test_type>;<group_by>;<design>`` or
        ``<test_type>;<group_by>;<design>;<layer>``.
    adata : AnnData or None
        AnnData used to resolve layer labels. When provided, the sanitized
        layer suffix is mapped back to the original layer key.

    Returns
    -------
    dict
        Dictionary with keys: ``test_type``, ``test_type_label``,
        ``group_by``, ``design``, ``design_label``, and ``layer``.

    Raises
    ------
    ValueError
        If the slot does not match the expected stat-test format.

    Examples
    --------
    >>> slot = "welch;condition;treated_vs_control"
    >>> parse_stat_test_varm_slot(slot)
    {'test_type': 'welch', 'test_type_label': "Welch's t-test",
     'group_by': 'condition', 'design': 'treated_vs_control',
     'design_label': 'treated vs control', 'layer': None}
    """
    if not isinstance(varm_slot, str) or not varm_slot:
        raise ValueError("varm_slot must be a non-empty string.")

    parts = varm_slot.split(";")
    if len(parts) not in (3, 4):
        raise ValueError(
            "varm_slot must have format '<test_type>;<group_by>;<design>' "
            "or '<test_type>;<group_by>;<design>;<layer>', "
            f"got '{varm_slot}'."
        )

    test_type = parts[0]
    group_by = parts[1]
    design_part = parts[2]
    layer_part = parts[3] if len(parts) == 4 else None

    if test_type not in STAT_TEST_METHOD_LABELS:
        raise ValueError(
            f"Test type '{test_type}' is not supported. "
            f"Supported types: {sorted(STAT_TEST_METHOD_LABELS)}."
        )

    if not group_by:
        raise ValueError("varm_slot is missing the group_by component.")

    if not design_part:
        raise ValueError("varm_slot is missing the design component.")

    layer = None
    if layer_part:
        if adata is not None and adata.layers:
            layer_map = {
                sanitize_string(name): name
                for name in adata.layers.keys()
            }
            if layer_part in layer_map:
                layer = layer_map[layer_part]
            else:
                raise ValueError(
                    f"When adata passed, the layer part of the varm_slot "
                    f"must contain the sanitized layer part for back-"
                    f"mapping. '{layer_part}' not found in adata varm layers"
                    f"(unsanitized): {adata.layers}."
                    )
        else:
            layer = layer_part

    if design_part.endswith("_vs_rest"):
        group = design_part[: -len("_vs_rest")]
        if not group:
            raise ValueError("Design is missing the group label.")
        design = f"{group}_vs_rest"
        design_label = f"{_pretty_design_label(group)} vs rest"
    elif "_vs_" in design_part:
        group1, group2 = design_part.split("_vs_", 1)
        if not group1 or not group2:
            raise ValueError("Design is missing group labels.")
        design = f"{group1}_vs_{group2}"
        design_label = (
            f"{_pretty_design_label(group1)} vs "
            f"{_pretty_design_label(group2)}"
        )
    else:
        raise ValueError(
            "Design must use '<group1>_vs_<group2>' or '<group>_vs_rest'."
        )

    test_info = {
        "test_type": test_type,
        "test_type_label": STAT_TEST_METHOD_LABELS[test_type],
        "group_by": group_by,
        "design": design,
        "design_label": design_label,
        "layer": layer,
    }

    return test_info


def _is_standard_hclustv_key(key: str, key_type: str = "linkage") -> bool:
    """
    Check if a key follows the standard hclust key format.

    Parameters
    ----------
    key : str
        The key to check.
    key_type : str
        Type of key: "linkage", "values", or "profiles".

    Returns
    -------
    bool
        True if the key follows the standard format.
    """
    prefix = f"hclustv_{key_type};"
    parts = key.split(";")
    return key.startswith(prefix) and len(parts) == 4


def _parse_hclustv_key_components(key: str) -> tuple[str, str, str] | None:
    """
    Extract (group_by, hash, layer) components from a standard hclust key.

    Returns None if the key does not follow the standard format.
    """
    parts = key.split(";")
    if len(parts) != 4:
        return None
    if not (parts[0].startswith("hclustv_")):
        return None
    return (parts[1], parts[2], parts[3])


def _resolve_hclustv_keys(
    adata: ad.AnnData,
    linkage_key: str = 'auto',
    values_key: str = 'auto',
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Resolve linkage and values keys from adata.uns.

    Auto-detects keys if not provided, validates existence, and returns
    the resolved key names.
    """
    linkage_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclustv_linkage;")
    ]
    values_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclustv_values;")
    ]

    linkage_auto = linkage_key == 'auto'
    values_auto = values_key == 'auto'

    if linkage_auto:
        if len(linkage_candidates) == 0:
            raise ValueError(
                "No hierarchical clustering results found in adata.uns. "
                "Run proteopy.tl.hclustv_tree() first."
            )
        if len(linkage_candidates) > 1:
            raise ValueError(
                "Multiple linkage matrices found in adata.uns. "
                "Please specify linkage_key explicitly. "
                f"Available keys: {linkage_candidates}"
            )
        linkage_key = linkage_candidates[0]
        if verbose:
            print(f"Using linkage matrix: adata.uns['{linkage_key}']")
    else:
        if linkage_key not in adata.uns:
            raise KeyError(
                f"Linkage key '{linkage_key}' not found in adata.uns."
            )

    if values_auto:
        if len(values_candidates) == 0:
            raise ValueError(
                "No profile values found in adata.uns. "
                "Run proteopy.tl.hclustv_tree() first."
            )
        if len(values_candidates) > 1:
            raise ValueError(
                "Multiple profile values found in adata.uns. "
                "Please specify values_key explicitly. "
                f"Available keys: {values_candidates}"
            )
        values_key = values_candidates[0]
        if verbose:
            print(f"Using profile values: adata.uns['{values_key}']")
    else:
        if values_key not in adata.uns:
            raise KeyError(
                f"Values key '{values_key}' not found in adata.uns."
            )

    # Validate matching components when both keys are auto-detected
    if linkage_auto and values_auto:
        linkage_components = _parse_hclustv_key_components(linkage_key)
        values_components = _parse_hclustv_key_components(values_key)

        if linkage_components is not None and values_components is not None:
            if linkage_components != values_components:
                raise ValueError(
                    f"Auto-detected keys have mismatched components. "
                    f"linkage_key '{linkage_key}' has (group_by, hash, layer) = "
                    f"{linkage_components}, but values_key '{values_key}' has "
                    f"{values_components}. Please specify keys explicitly."
                )

    return linkage_key, values_key


def _resolve_hclustv_cluster_key(
    adata: ad.AnnData,
    cluster_key: str = 'auto',
    verbose: bool = True,
) -> str:
    """
    Resolve cluster annotation key from adata.var columns.

    Auto-detects key if not provided, validates existence, and returns
    the resolved key name.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with cluster annotations stored in ``.var``.
    cluster_key : str
        Column in ``adata.var`` containing cluster annotations. When
        ``'auto'``, auto-detects the cluster key if exactly one
        ``'hclustv_cluster;...'`` column exists.
    verbose : bool
        Print status messages including auto-detected key.

    Returns
    -------
    str
        Resolved cluster key name.

    Raises
    ------
    ValueError
        If no cluster annotations are found or multiple candidates exist
        when ``cluster_key='auto'``.
    KeyError
        If the specified ``cluster_key`` is not found in ``adata.var``.
    """
    cluster_candidates = [
        col for col in adata.var.columns
        if col.startswith("hclustv_cluster;")
    ]

    if cluster_key == 'auto':
        if len(cluster_candidates) == 0:
            raise ValueError(
                "No cluster annotations found in adata.var. "
                "Run proteopy.tl.hclustv_cluster_ann() first."
            )
        if len(cluster_candidates) > 1:
            raise ValueError(
                "Multiple cluster annotation columns found in adata.var. "
                "Please specify cluster_key explicitly. "
                f"Available keys: {cluster_candidates}"
            )
        cluster_key = cluster_candidates[0]
        if verbose:
            print(f"Using cluster annotations: adata.var['{cluster_key}']")
    else:
        if cluster_key not in adata.var.columns:
            raise KeyError(
                f"Cluster key '{cluster_key}' not found in adata.var columns."
            )

    return cluster_key


def _resolve_hclustv_profile_key(
    adata: ad.AnnData,
    profile_key: str = 'auto',
    verbose: bool = True,
) -> str:
    """
    Resolve cluster profile key from adata.uns.

    Auto-detects key if not provided, validates existence, and returns
    the resolved key name.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with cluster profiles stored in ``.uns``.
    profile_key : str
        Key in ``adata.uns`` containing the profiles DataFrame. When
        ``'auto'``, auto-detects the profile key if exactly one
        ``'hclustv_profiles;...'`` key exists.
    verbose : bool
        Print status messages including auto-detected key.

    Returns
    -------
    str
        Resolved profile key name.

    Raises
    ------
    ValueError
        If no profiles are found or multiple candidates exist when
        ``profile_key='auto'``.
    KeyError
        If the specified ``profile_key`` is not found in ``adata.uns``.
    """
    profile_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclustv_profiles;")
    ]

    if profile_key == 'auto':
        if len(profile_candidates) == 0:
            raise ValueError(
                "No cluster profiles found in adata.uns. "
                "Run proteopy.tl.hclustv_profiles() first."
            )
        if len(profile_candidates) > 1:
            raise ValueError(
                "Multiple cluster profiles found in adata.uns. "
                "Please specify profile_key explicitly. "
                f"Available keys: {profile_candidates}"
            )
        profile_key = profile_candidates[0]
        if verbose:
            print(f"Using profiles: adata.uns['{profile_key}']")
    else:
        if profile_key not in adata.uns:
            raise KeyError(
                f"Profile key '{profile_key}' not found in adata.uns."
            )

    return profile_key
