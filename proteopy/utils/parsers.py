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
