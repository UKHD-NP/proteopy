from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence as SequenceABC
from pathlib import Path
from typing import Union
from urllib.parse import unquote

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import anndata as ad

from proteopy.pp.summarize_peptides_by_neighbourhood_union import (
    _read_fasta,
)
from proteopy.utils.anndata import check_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.pl._resolvers import _default_label_order

ColorScheme = Union[str, dict, SequenceABC, Colormap, Callable, None]

# Row-key sources keeping peptide and add_sequences rows apart
_PEPTIDE_ROWS = "group_by"
_ADDED_ROWS = "add_sequences"

# UniProt VSP note, e.g. "In isoform 2, isoform 3 and isoform 4. GC->AP"
_VSP_NOTE_RE = re.compile(r"^In (isoform .+?)\.(?:\s+(.*))?$")


def _find_sequence_positions(
    seq: str,
    ref_sequence: str,
) -> list[int]:
    """Find all start positions of ``seq`` within ``ref_sequence``."""
    positions = []
    start = 0
    while True:
        idx = ref_sequence.find(seq, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _resolve_seq_coord(
    name: str,
    entry: dict,
    ref_len: int,
) -> dict[str, dict]:
    """Validate and resolve a seq_coord entry to coordinates."""
    start, end = entry["seq_coord"]
    group = entry["group"]
    if start >= end:
        raise ValueError(
            f"Invalid interval [{start}, {end}) "
            f"for '{name}': start must be less "
            f"than end."
        )
    if start < 0 or end > ref_len:
        raise ValueError(
            f"Sequence coordinate [{start}, {end}) "
            f"for '{name}' is out of bounds "
            f"[0, {ref_len})."
        )
    return {
        name: {
            "start": start,
            "end": end,
            "group": group,
        }
    }


def _resolve_seq_string(
    name: str,
    entry: dict,
    ref_sequence: str,
    allow_multi_match: bool,
) -> dict[str, dict]:
    """Locate a seq string in the reference and return coordinates."""
    seq = entry["seq"]
    group = entry["group"]
    if not seq:
        raise ValueError(f"Entry '{name}' has an empty " f"'seq' string.")
    positions = _find_sequence_positions(
        seq,
        ref_sequence,
    )

    if not positions:
        raise ValueError(
            f"Sequence '{name}' ('{seq}') was not "
            f"found in the reference sequence."
        )

    if len(positions) > 1 and not allow_multi_match:
        raise ValueError(
            f"Sequence '{name}' ('{seq}') matches "
            f"the reference at {len(positions)} "
            f"positions ({positions}). Set "
            f"allow_multi_match=True to show all "
            f"matches as separate bars."
        )

    result = {}
    if len(positions) == 1:
        result[name] = {
            "start": positions[0],
            "end": positions[0] + len(seq),
            "group": group,
        }
    else:
        for i, pos in enumerate(positions, start=1):
            match_name = f"{name} (match {i})"
            result[match_name] = {
                "start": pos,
                "end": pos + len(seq),
                "group": group,
            }
    return result


def _resolve_sequences(
    ref_sequence: str,
    sequences: dict[str, dict],
    allow_multi_match: bool,
) -> dict[str, dict]:
    """Resolve each sequence entry to absolute ``[start, end)`` coordinates.

    For entries with a ``"seq"`` key the amino-acid string is located
    within ``ref_sequence`` via substring search.  Entries with a
    ``"seq_coord"`` key are validated against the reference length.
    When ``allow_multi_match`` is ``True`` and a substring appears
    more than once, one entry per match is emitted with the suffix
    ``" (match N)"``.

    Parameters
    ----------
    ref_sequence : str
        Full reference protein sequence.
    sequences : dict[str, dict]
        Mapping of sequence name to a dict containing
        ``"group"`` and either ``"seq"`` or ``"seq_coord"``.
    allow_multi_match : bool
        If ``False``, raise ``ValueError`` when a ``"seq"``
        string matches the reference at more than one position.

    Returns
    -------
    dict[str, dict]
        Mapping of (possibly expanded) sequence names to dicts
        with ``"start"``, ``"end"``, and ``"group"`` keys.

    Raises
    ------
    ValueError
        If a ``"seq"`` string is not found in the reference,
        if coordinates are out of bounds, if an entry lacks
        both ``"seq"`` and ``"seq_coord"``, or if a sequence
        multi-matches and ``allow_multi_match`` is ``False``.
    """
    ref_len = len(ref_sequence)
    resolved = {}

    for name, entry in sequences.items():
        if "seq" in entry and "seq_coord" in entry:
            raise ValueError(
                f"Entry '{name}' has both 'seq' and "
                f"'seq_coord'; provide only one."
            )

        if "seq_coord" in entry:
            result = _resolve_seq_coord(name, entry, ref_len)
        elif "seq" in entry:
            result = _resolve_seq_string(
                name,
                entry,
                ref_sequence,
                allow_multi_match,
            )
        else:
            raise ValueError(
                f"Entry '{name}' must have either 'seq' "
                f"or 'seq_coord' key."
            )
        resolved.update(result)

    return resolved


def _check_overlaps(
    groups: dict[str, list[tuple[int, int, str]]],
    row_labels: dict | None = None,
) -> None:
    """Raise ``ValueError`` if any two sequences within a group overlap."""
    row_labels = row_labels or {}
    for group_key, intervals in groups.items():
        group_name = row_labels.get(group_key, group_key)
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(len(sorted_intervals) - 1):
            _, end_a, name_a = sorted_intervals[i]
            start_b, _, name_b = sorted_intervals[i + 1]
            if end_a > start_b:
                raise ValueError(
                    f"Overlapping sequences in group "
                    f"'{group_name}': '{name_a}' "
                    f"[..., {end_a}) overlaps with "
                    f"'{name_b}' [{start_b}, ...). "
                    f"Set allow_overlaps=True to permit "
                    f"overlapping sequences."
                )


def _group_sequences(
    resolved: dict[str, dict],
    order: list[str] | None,
    default_order: list | None = None,
) -> tuple[list[str], dict[str, list]]:
    """Build group ordering and entries from resolved sequences."""
    group_order = []
    group_entries = {}
    for name, entry in resolved.items():
        g = entry["group"]
        if g not in group_entries:
            group_order.append(g)
            group_entries[g] = []
        group_entries[g].append((entry["start"], entry["end"], name))

    if order is not None:
        unknown = set(order) - set(group_order)
        if unknown:
            raise ValueError(
                f"Groups in 'order' not found in "
                f"sequences: {sorted(unknown)}."
            )
        group_order = list(order)
    elif default_order is not None:
        group_order = list(default_order)
    else:
        # Sort lexicographically so row order and color assignment
        # are deterministic and match other plots of the same groups.
        group_order = sorted(group_order, key=str)

    return group_order, group_entries


def _plot_sequences_on_reference(
    ref_sequence: str,
    sequences: dict[str, dict],
    color_scheme: ColorScheme = None,
    allow_overlaps: bool = False,
    allow_multi_match: bool = False,
    title: str | None = None,
    order: list[str] | None = None,
    default_order: list | None = None,
    row_labels: dict | None = None,
    row_colors: dict | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Render sequences as horizontal bars aligned to a reference sequence.

    Draws a grey reference bar at the bottom and one colored broken-barh
    row per group above it. Each entry in ``sequences`` must supply either a
    ``"seq"`` key (amino-acid string to locate by substring search) or a
    ``"seq_coord"`` key (pre-computed ``[start, end)`` tuple), plus a
    ``"group"`` key that determines row assignment and legend label.

    Parameters
    ----------
    ref_sequence : str
        Full reference protein sequence against which all sequences are
        plotted.
    sequences : dict[str, dict]
        Mapping of sequence name to a dict with keys:

        - ``"group"`` (*str*) -- row label for this sequence.
        - ``"seq"`` (*str*, optional) -- amino-acid string; matched by
          substring search within ``ref_sequence``.
        - ``"seq_coord"`` (*tuple[int, int]*, optional) -- explicit
          ``[start, end)`` coordinates; mutually exclusive with
          ``"seq"``.
    color_scheme : str | dict | list | Colormap | callable | None
        One color per group.
    allow_overlaps : bool
        Skip overlap validation when ``True``; raise ``ValueError``
        for any two sequences in the same group that overlap when
        ``False``.
    allow_multi_match : bool
        When ``True``, sequences matching the reference at more than
        one position are plotted as separate bars. When ``False``, a
        ``ValueError`` is raised for multi-matching sequences.
    title : str | None
        Axes title.
    order : list[str] | None
        Explicit ordering of group rows. Groups absent from ``order`` are
        excluded.
    default_order : list | None
        Row order, bottom to top, used when ``order`` is ``None``. Must
        list every group. When ``None``, groups are sorted
        lexicographically.
    row_labels : dict | None
        Mapping of group key to the label drawn for its row, so that
        distinct groups can share a label. Groups absent from the
        mapping are labeled with the group key itself.
    row_colors : dict | None
        Mapping of group key to bar color. When given, it replaces
        ``color_scheme`` resolution; groups absent from it are drawn
        in ``"C0"``.
    figsize : tuple[float, float] | None
        Figure dimensions ``(width, height)`` in inches passed to
        ``plt.subplots``.
    ax : matplotlib.axes.Axes | None
        Matplotlib Axes object to plot onto. If ``None``, a new
        figure and axes are created. The function always returns
        the Axes object used for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object used for plotting.
    """
    resolved = _resolve_sequences(
        ref_sequence,
        sequences,
        allow_multi_match,
    )

    group_order, group_entries = _group_sequences(
        resolved,
        order,
        default_order,
    )

    row_labels = row_labels or {}
    if not allow_overlaps:
        _check_overlaps(group_entries, row_labels)

    labels = [row_labels.get(g, g) for g in group_order]

    # Resolve colors
    if row_colors is not None:
        color_map = row_colors
    else:
        colors = _resolve_color_scheme(color_scheme, labels)
        color_map = {}
        if colors is not None:
            color_map = dict(zip(group_order, colors))

    # Create figure or use provided axes
    if ax is None:
        if figsize is None:
            figsize = (12, 1.5 + len(group_order) * 0.8)
        _, _ax = plt.subplots(figsize=figsize)
    else:
        _ax = ax

    ref_len = len(ref_sequence)
    bar_height = 0.6

    # Reference bar at y=0
    _ax.broken_barh(
        [(0, ref_len)],
        (0, bar_height),
        facecolors="lightgrey",
        edgecolors="grey",
        linewidth=0.5,
    )

    # Group bars stacked upward
    y_labels = ["Reference"]
    y_positions = [bar_height / 2]

    for i, group_name in enumerate(group_order):
        y_base = (i + 1) * (bar_height + 0.3)
        intervals = [
            (entry[0], entry[1] - entry[0])
            for entry in group_entries[group_name]
        ]
        face_color = color_map.get(group_name, "C0")
        _ax.broken_barh(
            intervals,
            (y_base, bar_height),
            facecolors=face_color,
            edgecolors="black",
            linewidth=0.5,
        )
        y_labels.append(row_labels.get(group_name, group_name))
        y_positions.append(y_base + bar_height / 2)

    # Configure axes
    _ax.set_xlim(0, ref_len)
    _ax.set_yticks(y_positions)
    _ax.set_yticklabels(y_labels)
    _ax.set_xlabel("Position")

    if title is not None:
        _ax.set_title(title)

    # Remove top and right spines
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    return _ax


def _extract_peptide_sequences(
    var_sub: pd.DataFrame,
    alt_pep_sequence_key: str | None,
) -> pd.Series:
    """Return peptide sequences from var subset."""
    if alt_pep_sequence_key is not None:
        if alt_pep_sequence_key not in var_sub.columns:
            raise KeyError(
                f"Column '{alt_pep_sequence_key}' not " f"found in adata.var."
            )
        return var_sub[alt_pep_sequence_key]
    return var_sub["peptide_id"]


def _ref_sequence_from_fasta(
    fasta_path: str,
    protein_id: str,
) -> str:
    """Look up ``protein_id`` among the accessions of a FASTA file."""
    sequences = _read_fasta(fasta_path)
    if protein_id not in sequences:
        raise ValueError(
            f"Protein '{protein_id}' not found in FASTA "
            f"'{fasta_path}'. Accessions are read from the "
            f"second '|'-delimited header field (e.g. "
            f"'sp|P12345|NAME_HUMAN' -> 'P12345'), or the "
            f"full header if it has no such field."
        )
    return sequences[protein_id]


def _ref_sequence_from_var(
    var: pd.DataFrame,
    ref_key: str,
    protein_id: str,
) -> str:
    """Return the single reference sequence stored for ``protein_id``."""
    values = var.loc[var["protein_id"] == protein_id, ref_key]
    if values.empty:
        raise ValueError(
            f"No peptides found where "
            f"adata.var['protein_id'] == '{protein_id}'."
        )
    if values.isna().any():
        raise ValueError(
            f"adata.var['{ref_key}'] contains missing values "
            f"for protein '{protein_id}'."
        )
    unique = values.astype(str).unique()
    if len(unique) > 1:
        raise ValueError(
            f"adata.var['{ref_key}'] holds {len(unique)} "
            f"different sequences for protein "
            f"'{protein_id}'; expected exactly one."
        )
    return unique[0]


def _parse_vsp_note(
    note: str,
) -> tuple[list[str], str | None] | None:
    """Split a VSP note into isoform names and the replaced residues.

    Returns ``None`` if the note does not follow the UniProt format.
    The replaced residues are ``None`` for deletions (``"Missing"``).
    """
    match = _VSP_NOTE_RE.match(note)
    if match is None:
        return None
    isoforms = re.split(r", | and ", match.group(1))
    change = match.group(2) or ""
    original = change.split("->")[0].strip() if "->" in change else None
    return isoforms, original


def _isoform_sequences_from_gff(
    gff_path: str,
    protein_id: str,
    ref_sequence: str,
) -> dict[str, dict]:
    """Build ``add_sequences`` entries from UniProt VSP features.

    Each "Alternative sequence" feature of ``protein_id`` becomes one
    bar per isoform it belongs to, grouped by isoform. GFF coordinates
    are 1-based and closed; they are converted to 0-based half-open.
    """
    sequences = {}
    prefix = f"{protein_id}\t"
    with open(gff_path) as handle:
        for line in handle:
            if not line.startswith(prefix):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "Alternative sequence":
                continue
            attrs = dict(
                attr.split("=", 1)
                for attr in fields[8].split(";")
                if "=" in attr
            )
            vsp_id = attrs.get("ID", f"VSP_{fields[3]}-{fields[4]}")
            parsed = _parse_vsp_note(unquote(attrs.get("Note", "")))
            if parsed is None:
                warnings.warn(
                    f"Skipping {vsp_id} of '{protein_id}': "
                    f"unrecognised note "
                    f"'{unquote(attrs.get('Note', ''))}'.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            isoforms, original = parsed
            start, end = int(fields[3]) - 1, int(fields[4])

            # The replaced residues are stated for substitutions; a
            # mismatch means ref_sequence is not the canonical
            # sequence the coordinates refer to.
            if original is not None and ref_sequence[start:end] != original:
                raise ValueError(
                    f"{vsp_id} of '{protein_id}' replaces "
                    f"'{original}' at [{start}, {end}), but the "
                    f"reference sequence has "
                    f"'{ref_sequence[start:end]}' there. UniProt "
                    f"VSP coordinates refer to the canonical "
                    f"sequence; pass that as ref_sequence."
                )
            for isoform in isoforms:
                sequences[f"{vsp_id} ({isoform})"] = {
                    "seq_coord": (start, end),
                    "group": isoform,
                }

    if not sequences:
        warnings.warn(
            f"No 'Alternative sequence' features found for "
            f"'{protein_id}' in '{gff_path}'.",
            UserWarning,
            stacklevel=3,
        )
    return sequences


def _resolve_ref_sequence(
    ref_sequence: str,
    var: pd.DataFrame,
    protein_id: str,
) -> str:
    """Resolve ``ref_sequence`` to an amino-acid string.

    Checked in order: a FASTA annotator path (``.fasta`` suffix,
    case-insensitive), a column in ``var``, and finally a literal
    amino-acid sequence.
    """
    if ref_sequence.lower().endswith(".fasta"):
        return _ref_sequence_from_fasta(ref_sequence, protein_id)
    if ref_sequence in var.columns:
        return _ref_sequence_from_var(var, ref_sequence, protein_id)
    return ref_sequence


def _extract_groups(
    var_sub: pd.DataFrame,
    group_by: str | None,
    filter_value: str,
) -> pd.Series:
    """Return group labels for each peptide in var subset."""
    if group_by is not None:
        if group_by not in var_sub.columns:
            raise KeyError(f"Column '{group_by}' not found in " f"adata.var.")
        return var_sub[group_by]
    return pd.Series(
        filter_value,
        index=var_sub.index,
    )


def _natural_sort_key(label) -> list:
    """Sort key ordering embedded numbers numerically (2 before 10)."""
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(label))
    ]


def _default_row_order(
    groups: pd.Series,
    add_sequences: dict[str, dict] | None,
) -> list:
    """Return the default row order, bottom to top.

    Rows are keyed by ``(source, label)`` so that a peptide group and
    an ``add_sequences`` group sharing a label stay separate rows.
    Peptide groups come first, in category order if ``groups`` is
    categorical (categories without peptides dropped), else in
    lexicographic order of their ``str`` values. The
    ``add_sequences`` groups follow in natural order, so that
    ``"isoform 2"`` precedes ``"isoform 10"``.
    """
    pep_order = _default_label_order(groups)

    add_order = []
    if add_sequences:
        add_order = sorted(
            {entry["group"] for entry in add_sequences.values()},
            key=_natural_sort_key,
        )

    return [(_PEPTIDE_ROWS, g) for g in pep_order] + [
        (_ADDED_ROWS, g) for g in add_order
    ]


def _row_colors(
    color_scheme: ColorScheme,
    row_keys: list[tuple],
) -> dict[tuple, object]:
    """Assign bar colors by position among the drawn rows.

    ``color_scheme`` is resolved over the peptide groups in the order
    they are drawn. With the default order this is the order in which
    :func:`proteopy.pl.peptide_intensities` colors its ``color``
    column (and hence ``proteoform_intensities`` its proteoforms), so
    a proteoform has the same color in both plots. The
    ``add_sequences`` groups continue the color sequence where the
    peptide groups end.
    """
    pep_labels = [label for src, label in row_keys if src == _PEPTIDE_ROWS]
    add_labels = [label for src, label in row_keys if src == _ADDED_ROWS]

    pep_colors = _resolve_color_scheme(color_scheme, pep_labels) or []
    all_colors = (
        _resolve_color_scheme(color_scheme, pep_labels + add_labels) or []
    )
    colors = dict(
        zip([(_PEPTIDE_ROWS, label) for label in pep_labels], pep_colors)
    )
    colors.update(
        zip(
            [(_ADDED_ROWS, label) for label in add_labels],
            all_colors[len(pep_labels) :],
        )
    )
    return colors


def _order_to_row_keys(
    order: list | None,
    row_keys: list[tuple],
) -> list[tuple] | None:
    """Translate user-facing row labels in ``order`` to row keys.

    A label shared by a peptide group and an ``add_sequences`` group
    selects both rows, the peptide row first.
    """
    if order is None:
        return None
    selected = []
    unknown = []
    for label in order:
        matches = [k for k in row_keys if str(k[1]) == str(label)]
        if not matches:
            unknown.append(label)
        selected.extend(matches)
    if unknown:
        raise ValueError(
            f"Groups in 'order' not found in "
            f"sequences: {sorted(map(str, unknown))}."
        )
    return selected


def _build_sequences_dict(
    var_sub: pd.DataFrame,
    pep_seqs: pd.Series,
    groups: pd.Series,
    add_sequences: dict[str, dict] | None,
) -> dict[str, dict]:
    """Build sequences dict from peptides and merge additions."""
    sequences = {}
    for pep_id, seq_val, grp_val in zip(
        var_sub.index,
        pep_seqs,
        groups,
    ):
        if pd.isna(grp_val):
            continue
        sequences[pep_id] = {
            "seq": str(seq_val),
            "group": (_PEPTIDE_ROWS, grp_val),
        }

    if add_sequences is not None:
        conflicts = set(sequences) & set(add_sequences)
        if conflicts:
            raise ValueError(
                f"Name conflicts between peptide "
                f"sequences and additional sequences: "
                f"{sorted(conflicts)}."
            )
        sequences.update(
            {
                name: {**entry, "group": (_ADDED_ROWS, entry["group"])}
                for name, entry in add_sequences.items()
            }
        )

    return sequences


def peptides_on_sequence(
    adata: ad.AnnData,
    filter_key: str,
    filter_value: str,
    group_by: str | None = None,
    ref_sequence: str | None = None,
    alt_pep_sequence_key: str | None = None,
    add_sequences: dict[str, dict] | None = None,
    allow_overlaps: bool = False,
    allow_multi_match: bool = False,
    color_scheme: ColorScheme = None,
    title: str | None = None,
    order: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot peptide coverage on a reference sequence as horizontal bars.

    Filters peptides from a peptide-level :class:`~anndata.AnnData` using the
    ``filter_key`` and ``filter_value`` parameters, which are then drawn as
    broken bars on top of a grey reference bar. Extra sequences (e.g., domains
    or post-translational modification sites) can be overlaid via the
    ``add_sequences`` parameter.

    Parameters
    ----------
    adata : AnnData
        Peptide-level :class:`~anndata.AnnData`.
    filter_key : str
        Column in ``adata.var`` used to select peptides.
    filter_value : str
        Value in the ``filter_key`` column to match.
    group_by : str | None
        Column in ``adata.var`` used to assign peptides to
        colored rows. When ``None``, all peptides are placed
        in a single row labeled ``filter_value``. Peptides with a
        missing value in this column are not plotted.
    ref_sequence : str | None
        Full reference protein sequence as an amino-acid string.
    alt_pep_sequence_key : str | None
        Column in ``adata.var`` whose values are used as peptide
        amino-acid strings. When ``None``,
        ``adata.var["peptide_id"]`` is used.
    add_sequences : dict[str, dict] | None
        Additional named sequences to overlay (e.g., domains).
        Each value must be a dict with a ``"group"`` key and
        either a ``"seq"`` or ``"seq_coord"`` key. Coordinates
        are 0-based throughout. ``"seq_coord"`` must be a
        half-open ``[start, end)`` tuple where ``start >= 0``
        and ``end <= len(ref_sequence)``.
    allow_overlaps : bool
        When ``False``, raise ``ValueError`` if two sequences in
        the same group overlap positionally.
    allow_multi_match : bool
        When ``True``, a peptide sequence matching the reference at
        multiple positions is shown as one bar per match, labeled
        ``"<name> (match N)"``. When ``False``, a ``ValueError``
        is raised for ambiguous matches.
    color_scheme : str | dict | list | Colormap | callable | None
        Color specification for groups. The ``group_by`` groups
        take colors in the order their rows are drawn, and the
        ``add_sequences`` groups continue the color sequence
        where they end. With the default order (category order
        if categorical, else lexicographic) this matches
        :func:`~proteopy.pl.proteoform_intensities` and
        :func:`~proteopy.pl.peptide_intensities`, so the same
        proteoform gets the same color in both plots. A custom
        ``order`` recolors the rows by their new positions.
    title : str | None
        Axes title.
    order : list[str] | None
        Group rows to draw, bottom to top, directly above the
        reference. Groups absent from ``order`` are excluded.
        When ``None``, the peptide groups are drawn first, in
        the category order of ``group_by`` if it is
        categorical (categories without peptides are
        dropped), else in lexicographic order of the
        ``str``-coerced values. To change it, store
        ``group_by`` as an ordered Categorical. The
        ``add_sequences`` groups follow above in natural
        order (``"isoform 2"`` before ``"isoform 10"``). A
        peptide group and an ``add_sequences`` group with the
        same label are drawn as separate rows; naming that
        label in ``order`` selects both.
    figsize : tuple[float, float] | None
        Figure dimensions ``(width, height)`` in inches.
    show : bool
        Call ``plt.show()`` at the end.
    save : str | Path | None
        Path at which to save the figure (300 dpi,
        ``bbox_inches="tight"``). When ``None``, the figure is
        not saved.
    ax : matplotlib.axes.Axes | None
        Matplotlib Axes object to plot onto. If ``None``, a new
        figure and axes are created. The function always returns
        the Axes object used for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object used for plotting.

    Raises
    ------
    ValueError
        If ``adata`` is not at peptide level.
    ValueError
        If ``ref_sequence`` is ``None``.
    ValueError
        If no peptides match ``filter_value`` in
        ``adata.var[filter_key]``.
    ValueError
        If a peptide sequence is not found in the reference.
    ValueError
        If a peptide matches the reference at multiple positions
        and ``allow_multi_match=False``.
    ValueError
        If two sequences in the same group overlap and
        ``allow_overlaps=False``.
    KeyError
        If ``filter_key``, ``alt_pep_sequence_key``, or
        ``group_by`` is not found in ``adata.var``.

    Examples
    --------
    Build a minimal peptide-level AnnData with two proteins.
    ``"ELPGW"`` belongs to P2 and is excluded by the filter.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> ref = "MKTAYIAKQRQISFVKSHFSRQDEEP"
    >>> peptides = ["KTAYIAK", "QRQISFVK", "SHFSRQ", "ELPGW"]
    >>> var = pd.DataFrame(
    ...     {
    ...         "peptide_id": peptides,
    ...         "protein_id": ["P1", "P1", "P1", "P2"],
    ...         "proteoform": ["A", "A", "B", "A"],
    ...     },
    ...     index=peptides,
    ... )
    >>> obs = pd.DataFrame(
    ...     {"sample_id": ["S1", "S2"]},
    ...     index=["S1", "S2"],
    ... )
    >>> adata = ad.AnnData(
    ...     X=np.ones((2, 4)),
    ...     obs=obs,
    ...     var=var,
    ... )

    Filter to P1; group peptides by proteoform.

    >>> pr.pl.peptides_on_sequence(
    ...     adata,
    ...     filter_key="protein_id",
    ...     filter_value="P1",
    ...     ref_sequence=ref,
    ...     group_by="proteoform",
    ... )
    """
    _, level = check_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "peptides_on_sequence requires peptide-level "
            "data. The provided AnnData is at the "
            f"'{level}' level."
        )

    if ref_sequence is None:
        raise ValueError("'ref_sequence' must be provided.")

    # Filter .var
    var = adata.var
    if filter_key not in var.columns:
        raise KeyError(f"Column '{filter_key}' not found in " f"adata.var.")
    mask = var[filter_key] == filter_value
    if not mask.any():
        raise ValueError(
            f"No peptides found where "
            f"adata.var['{filter_key}'] == "
            f"'{filter_value}'."
        )
    var_sub = var.loc[mask]

    pep_seqs = _extract_peptide_sequences(
        var_sub,
        alt_pep_sequence_key,
    )
    groups = _extract_groups(
        var_sub,
        group_by,
        filter_value,
    )
    sequences = _build_sequences_dict(
        var_sub,
        pep_seqs,
        groups,
        add_sequences,
    )
    default_order = _default_row_order(groups, add_sequences)
    row_order = _order_to_row_keys(order, default_order)

    # Resolve title: default to filter_value
    if title is None:
        title = str(filter_value)

    # Plot
    _ax = _plot_sequences_on_reference(
        ref_sequence=ref_sequence,
        sequences=sequences,
        color_scheme=color_scheme,
        allow_overlaps=allow_overlaps,
        allow_multi_match=allow_multi_match,
        title=title,
        order=row_order,
        default_order=default_order,
        row_labels={key: key[1] for key in default_order},
        row_colors=_row_colors(color_scheme, row_order or default_order),
        figsize=figsize,
        ax=ax,
    )

    fig = _ax.figure

    if ax is None:
        fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return _ax


def peptides_on_prot_sequence(
    adata: ad.AnnData,
    protein_id: str,
    group_by: str | None = None,
    ref_sequence: str | None = None,
    alt_pep_sequence_key: str | None = None,
    add_sequences: dict[str, dict] | str | None = None,
    allow_overlaps: bool = False,
    allow_multi_match: bool = False,
    color_scheme: ColorScheme = None,
    title: str | None = None,
    order: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot peptide coverage on a protein reference sequence.

    Filters peptides belonging to the given ``protein_id`` and draws them as
    broken bars on top of a grey reference bar. Extra sequences (e.g., domains
    or post-translational modification sites) can be overlaid via the
    ``add_sequences`` parameter.

    Parameters
    ----------
    adata : AnnData
        Peptide-level :class:`~anndata.AnnData`.
    protein_id : str
        Value in ``adata.var["protein_id"]`` to select
        peptides for.
    group_by : str | None
        Column in ``adata.var`` used to assign peptides to
        colored rows. When ``None``, all peptides are placed
        in a single row labeled ``protein_id``. Peptides with a
        missing value in this column are not plotted.
    ref_sequence : str | None
        Reference protein sequence. Resolved in this order:

        1. **FASTA annotator** -- if the string ends with
           ``.fasta`` (case-insensitive), it is read as the
           path to a reference proteome FASTA file. The entry
           whose accession equals ``protein_id`` is used. The
           accession is the second ``|``-delimited header field
           (UniProt-style ``sp|P12345|NAME_HUMAN`` ->
           ``P12345``), or the full header if it has no such
           field.
        2. **adata.var column** -- if the string names a column
           in ``adata.var``, the sequence stored in that column
           for the peptides of ``protein_id`` is used. All
           these peptides must carry the same, non-missing
           sequence.
        3. **Amino-acid string** -- otherwise the string itself
           is used as the reference sequence.
    alt_pep_sequence_key : str | None
        Column in ``adata.var`` whose values are used as
        peptide amino-acid strings. When ``None``,
        ``adata.var["peptide_id"]`` is used.
    add_sequences : dict[str, dict] | str | None
        Additional sequences to overlay. Either:

        - **dict** -- named sequences (e.g., domains). Each
          value must be a dict with a ``"group"`` key and
          either a ``"seq"`` or ``"seq_coord"`` key.
          Coordinates are 0-based throughout. ``"seq_coord"``
          must be a half-open ``[start, end)`` tuple where
          ``start >= 0`` and ``end <= len(ref_sequence)``.
        - **GFF isoform annotator** -- a path ending with
          ``.gff`` (case-insensitive) to a UniProt GFF
          download. Its "Alternative sequence" (VSP) features
          for ``protein_id`` are drawn one row per isoform,
          marking the reference regions that the isoform
          replaces or deletes. A feature shared by several
          isoforms appears in each of their rows. The
          coordinates refer to the UniProt canonical
          sequence, so ``ref_sequence`` must be that
          sequence. Isoform rows are labeled as in UniProt
          (e.g. ``"isoform 2"``); see ``order`` for their
          placement.
    allow_overlaps : bool
        When ``False``, raise ``ValueError`` if two
        sequences in the same group overlap positionally.
    allow_multi_match : bool
        When ``True``, a peptide sequence matching the
        reference at multiple positions is shown as one bar
        per match, labeled ``"<name> (match N)"``. When
        ``False``, a ``ValueError`` is raised for ambiguous
        matches.
    color_scheme : str | dict | list | Colormap | callable | None
        Color specification for groups. The ``group_by`` groups
        take colors in the order their rows are drawn, and the
        ``add_sequences`` groups continue the color sequence
        where they end. With the default order (category order
        if categorical, else lexicographic) this matches
        :func:`~proteopy.pl.proteoform_intensities` and
        :func:`~proteopy.pl.peptide_intensities`, so the same
        proteoform gets the same color in both plots. A custom
        ``order`` recolors the rows by their new positions.
    title : str | None
        Axes title.
    order : list[str] | None
        Group rows to draw, bottom to top, directly above the
        reference. Groups absent from ``order`` are excluded.
        When ``None``, the peptide groups are drawn first, in
        the category order of ``group_by`` if it is
        categorical (categories without peptides are
        dropped), else in lexicographic order of the
        ``str``-coerced values. To change it, store
        ``group_by`` as an ordered Categorical. The
        ``add_sequences`` groups follow above in natural
        order (``"isoform 2"`` before ``"isoform 10"``). A
        peptide group and an ``add_sequences`` group with the
        same label are drawn as separate rows; naming that
        label in ``order`` selects both.
    figsize : tuple[float, float] | None
        Figure dimensions ``(width, height)`` in inches.
    show : bool
        Call ``plt.show()`` at the end.
    save : str | Path | None
        Path at which to save the figure (300 dpi,
        ``bbox_inches="tight"``). When ``None``, the figure
        is not saved.
    ax : matplotlib.axes.Axes | None
        Matplotlib Axes object to plot onto. If ``None``, a
        new figure and axes are created. The function always
        returns the Axes object used for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object used for plotting.

    Raises
    ------
    ValueError
        If ``adata`` is not at peptide level.
    ValueError
        If ``ref_sequence`` is ``None``.
    ValueError
        If no peptides match ``protein_id`` in
        ``adata.var["protein_id"]``.
    ValueError
        If ``ref_sequence`` is a FASTA path lacking
        ``protein_id``, or a ``adata.var`` column holding
        missing or differing sequences for ``protein_id``.
    ValueError
        If ``add_sequences`` is a string not ending with
        ``.gff``.
    ValueError
        If ``add_sequences`` is a GFF path and a substitution
        feature does not match ``ref_sequence`` at its
        coordinates.
    FileNotFoundError
        If ``ref_sequence`` is a FASTA path, or
        ``add_sequences`` a GFF path, that does not exist.
    ValueError
        If a peptide sequence is not found in the reference.
    ValueError
        If a peptide matches the reference at multiple
        positions and ``allow_multi_match=False``.
    ValueError
        If two sequences in the same group overlap and
        ``allow_overlaps=False``.
    KeyError
        If ``alt_pep_sequence_key`` or ``group_by`` is not
        found in ``adata.var``.

    Warns
    -----
    UserWarning
        If ``add_sequences`` is a GFF path without
        "Alternative sequence" features for ``protein_id``.

    Examples
    --------
    Build a minimal peptide-level AnnData with two proteins.
    ``"ELPGW"`` belongs to P2 and is excluded by the filter.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> ref = "MKTAYIAKQRQISFVKSHFSRQDEEP"
    >>> peptides = [
    ...     "KTAYIAK", "QRQISFVK", "SHFSRQ", "ELPGW",
    ... ]
    >>> var = pd.DataFrame(
    ...     {
    ...         "peptide_id": peptides,
    ...         "protein_id": ["P1", "P1", "P1", "P2"],
    ...         "proteoform": ["A", "A", "B", "A"],
    ...     },
    ...     index=peptides,
    ... )
    >>> obs = pd.DataFrame(
    ...     {"sample_id": ["S1", "S2"]},
    ...     index=["S1", "S2"],
    ... )
    >>> adata = ad.AnnData(
    ...     X=np.ones((2, 4)),
    ...     obs=obs,
    ...     var=var,
    ... )

    Filter to P1; group peptides by proteoform.

    >>> pr.pl.peptides_on_prot_sequence(
    ...     adata,
    ...     protein_id="P1",
    ...     ref_sequence=ref,
    ...     group_by="proteoform",
    ... )

    The reference sequence can also be read from a reference
    proteome FASTA, matched by accession, or from an
    ``adata.var`` column.

    >>> pr.pl.peptides_on_prot_sequence(
    ...     adata,
    ...     protein_id="P1",
    ...     ref_sequence="reference_proteome.fasta",
    ... )
    >>> adata.var["protein_sequence"] = [ref, ref, ref, "ELPGWK"]
    >>> pr.pl.peptides_on_prot_sequence(
    ...     adata,
    ...     protein_id="P1",
    ...     ref_sequence="protein_sequence",
    ... )

    Mark the regions of the canonical sequence that differ in
    each isoform, using UniProt "Alternative sequence" features
    (``protein_id`` must be the UniProt accession).

    >>> pr.pl.peptides_on_prot_sequence(
    ...     adata,
    ...     protein_id="O00154",
    ...     ref_sequence="uniprot_human.fasta",
    ...     add_sequences="uniprot_human.gff",
    ... )
    """
    check_proteodata(adata)
    if isinstance(add_sequences, str) and not (
        add_sequences.lower().endswith(".gff")
    ):
        raise ValueError(
            f"add_sequences must be a dict or a path ending with "
            f"'.gff', got '{add_sequences}'."
        )
    if ref_sequence is not None:
        ref_sequence = _resolve_ref_sequence(
            ref_sequence,
            adata.var,
            protein_id,
        )
        if isinstance(add_sequences, str):
            add_sequences = _isoform_sequences_from_gff(
                add_sequences,
                protein_id,
                ref_sequence,
            )

    return peptides_on_sequence(
        adata,
        filter_key="protein_id",
        filter_value=protein_id,
        group_by=group_by,
        ref_sequence=ref_sequence,
        alt_pep_sequence_key=alt_pep_sequence_key,
        add_sequences=add_sequences,
        allow_overlaps=allow_overlaps,
        allow_multi_match=allow_multi_match,
        color_scheme=color_scheme,
        title=title,
        order=order,
        figsize=figsize,
        show=show,
        save=save,
        ax=ax,
    )
