from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import anndata as ad

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme


def _find_sequence_positions(
    seq: str,
    ref_sequence: str,
) -> list[int]:
    positions = []
    start = 0
    while True:
        idx = ref_sequence.find(seq, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _resolve_sequences(
    ref_sequence: str,
    sequences: dict[str, dict],
    allow_multi_match: bool,
) -> dict[str, dict]:
    ref_len = len(ref_sequence)
    resolved = {}

    for name, entry in sequences.items():
        group = entry["group"]

        if "seq_coord" in entry:
            start, end = entry["seq_coord"]
            if start < 0 or end > ref_len or start >= end:
                raise ValueError(
                    f"Sequence coordinate [{start}, {end}) "
                    f"for '{name}' is out of bounds "
                    f"[0, {ref_len})."
                )
            resolved[name] = {
                "start": start,
                "end": end,
                "group": group,
            }

        elif "seq" in entry:
            seq = entry["seq"]
            positions = _find_sequence_positions(
                seq, ref_sequence,
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

            if len(positions) == 1:
                resolved[name] = {
                    "start": positions[0],
                    "end": positions[0] + len(seq),
                    "group": group,
                }
            else:
                for i, pos in enumerate(positions, start=1):
                    match_name = f"{name} (match {i})"
                    resolved[match_name] = {
                        "start": pos,
                        "end": pos + len(seq),
                        "group": group,
                    }

        else:
            raise ValueError(
                f"Entry '{name}' must have either 'seq' "
                f"or 'seq_coord' key."
            )

    return resolved


def _check_overlaps(
    groups: dict[str, list[tuple[int, int, str]]],
) -> None:
    for group_name, intervals in groups.items():
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


def _plot_sequences_on_reference(
    ref_sequence: str,
    sequences: dict[str, dict],
    color_scheme=None,
    allow_overlaps: bool = False,
    allow_multi_match: bool = False,
    title: str | None = None,
    order: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    ax_obj: Axes | None = None,
) -> tuple[Axes, bool]:
    resolved = _resolve_sequences(
        ref_sequence, sequences, allow_multi_match,
    )

    # Group resolved sequences preserving insertion order
    group_order = []
    group_entries: dict[str, list[tuple[int, int, str]]] = {}
    for name, entry in resolved.items():
        g = entry["group"]
        if g not in group_entries:
            group_order.append(g)
            group_entries[g] = []
        group_entries[g].append(
            (entry["start"], entry["end"], name)
        )

    # Apply user-specified order
    if order is not None:
        unknown = set(order) - set(group_order)
        if unknown:
            raise ValueError(
                f"Groups in 'order' not found in "
                f"sequences: {sorted(unknown)}."
            )
        group_order = list(order)

    if not allow_overlaps:
        _check_overlaps(group_entries)

    # Resolve colors
    colors = _resolve_color_scheme(color_scheme, group_order)
    if colors is None:
        cycle = plt.rcParams.get("axes.prop_cycle")
        if cycle is not None:
            default_colors = cycle.by_key().get("color", [])
            if default_colors:
                colors = [
                    default_colors[i % len(default_colors)]
                    for i in range(len(group_order))
                ]
    color_map = {}
    if colors is not None:
        color_map = dict(zip(group_order, colors))

    # Create figure or use provided axes
    created_fig = False
    if ax_obj is None:
        if figsize is None:
            figsize = (12, 1.5 + len(group_order) * 0.8)
        fig, _ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        _ax = ax_obj

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
        y_labels.append(group_name)
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

    return _ax, created_fig


def peptides_on_sequence(
    adata: ad.AnnData,
    protein_id: str,
    group_by: str | None = None,
    ref_sequence: str | None = None,
    ref_sequence_key_chain: tuple | None = None,
    alt_pep_sequence_key: str | None = None,
    add_sequences: dict[str, dict] | None = None,
    add_sequences_key_chain: tuple | None = None,
    allow_overlaps: bool = False,
    allow_multi_match: bool = False,
    color_scheme=None,
    title: str | None = None,
    order: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes | None:
    _, level = check_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "peptides_on_sequence requires peptide-level "
            "data. The provided AnnData is at the "
            f"'{level}' level."
        )

    # Validate mutual exclusivity of ref_sequence args
    if ref_sequence is not None and ref_sequence_key_chain is not None:
        raise ValueError(
            "Provide either 'ref_sequence' or "
            "'ref_sequence_key_chain', not both."
        )
    if ref_sequence is None and ref_sequence_key_chain is None:
        raise ValueError(
            "Provide either 'ref_sequence' or "
            "'ref_sequence_key_chain' to supply the "
            "reference protein sequence."
        )

    # Resolve reference sequence
    if ref_sequence_key_chain is not None:
        obj = adata.uns
        for key in ref_sequence_key_chain:
            if not isinstance(obj, dict) or key not in obj:
                raise KeyError(
                    f"Key '{key}' not found while "
                    f"traversing .uns via "
                    f"ref_sequence_key_chain "
                    f"{ref_sequence_key_chain}."
                )
            obj = obj[key]
        if not isinstance(obj, str):
            raise TypeError(
                "The value at ref_sequence_key_chain "
                f"{ref_sequence_key_chain} must be a "
                f"string, got {type(obj).__name__}."
            )
        ref_sequence = obj

    # Filter .var to matching protein_id
    var = adata.var
    mask = var["protein_id"] == protein_id
    if not mask.any():
        raise ValueError(
            f"No peptides found for protein_id "
            f"'{protein_id}' in adata.var."
        )
    var_sub = var.loc[mask]

    # Get peptide sequences
    if alt_pep_sequence_key is not None:
        if alt_pep_sequence_key not in var_sub.columns:
            raise KeyError(
                f"Column '{alt_pep_sequence_key}' not "
                f"found in adata.var."
            )
        pep_seqs = var_sub[alt_pep_sequence_key]
    else:
        pep_seqs = var_sub.index

    # Get groups
    if group_by is not None:
        if group_by not in var_sub.columns:
            raise KeyError(
                f"Column '{group_by}' not found in "
                f"adata.var."
            )
        groups = var_sub[group_by]
    else:
        groups = pd.Series(
            protein_id, index=var_sub.index,
        )

    # Build sequences dict
    sequences = {}
    for pep_id, seq_val, grp_val in zip(
        var_sub.index, pep_seqs, groups,
    ):
        if pd.isna(grp_val):
            continue
        sequences[pep_id] = {
            "seq": str(seq_val),
            "group": str(grp_val),
        }

    # Validate mutual exclusivity of add_sequences args
    if add_sequences is not None:
        if add_sequences_key_chain is not None:
            raise ValueError(
                "Provide either 'add_sequences' or "
                "'add_sequences_key_chain', not both."
            )

    # Merge additional sequences
    add_seqs = None
    if add_sequences is not None:
        add_seqs = add_sequences
    elif add_sequences_key_chain is not None:
        obj = adata.uns
        for key in add_sequences_key_chain:
            if not isinstance(obj, dict) or key not in obj:
                raise KeyError(
                    f"Key '{key}' not found while "
                    f"traversing .uns via "
                    f"add_sequences_key_chain "
                    f"{add_sequences_key_chain}."
                )
            obj = obj[key]
        if not isinstance(obj, dict):
            raise TypeError(
                "The value at add_sequences_key_chain "
                f"{add_sequences_key_chain} must be a "
                f"dict, got {type(obj).__name__}."
            )
        add_seqs = obj

    if add_seqs is not None:
        conflicts = set(sequences) & set(add_seqs)
        if conflicts:
            raise ValueError(
                f"Name conflicts between peptide "
                f"sequences and additional sequences: "
                f"{sorted(conflicts)}."
            )
        sequences.update(add_seqs)

    # Resolve title: default to protein_id
    if title is None:
        title = protein_id

    # Plot
    _ax, created_fig = _plot_sequences_on_reference(
        ref_sequence=ref_sequence,
        sequences=sequences,
        color_scheme=color_scheme,
        allow_overlaps=allow_overlaps,
        allow_multi_match=allow_multi_match,
        title=title,
        order=order,
        figsize=figsize,
        ax_obj=ax,
    )

    fig = _ax.figure

    if created_fig:
        fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    if ax is not None:
        return _ax
    return None
