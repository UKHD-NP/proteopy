import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from proteopy.pl import peptide_intensities, peptides_on_prot_sequence

matplotlib.use("Agg")

REF = "MKTAYIAKQRQISFVKSHFSRQDEEP"
PEPTIDES = ["KTAYIAK", "QRQISFVK", "SHFSRQ", "ELPGW"]


@pytest.fixture
def adata():
    var = pd.DataFrame(
        {
            "peptide_id": PEPTIDES,
            "protein_id": ["P1", "P1", "P1", "P2"],
            "protein_sequence": [REF, REF, REF, "ELPGWK"],
        },
        index=PEPTIDES,
    )
    obs = pd.DataFrame({"sample_id": ["S1", "S2"]}, index=["S1", "S2"])
    return ad.AnnData(X=np.ones((2, 4)), obs=obs, var=var)


@pytest.fixture
def fasta_path(tmp_path):
    path = tmp_path / "proteome.fasta"
    path.write_text(
        ">sp|P2|PROT2_HUMAN Protein 2\n"
        "ELPGWK\n"
        ">sp|P1|PROT1_HUMAN Protein 1\n"
        f"{REF[:10]}\n"
        f"{REF[10:]}\n"
    )
    return path


def _bar_spans(ax):
    """Return the 0-based bar start positions of each row, bottom first."""
    return [
        sorted(int(p.vertices[:, 0].min()) for p in coll.get_paths())
        for coll in ax.collections
    ]


def _plot(adata, **kwargs):
    _, ax = plt.subplots()
    peptides_on_prot_sequence(
        adata,
        protein_id="P1",
        show=False,
        ax=ax,
        **kwargs,
    )
    return ax


def test_ref_sequence_literal_string(adata):
    ax = _plot(adata, ref_sequence=REF)
    assert ax.get_xlim() == (0, len(REF))
    plt.close("all")


def test_ref_sequence_from_var_column(adata):
    ax = _plot(adata, ref_sequence="protein_sequence")
    assert ax.get_xlim() == (0, len(REF))
    assert _bar_spans(ax)[1] == [1, 8, 16]
    plt.close("all")


def test_ref_sequence_var_column_inconsistent(adata):
    adata.var.loc["SHFSRQ", "protein_sequence"] = REF + "K"
    with pytest.raises(ValueError, match="different sequences"):
        _plot(adata, ref_sequence="protein_sequence")


def test_ref_sequence_var_column_missing_value(adata):
    adata.var["protein_sequence"] = adata.var["protein_sequence"].astype(
        object
    )
    adata.var.loc["SHFSRQ", "protein_sequence"] = None
    with pytest.raises(ValueError, match="missing values"):
        _plot(adata, ref_sequence="protein_sequence")


def test_ref_sequence_var_column_unknown_protein(adata):
    with pytest.raises(ValueError, match="No peptides found"):
        peptides_on_prot_sequence(
            adata,
            protein_id="P9",
            ref_sequence="protein_sequence",
            show=False,
        )


def test_ref_sequence_from_fasta(adata, fasta_path):
    ax = _plot(adata, ref_sequence=str(fasta_path))
    assert ax.get_xlim() == (0, len(REF))
    assert _bar_spans(ax)[1] == [1, 8, 16]
    plt.close("all")


def test_ref_sequence_fasta_suffix_case_insensitive(adata, fasta_path):
    upper = fasta_path.with_name("proteome.FASTA")
    fasta_path.rename(upper)
    ax = _plot(adata, ref_sequence=str(upper))
    assert ax.get_xlim() == (0, len(REF))
    plt.close("all")


def test_ref_sequence_fasta_takes_precedence_over_var(adata, fasta_path):
    adata.var[str(fasta_path)] = "MKTAYIAKQRQISFVKSHFSRQDEEPAAAA"
    ax = _plot(adata, ref_sequence=str(fasta_path))
    assert ax.get_xlim() == (0, len(REF))
    plt.close("all")


def test_ref_sequence_fasta_missing_accession(adata, tmp_path):
    path = tmp_path / "other.fasta"
    path.write_text(">sp|P2|PROT2_HUMAN\nELPGWK\n")
    with pytest.raises(ValueError, match="not found in FASTA"):
        _plot(adata, ref_sequence=str(path))


def test_ref_sequence_fasta_file_not_found(adata, tmp_path):
    with pytest.raises(FileNotFoundError):
        _plot(adata, ref_sequence=str(tmp_path / "absent.fasta"))


def test_peptide_sequences_from_peptide_id_not_var_names(adata):
    adata.var_names = [f"pep{i}" for i in range(adata.n_vars)]
    adata.var["peptide_id"] = adata.var_names
    adata.var["stripped"] = PEPTIDES
    # var_names are no longer amino-acid strings, so only the
    # alt_pep_sequence_key column locates the peptides.
    ax = _plot(
        adata,
        ref_sequence=REF,
        alt_pep_sequence_key="stripped",
    )
    assert _bar_spans(ax)[1] == [1, 8, 16]
    plt.close("all")


# 1-based closed coordinates, as in UniProt GFF downloads.
GFF = (
    "##gff-version 3\n"
    "P1\tUniProtKB\tChain\t1\t26\t.\t.\t.\tID=PRO_1;Note=Protein 1\t\n"
    "P1\tUniProtKB\tAlternative sequence\t1\t3\t.\t.\t.\t"
    "ID=VSP_1;Note=In isoform 2%2C isoform 3. MKT->MA;"
    "evidence=ECO:0000303\t\n"
    "P1\tUniProtKB\tAlternative sequence\t20\t26\t.\t.\t.\t"
    "ID=VSP_2;Note=In isoform 2. Missing\t\n"
    "P1\tUniProtKB\tAlternative sequence\t9\t12\t.\t.\t.\t"
    "ID=VSP_3;Note=In isoform 3. QRQI->Q\t\n"
    "P2\tUniProtKB\tAlternative sequence\t1\t2\t.\t.\t.\t"
    "ID=VSP_4;Note=In isoform 2. Missing\t\n"
)


@pytest.fixture
def gff_path(tmp_path):
    path = tmp_path / "proteome.gff"
    path.write_text(GFF)
    return path


def _rows(ax):
    """Map each y-tick label to the 0-based bar starts of its row."""
    labels = [t.get_text() for t in ax.get_yticklabels()]
    return dict(zip(labels, _bar_spans(ax)))


def test_add_sequences_gff_rows_per_isoform(adata, gff_path):
    ax = _plot(adata, ref_sequence=REF, add_sequences=str(gff_path))
    rows = _rows(ax)
    assert rows["isoform 2"] == [0, 19]
    assert rows["isoform 3"] == [0, 8]
    assert rows["P1"] == [1, 8, 16]
    plt.close("all")


def test_add_sequences_gff_half_open_ends(adata, gff_path):
    ax = _plot(adata, ref_sequence=REF, add_sequences=str(gff_path))
    labels = [t.get_text() for t in ax.get_yticklabels()]
    coll = ax.collections[labels.index("isoform 2")]
    ends = sorted(int(p.vertices[:, 0].max()) for p in coll.get_paths())
    assert ends == [3, len(REF)]
    plt.close("all")


def test_add_sequences_gff_suffix_case_insensitive(adata, gff_path):
    upper = gff_path.with_name("proteome.GFF")
    gff_path.rename(upper)
    ax = _plot(adata, ref_sequence=REF, add_sequences=str(upper))
    assert "isoform 2" in _rows(ax)
    plt.close("all")


def test_add_sequences_gff_with_fasta_reference(
    adata,
    fasta_path,
    gff_path,
):
    ax = _plot(
        adata,
        ref_sequence=str(fasta_path),
        add_sequences=str(gff_path),
    )
    assert _rows(ax)["isoform 3"] == [0, 8]
    plt.close("all")


def test_add_sequences_gff_reference_mismatch(adata, gff_path):
    with pytest.raises(ValueError, match="canonical"):
        _plot(
            adata,
            ref_sequence="MAAAYIAKQRQISFVKSHFSRQDEEP",
            add_sequences=str(gff_path),
        )


def test_add_sequences_gff_no_features_warns(adata, tmp_path):
    path = tmp_path / "empty.gff"
    path.write_text("##gff-version 3\n")
    with pytest.warns(UserWarning, match="No 'Alternative sequence'"):
        ax = _plot(adata, ref_sequence=REF, add_sequences=str(path))
    assert list(_rows(ax)) == ["Reference", "P1"]
    plt.close("all")


def test_add_sequences_unsupported_string(adata):
    with pytest.raises(ValueError, match="'.gff'"):
        _plot(adata, ref_sequence=REF, add_sequences="isoforms.tsv")


def _row_labels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def test_row_order_peptide_groups_below_isoforms(adata):
    adata.var["proteoform_id"] = ["P1_1", "P1_0", "P1_1", "P2_0"]
    add = {
        f"VSP_{n}": {"seq_coord": (0, 3), "group": f"isoform {n}"}
        for n in (10, 2, 9)
    }
    ax = _plot(
        adata,
        ref_sequence=REF,
        add_sequences=add,
        group_by="proteoform_id",
    )
    assert _row_labels(ax) == [
        "Reference",
        "P1_0",
        "P1_1",
        "isoform 2",
        "isoform 9",
        "isoform 10",
    ]
    plt.close("all")


def test_row_order_lexicographic_peptide_groups(adata, gff_path):
    # "variable" sorts after "isoform …" but must stay below it.
    adata.var["region"] = ["variable", "constant", "variable", "constant"]
    ax = _plot(
        adata,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        group_by="region",
    )
    assert _row_labels(ax) == [
        "Reference",
        "constant",
        "variable",
        "isoform 2",
        "isoform 3",
    ]
    plt.close("all")


def test_row_order_categorical_peptide_groups(adata, gff_path):
    adata.var["region"] = pd.Categorical(
        ["A", "B", "A", "Z"],
        categories=["B", "A", "Z"],
        ordered=True,
    )
    ax = _plot(
        adata,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        group_by="region",
    )
    # "Z" belongs to P2 only, so it has no row.
    assert _row_labels(ax) == [
        "Reference",
        "B",
        "A",
        "isoform 2",
        "isoform 3",
    ]
    plt.close("all")


def test_row_order_explicit_order_wins(adata, gff_path):
    ax = _plot(
        adata,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        order=["isoform 3", "P1"],
    )
    assert _row_labels(ax) == ["Reference", "isoform 3", "P1"]
    plt.close("all")


def test_shared_label_kept_on_separate_rows(adata):
    add = {"dom": {"seq_coord": (0, 3), "group": "P1"}}
    ax = _plot(adata, ref_sequence=REF, add_sequences=add)
    assert _row_labels(ax) == ["Reference", "P1", "P1"]
    # Peptide row below, add_sequences row above.
    assert _bar_spans(ax)[1:] == [[1, 8, 16], [0]]
    plt.close("all")


def test_shared_label_in_order_selects_both_rows(adata):
    add = {"dom": {"seq_coord": (0, 3), "group": "P1"}}
    ax = _plot(
        adata,
        ref_sequence=REF,
        add_sequences=add,
        order=["P1"],
    )
    assert _row_labels(ax) == ["Reference", "P1", "P1"]
    assert _bar_spans(ax)[1:] == [[1, 8, 16], [0]]
    plt.close("all")


def test_overlap_error_names_drawn_label(adata):
    add = {
        "a": {"seq_coord": (0, 5), "group": "domains"},
        "b": {"seq_coord": (3, 8), "group": "domains"},
    }
    with pytest.raises(ValueError, match="group 'domains'"):
        _plot(adata, ref_sequence=REF, add_sequences=add)


def _line_colors(adata, **kwargs):
    """Map each proteoform to its line color in proteoform_intensities.

    ``proteoform_intensities`` is ``peptide_intensities`` with
    ``color="proteoform_id"``; it is called directly because pylint
    cannot see through the partial's signature.
    """
    adata.X = (
        np.arange(adata.n_obs * adata.n_vars, dtype=float).reshape(
            adata.n_obs, adata.n_vars
        )
        + 1
    )
    _, ax = plt.subplots()
    peptide_intensities(
        adata,
        protein_ids="P1",
        color="proteoform_id",
        show=False,
        ax=ax,
        **kwargs,
    )
    legend = ax.get_legend()
    colors = {
        text.get_text(): to_hex(handle.get_color())
        for text, handle in zip(legend.get_texts(), legend.legend_handles)
    }
    plt.close("all")
    return colors


def _bar_colors(ax):
    """Map each row label to its bar color (last row wins on clashes)."""
    return {
        label: to_hex(coll.get_facecolor()[0])
        for label, coll in zip(_row_labels(ax), ax.collections)
    }


@pytest.fixture
def adata_pf(adata):
    adata.var["proteoform_id"] = ["P1_1", "P1_0", "P1_1", "P2_0"]
    return adata


PROTEOFORM_COLOR_CASES = {
    "default": {},
    "colormap": {"color_scheme": "viridis"},
    "color_list": {"color_scheme": ["red", "green", "blue", "black"]},
}


@pytest.mark.parametrize("case", PROTEOFORM_COLOR_CASES)
def test_proteoform_colors_match_proteoform_intensities(
    adata_pf,
    gff_path,
    case,
):
    kwargs = PROTEOFORM_COLOR_CASES[case]
    ax = _plot(
        adata_pf,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        group_by="proteoform_id",
        **kwargs,
    )
    bars = _bar_colors(ax)
    plt.close("all")
    color_kwargs = {k: v for k, v in kwargs.items() if k == "color_scheme"}
    lines = _line_colors(adata_pf, **color_kwargs)
    assert {pf: bars[pf] for pf in ("P1_0", "P1_1")} == lines


def test_proteoform_colors_follow_category_order(adata_pf):
    adata_pf.var["proteoform_id"] = pd.Categorical(
        adata_pf.var["proteoform_id"],
        categories=["P2_0", "P1_1", "P1_0"],
        ordered=True,
    )
    ax = _plot(adata_pf, ref_sequence=REF, group_by="proteoform_id")
    assert _row_labels(ax) == ["Reference", "P1_1", "P1_0"]
    bars = _bar_colors(ax)
    plt.close("all")
    # P2_0 has no peptides of P1, so P1_1 takes the first color.
    assert bars == {
        "Reference": to_hex("lightgrey"),
        "P1_1": to_hex("C0"),
        "P1_0": to_hex("C1"),
    }
    assert {pf: bars[pf] for pf in ("P1_0", "P1_1")} == _line_colors(adata_pf)


def test_explicit_order_recolors_by_row_position(adata_pf, gff_path):
    ax = _plot(
        adata_pf,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        group_by="proteoform_id",
        order=["isoform 3", "P1_1", "isoform 2", "P1_0"],
    )
    assert _bar_colors(ax) == {
        "Reference": to_hex("lightgrey"),
        "isoform 3": to_hex("C2"),
        "P1_1": to_hex("C0"),
        "isoform 2": to_hex("C3"),
        "P1_0": to_hex("C1"),
    }
    plt.close("all")


def test_add_sequences_colors_continue_after_peptide_groups(
    adata_pf,
    gff_path,
):
    ax = _plot(
        adata_pf,
        ref_sequence=REF,
        add_sequences=str(gff_path),
        group_by="proteoform_id",
    )
    assert _bar_colors(ax) == {
        "Reference": to_hex("lightgrey"),
        "P1_0": to_hex("C0"),
        "P1_1": to_hex("C1"),
        "isoform 2": to_hex("C2"),
        "isoform 3": to_hex("C3"),
    }
    plt.close("all")


def test_missing_group_values_are_not_plotted(adata):
    adata.var["region"] = pd.Series(
        ["A", None, "B", "A"],
        index=adata.var_names,
        dtype=object,
    )
    ax = _plot(adata, ref_sequence=REF, group_by="region")
    # QRQISFVK (start 8) has no region: no bar and no "nan"/"NA" row.
    assert _row_labels(ax) == ["Reference", "A", "B"]
    assert _bar_spans(ax)[1:] == [[1], [16]]
    plt.close("all")
