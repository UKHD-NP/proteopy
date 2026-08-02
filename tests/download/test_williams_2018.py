"""Tests for proteopy.download.williams_2018."""

import hashlib

import pandas as pd
import pytest

from proteopy.download import williams_2018


# -- Expected values -------------------------------------------------

# Changed by the missing-value fix in datasets.williams_2018. The var
# and sample annotation hashes below are deliberately UNCHANGED: the
# fix touches .X only, and their staying put is part of the evidence
# that it is correctly scoped.
_EXPECTED_INTENSITIES_HASH = (
    "0444cade741974e18c9d04ff4661bb7b" "6c84437cc23f2e643b097a1ad7844012"
)
_EXPECTED_VAR_HASH = (
    "827b32fd2962cd18a7a990d56eab0e64" "daa2a244b6226fe2d242106f185b2161"
)
_EXPECTED_SAMPLE_HASH = (
    "8cca98fa3a38df78b78912f3ef7daed5" "7f82902485d61d90db5a823c1ed4f031"
)

_EXPECTED_INTENSITIES_COLUMNS = [
    "sample_id",
    "peptide_id",
    "intensity",
]
_EXPECTED_VAR_COLUMNS = [
    "peptide_id",
    "protein_id",
    "gene_id",
]
_EXPECTED_SAMPLE_COLUMNS = [
    "sample_id",
    "tissue",
    "mouse_id",
]
_EXPECTED_TISSUES = [
    "BAT",
    "Brain",
    "Heart",
    "Liver",
    "Quad",
]


# -- Helpers ---------------------------------------------------------


def _files(tmp_path, ext=".tsv"):
    return (
        tmp_path / f"intensities{ext}",
        tmp_path / f"var_annotation{ext}",
        tmp_path / f"sample_annotation{ext}",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_zero(token: str) -> bool:
    """True if a raw field parses to exactly 0.0, False if non-numeric."""
    try:
        return float(token) == 0.0
    except ValueError:
        return False


# -- Content tests ---------------------------------------------------


class TestWilliams2018Download:
    """Verify downloaded file content, structure, and error handling."""

    @pytest.fixture(scope="class")
    def files(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("williams_dl")
        p = _files(tmp)
        williams_2018(*p)
        return p

    def test_files_created(self, files):
        for p in files:
            assert p.exists()

    def test_intensities_columns(self, files):
        df = pd.read_csv(files[0], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_INTENSITIES_COLUMNS

    def test_var_annotation_columns(self, files):
        df = pd.read_csv(files[1], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_VAR_COLUMNS

    def test_sample_annotation_columns(self, files):
        df = pd.read_csv(files[2], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_SAMPLE_COLUMNS

    def test_intensities_hash(self, files):
        assert _sha256(files[0].read_bytes()) == _EXPECTED_INTENSITIES_HASH

    def test_var_annotation_hash(self, files):
        assert _sha256(files[1].read_bytes()) == _EXPECTED_VAR_HASH

    def test_sample_annotation_hash(self, files):
        assert _sha256(files[2].read_bytes()) == _EXPECTED_SAMPLE_HASH

    def test_zeros_and_missing_survive_serialisation(self, files):
        """A zero must round-trip as 0.0, not as an empty field.

        The dataset loader keeps zeros and missing values distinct;
        this checks that writing to TSV does not collapse them again.
        Read with pandas' NA handling disabled so that the two remain
        distinguishable in the parsed frame.
        """
        df = pd.read_csv(
            files[0],
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
        values = df["intensity"]
        n_zero = (values.map(_is_zero)).sum()
        n_missing = (values == "").sum()
        assert n_zero == 13547
        assert n_missing == 3584

    def test_sample_count(self, files):
        df = pd.read_csv(files[2], sep="\t")
        assert len(df) == 40

    def test_tissues_in_file(self, files):
        df = pd.read_csv(files[2], sep="\t")
        assert sorted(df["tissue"].unique()) == _EXPECTED_TISSUES

    def test_csv_extension_uses_comma(self, tmp_path):
        p = _files(tmp_path, ext=".csv")
        williams_2018(*p)
        df = pd.read_csv(p[0], sep=",", nrows=0)
        assert df.columns.tolist() == _EXPECTED_INTENSITIES_COLUMNS

    def test_tsv_extension_uses_tab(self, tmp_path):
        p = _files(tmp_path, ext=".tsv")
        williams_2018(*p)
        df = pd.read_csv(p[0], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_INTENSITIES_COLUMNS

    def test_file_exists_error(self, tmp_path):
        p = _files(tmp_path)
        williams_2018(*p)
        with pytest.raises(FileExistsError):
            williams_2018(*p)

    def test_force_overwrites(self, tmp_path):
        p = _files(tmp_path)
        dummy = b"dummy"
        for path in p:
            path.write_bytes(dummy)
        williams_2018(*p, force=True)
        for path in p:
            assert path.read_bytes() != dummy
        assert _sha256(p[0].read_bytes()) == _EXPECTED_INTENSITIES_HASH
        assert _sha256(p[1].read_bytes()) == _EXPECTED_VAR_HASH
        assert _sha256(p[2].read_bytes()) == _EXPECTED_SAMPLE_HASH

    def test_overlapping_paths_raises(self, tmp_path):
        same = tmp_path / "same.tsv"
        with pytest.raises(ValueError, match="same path"):
            williams_2018(same, same, tmp_path / "other.tsv")

    def test_invalid_path_type_raises(self, tmp_path):
        with pytest.raises(
            TypeError,
            match="must be str or Path",
        ):
            williams_2018(
                123,
                tmp_path / "v.tsv",
                tmp_path / "s.tsv",
            )

    def test_invalid_sep_type_raises(self, tmp_path):
        p = _files(tmp_path)
        with pytest.raises(
            TypeError,
            match="sep must be str or None",
        ):
            williams_2018(*p, sep=123)

    def test_fill_na_bool_raises(self, tmp_path):
        p = _files(tmp_path)
        with pytest.raises(
            TypeError,
            match="fill_na must be",
        ):
            williams_2018(*p, fill_na=True)

    def test_force_non_bool_raises(self, tmp_path):
        p = _files(tmp_path)
        with pytest.raises(
            TypeError,
            match="force must be bool",
        ):
            williams_2018(*p, force=1)

    def test_fill_na_zero_removes_nan(self, tmp_path):
        p = _files(tmp_path)
        williams_2018(*p, fill_na=0)
        df = pd.read_csv(p[0], sep="\t")
        assert not df["intensity"].isna().any()

    def test_zero_to_na_removes_zeros(self, tmp_path):
        p = _files(tmp_path)
        williams_2018(*p, zero_to_na=True)
        df = pd.read_csv(
            p[0],
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
        values = df["intensity"]
        assert values.map(_is_zero).sum() == 0
        assert (values == "").sum() == 3584 + 13547

    def test_zero_to_na_with_fill_na_raises(self, tmp_path):
        p = _files(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            williams_2018(*p, zero_to_na=True, fill_na=0)

    def test_zero_to_na_non_bool_raises(self, tmp_path):
        p = _files(tmp_path)
        with pytest.raises(TypeError, match="zero_to_na must be bool"):
            williams_2018(*p, zero_to_na="yes")

    def test_default_intensities_contain_nan(self, tmp_path):
        p = _files(tmp_path)
        williams_2018(*p)
        df = pd.read_csv(p[0], sep="\t")
        assert df["intensity"].isna().any()
