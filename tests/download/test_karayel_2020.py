"""Tests for proteopy.download.karayel_2020."""
import hashlib

import pandas as pd
import pytest

from proteopy.download import karayel_2020


# -- Expected values -------------------------------------------------

_EXPECTED_INTENSITIES_HASH = (
    "0a87e35cba89def63e8745776728d1d9"
    "2510fb2ecee1a3cf7dc092881cf7c660"
)
_EXPECTED_VAR_HASH = (
    "1932d3b6568ef923fca9079a1fa1915c"
    "ea00fca33c9094c1b4b9443584967e73"
)
_EXPECTED_SAMPLE_HASH = (
    "996521c86b23958ec642d531a79c9c7f"
    "28dc8676ad9cc261a7ec86bf1feaa012"
)

_EXPECTED_INTENSITIES_COLUMNS = [
    "sample_id", "protein_id", "intensity",
]
_EXPECTED_VAR_COLUMNS = ["protein_id", "gene_id"]
_EXPECTED_SAMPLE_COLUMNS = [
    "sample_id", "cell_type", "replicate",
]
_EXPECTED_CELL_TYPES = [
    "LBaso", "Ortho", "Poly", "ProE&EBaso", "Progenitor",
]


# -- Helpers ---------------------------------------------------------

def _make_paths(tmp_path, ext=".tsv"):
    return (
        tmp_path / f"intensities{ext}",
        tmp_path / f"var_annotation{ext}",
        tmp_path / f"sample_annotation{ext}",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- Content tests ---------------------------------------------------

class TestKarayel2020Download:
    """Verify downloaded file content and structure."""

    @pytest.fixture(scope="class")
    def paths(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("karayel_dl")
        p = _make_paths(tmp)
        karayel_2020(*p)
        return p

    def test_files_created(self, paths):
        for p in paths:
            assert p.exists()

    def test_intensities_columns(self, paths):
        df = pd.read_csv(paths[0], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )

    def test_var_annotation_columns(self, paths):
        df = pd.read_csv(paths[1], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_VAR_COLUMNS

    def test_sample_annotation_columns(self, paths):
        df = pd.read_csv(paths[2], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_SAMPLE_COLUMNS
        )

    def test_intensities_hash(self, paths):
        assert (
            _sha256(paths[0].read_bytes())
            == _EXPECTED_INTENSITIES_HASH
        )

    def test_var_annotation_hash(self, paths):
        assert (
            _sha256(paths[1].read_bytes())
            == _EXPECTED_VAR_HASH
        )

    def test_sample_annotation_hash(self, paths):
        assert (
            _sha256(paths[2].read_bytes())
            == _EXPECTED_SAMPLE_HASH
        )

    def test_sample_count(self, paths):
        df = pd.read_csv(paths[2], sep="\t")
        assert len(df) == 20

    def test_cell_types_in_file(self, paths):
        df = pd.read_csv(paths[2], sep="\t")
        assert (
            sorted(df["cell_type"].unique())
            == _EXPECTED_CELL_TYPES
        )


# -- Separator tests -------------------------------------------------

class TestKarayel2020DownloadSeparator:
    """Verify separator auto-detection from file extension."""

    def test_csv_extension_uses_comma(self, tmp_path):
        p = _make_paths(tmp_path, ext=".csv")
        karayel_2020(*p)
        df = pd.read_csv(p[0], sep=",", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )

    def test_tsv_extension_uses_tab(self, tmp_path):
        p = _make_paths(tmp_path, ext=".tsv")
        karayel_2020(*p)
        df = pd.read_csv(p[0], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )


# -- Error tests -----------------------------------------------------

class TestKarayel2020DownloadErrors:
    """Verify input validation and error handling."""

    def test_file_exists_error(self, tmp_path):
        p = _make_paths(tmp_path)
        karayel_2020(*p)
        with pytest.raises(FileExistsError):
            karayel_2020(*p)

    def test_force_overwrites(self, tmp_path):
        p = _make_paths(tmp_path)
        karayel_2020(*p)
        karayel_2020(*p, force=True)
        for path in p:
            assert path.exists()

    def test_overlapping_paths_raises(self, tmp_path):
        same = tmp_path / "same.tsv"
        with pytest.raises(ValueError, match="same path"):
            karayel_2020(same, same, tmp_path / "other.tsv")

    def test_invalid_path_type_raises(self, tmp_path):
        with pytest.raises(
            TypeError, match="must be str or Path",
        ):
            karayel_2020(
                123,
                tmp_path / "v.tsv",
                tmp_path / "s.tsv",
            )

    def test_invalid_sep_type_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="sep must be str or None",
        ):
            karayel_2020(*p, sep=123)

    def test_fill_na_bool_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="fill_na must be",
        ):
            karayel_2020(*p, fill_na=True)

    def test_force_non_bool_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="force must be bool",
        ):
            karayel_2020(*p, force=1)
