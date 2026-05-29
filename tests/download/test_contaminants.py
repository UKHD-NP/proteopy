"""
Unit tests for ``proteopy.download.contaminants``.

Download fidelity is checked by SHA-256 hashing the resulting file and
comparing it against the hash of the expected payload (computed inline
from in-code byte constants for mocked tests, or pinned for the two
real-download tests).
"""

import hashlib
import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ``proteopy.download.__init__`` re-exports ``contaminants`` (the function),
# shadowing the submodule name on attribute lookup. Resolve via sys.modules
# to get the actual module object for ``patch.object``.
contam_mod = importlib.import_module("proteopy.download.contaminants")
from proteopy.download.contaminants import (  # noqa: E402
    _is_uniprot_accession,
    check_uniprot_accession_nr,
    contaminants,
)


# ---------------------------------------------------------------------------
# In-code FASTA payloads
# ---------------------------------------------------------------------------

FRANKENFIELD_RAW = (
    b">sp|P12345|HUMAN_PROT first description\n"
    b"MAAAAACDEFGHIKLMNPQRSTVWY\n"
    b">sp|Cont_P67890|MOUSE_PROT contaminant entry\n"
    b"MGGGGGHHHIIIKKKLLL\n"
    b">sp|AAAA1|MANUAL_ID manually curated entry\n"
    b"KKLLLMMNN\n"
)

# Byte-exact output produced by ``_format_fasta`` with the
# ``_format_frankenfield_header`` formatter applied to FRANKENFIELD_RAW.
# Only difference: the ``Cont_`` prefix on the second accession is stripped.
FRANKENFIELD_FORMATTED = (
    b">sp|P12345|HUMAN_PROT first description\n"
    b"MAAAAACDEFGHIKLMNPQRSTVWY\n"
    b">sp|P67890|MOUSE_PROT contaminant entry\n"
    b"MGGGGGHHHIIIKKKLLL\n"
    b">sp|AAAA1|MANUAL_ID manually curated entry\n"
    b"KKLLLMMNN\n"
)

GPM_RAW = (
    b">sp|P00001|CRAP_ENTRY1 example cRAP entry\n"
    b"MAAAACDEF\n"
    b">sp|P00002|CRAP_ENTRY2 second entry\n"
    b"MGGGGGHHH\n"
)

# Pinned hashes for real-download tests. Re-pin if upstream rotates content.
# Recorded 2026-05-11 from the URLs in ``contam_mod._SOURCE_MAP``.
EXPECTED_FRANKENFIELD_REMOTE_HASH = (
    "b4c1c74438e3d60ee93546a4b717225da318d7c4c30344b91fef4cb8cf6e9f89"
)
EXPECTED_GPM_REMOTE_HASH = (
    "4b0e6e97ab1d618baa38be612a787d884a781e07f627b919c4b29ac064db5382"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_download_mock(payload: bytes):
    """Return a fake ``_download(url, dest)`` that writes payload to dest."""

    def fake(url, dest):
        Path(dest).write_bytes(payload)

    return fake


# ---------------------------------------------------------------------------
# 1. _is_uniprot_accession
# ---------------------------------------------------------------------------


class TestIsUniprotAccession:
    @pytest.mark.parametrize(
        "accession",
        [
            "P12345",       # Swiss-Prot, [OPQ] branch
            "O00001",       # Swiss-Prot, [OPQ] branch with zeros
            "Q9Y6K9",       # Swiss-Prot, mixed alphanumeric
            "A0A0A0A0A1",   # TrEMBL 10-char, [A-NR-Z] branch
            "P12345-2",     # isoform suffix
            "A0A0A0A0A1-12",  # TrEMBL with two-digit isoform
        ],
    )
    def test_valid_accessions(self, accession):
        assert _is_uniprot_accession(accession) is True

    @pytest.mark.parametrize(
        "accession",
        [
            "",             # empty
            "p12345",       # lowercase
            "P1234",        # too short
            "P123456",      # too long for [OPQ] branch
            "X12345",       # second char not a letter-then-3 pattern
            "12345P",       # leading digit
            "1ABCDE",       # leading digit
            "P12345-",      # dangling isoform separator
            "P12345-123",   # isoform suffix too long
        ],
    )
    def test_invalid_accessions(self, accession):
        assert _is_uniprot_accession(accession) is False


# ---------------------------------------------------------------------------
# 2. check_uniprot_accession_nr
# ---------------------------------------------------------------------------


class TestCheckUniprotAccessionNr:
    def test_valid_returns_none(self):
        assert check_uniprot_accession_nr("P12345") is None

    @pytest.mark.parametrize("accession", ["", "p12345", "BADID", "12345"])
    def test_invalid_raises_value_error(self, accession):
        with pytest.raises(ValueError, match="not a valid UniProt accession"):
            check_uniprot_accession_nr(accession)


# ---------------------------------------------------------------------------
# 3. contaminants (mocked + opt-in real-download)
#
# Frankenfield-header and FASTA-rewrite behaviour is exercised through the
# public ``contaminants()`` function only; the private helpers
# ``_format_frankenfield_header`` and ``_format_fasta`` are not tested
# directly.
# ---------------------------------------------------------------------------


class TestContaminants:
    # -- error inputs --------------------------------------------------------

    def test_unsupported_source_raises(self):
        with pytest.raises(ValueError, match="Unsupported source"):
            contaminants(source="bogus")

    # -- happy paths with hash checks ---------------------------------------

    def test_gpm_crap_hash_matches_raw_payload(self, tmp_path):
        dst = tmp_path / "gpm.fasta"
        with patch.object(
            contam_mod, "_download", _make_download_mock(GPM_RAW),
        ):
            result = contaminants(source="gpm_crap", path=dst)

        assert result == dst
        assert result.exists()
        assert _sha256(result) == _sha256_bytes(GPM_RAW)

    def test_frankenfield_hash_matches_formatted_payload(self, tmp_path):
        """Confirms ``Cont_`` prefixes are stripped byte-exactly."""
        dst = tmp_path / "frank.fasta"
        with patch.object(
            contam_mod,
            "_download",
            _make_download_mock(FRANKENFIELD_RAW),
        ):
            result = contaminants(source="frankenfield2022", path=dst)

        assert result == dst
        assert result.exists()
        assert _sha256(result) == _sha256_bytes(FRANKENFIELD_FORMATTED)

    @pytest.mark.parametrize(
        "source,payload,expected",
        [
            ("gpm_crap", GPM_RAW, GPM_RAW),
            ("frankenfield2022", FRANKENFIELD_RAW, FRANKENFIELD_FORMATTED),
        ],
    )
    def test_returns_path_to_destination(
        self, tmp_path, source, payload, expected,
    ):
        dst = tmp_path / f"{source}.fasta"
        with patch.object(
            contam_mod, "_download", _make_download_mock(payload),
        ):
            result = contaminants(source=source, path=dst)

        assert isinstance(result, Path)
        assert result == dst
        assert _sha256(result) == _sha256_bytes(expected)

    # -- default path with date suffix --------------------------------------

    @pytest.mark.parametrize(
        "source, payload, expected, default_stem",
        [
            (
                "gpm_crap", GPM_RAW, GPM_RAW,
                "contaminants_gpm-crap",
            ),
            (
                "frankenfield2022", FRANKENFIELD_RAW, FRANKENFIELD_FORMATTED,
                "contaminants_frankenfield2022",
            ),
        ],
    )
    def test_default_path_appends_md5_digest(
        self, tmp_path, monkeypatch, source, payload, expected, default_stem,
    ):
        """
        With ``path=None`` the function writes to a default file in the
        current working directory whose stem carries the first 8 hex chars
        of the MD5 of the final candidate bytes (post-formatting). The
        internal ``TemporaryDirectory`` must be torn down even when the
        caller did not supply a path.
        """
        monkeypatch.chdir(tmp_path)
        expected_md5 = hashlib.md5(expected).hexdigest()[:8]

        captured_temp_dirs = []
        orig_td = tempfile.TemporaryDirectory

        def spy_td(*args, **kwargs):
            obj = orig_td(*args, **kwargs)
            captured_temp_dirs.append(Path(obj.name))
            return obj

        monkeypatch.setattr(tempfile, "TemporaryDirectory", spy_td)

        with patch.object(
            contam_mod, "_download", _make_download_mock(payload),
        ):
            result = contaminants(source=source, path=None)

        expected_rel = Path(f"{default_stem}_{expected_md5}.fasta")
        assert result == expected_rel
        assert (tmp_path / expected_rel).exists()
        assert _sha256(tmp_path / expected_rel) == _sha256_bytes(expected)

        # The internal temp directory must be cleaned up on the success
        # path even when ``path`` was not supplied.
        assert len(captured_temp_dirs) == 1
        assert not captured_temp_dirs[0].exists()

    # -- parent directory creation -----------------------------------------

    def test_parent_directory_is_created(self, tmp_path):
        dst = tmp_path / "nested" / "sub" / "x.fasta"
        assert not dst.parent.exists()

        with patch.object(
            contam_mod, "_download", _make_download_mock(GPM_RAW),
        ):
            result = contaminants(source="gpm_crap", path=dst)

        assert dst.parent.is_dir()
        assert result.exists()

    # -- force=False / force=True ------------------------------------------

    def test_existing_file_force_false_raises_file_exists(self, tmp_path):
        """Pre-existing destination + ``force=False`` must raise
        ``FileExistsError`` and leave the existing bytes untouched. The
        downloader must not even be invoked."""
        dst = tmp_path / "exists.fasta"
        sentinel = b"pre-existing bytes that must not be overwritten\n"
        dst.write_bytes(sentinel)

        with patch.object(contam_mod, "_download") as patched:
            patched.side_effect = _make_download_mock(GPM_RAW)
            with pytest.raises(FileExistsError, match="already exists"):
                contaminants(source="gpm_crap", path=dst, force=False)

        assert patched.call_count == 0
        assert dst.read_bytes() == sentinel

    @pytest.mark.parametrize(
        "source,payload,expected",
        [
            ("gpm_crap", GPM_RAW, GPM_RAW),
            ("frankenfield2022", FRANKENFIELD_RAW, FRANKENFIELD_FORMATTED),
        ],
    )
    def test_existing_file_force_true_overwrites(
        self, tmp_path, source, payload, expected,
    ):
        dst = tmp_path / f"{source}.fasta"
        dst.write_bytes(b"pre-existing bytes that must not be overwritten\n")

        with patch.object(
            contam_mod, "_download", _make_download_mock(payload),
        ):
            result = contaminants(source=source, path=dst, force=True)

        assert result == dst
        assert _sha256(result) == _sha256_bytes(expected)

    # -- formatter failure cleans up temp file -----------------------------

    @pytest.mark.parametrize(
        "bad_payload, error_match",
        [
            (
                b">sp|P12345 missing third pipe segment\nMAAAA\n",
                "exactly three",
            ),
            (
                b">sp|P12345|HUMAN_PROT|extra desc\nMAAAA\n",
                "exactly three",
            ),
            (
                b">sp|BADID|HUMAN_PROT desc\nMAAAA\n",
                "not a valid UniProt accession",
            ),
        ],
    )
    def test_formatter_failure_propagates_and_cleans_up_temp(
        self, tmp_path, monkeypatch, bad_payload, error_match,
    ):
        """
        All Frankenfield-header validation errors raised by the internal
        formatter must propagate out of ``contaminants()`` and the
        ``TemporaryDirectory`` used internally must be torn down on
        failure. Verified through the public function only — no direct
        call to the ``_format_frankenfield_header`` / ``_format_fasta``
        helpers.
        """
        def fake_download(url, dest):
            Path(dest).write_bytes(bad_payload)

        captured = []
        orig_td = tempfile.TemporaryDirectory

        def spy_td(*args, **kwargs):
            obj = orig_td(*args, **kwargs)
            captured.append(Path(obj.name))
            return obj

        monkeypatch.setattr(tempfile, "TemporaryDirectory", spy_td)

        dst = tmp_path / "frank.fasta"
        with patch.object(contam_mod, "_download", fake_download):
            with pytest.raises(ValueError, match=error_match):
                contaminants(source="frankenfield2022", path=dst)

        assert len(captured) == 1
        assert not captured[0].exists()

    # -- opt-in real-download tests ----------------------------------------

    def test_real_frankenfield_download(self, tmp_path):
        """
        Real download from the Hao lab GitHub mirror. Hash pinned to current
        upstream contents; re-pin ``EXPECTED_FRANKENFIELD_REMOTE_HASH`` if
        Hao lab updates the FASTA (see ``_SOURCE_MAP['frankenfield2022']``).
        """
        dst = tmp_path / "frankenfield_real.fasta"
        result = contaminants(source="frankenfield2022", path=dst)

        assert result.exists()
        assert result.stat().st_size > 1024
        assert _sha256(result) == EXPECTED_FRANKENFIELD_REMOTE_HASH

    def test_real_gpm_crap_download(self, tmp_path):
        """
        Real download from GPM cRAP via FTP. Hash pinned to current upstream
        contents; re-pin ``EXPECTED_GPM_REMOTE_HASH`` if upstream rotates
        (see ``_SOURCE_MAP['gpm_crap']``).
        """
        dst = tmp_path / "gpm_real.fasta"
        result = contaminants(source="gpm_crap", path=dst)

        assert result.exists()
        assert result.stat().st_size > 1024
        assert _sha256(result) == EXPECTED_GPM_REMOTE_HASH
