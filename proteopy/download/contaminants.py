"""
Utilities for downloading contaminant FASTA files.
"""

from pathlib import Path
from urllib.request import urlopen
from collections.abc import Callable
import hashlib
import os
import re
import shutil
import tempfile


_DOWNLOAD_TIMEOUT_SECONDS = 60


def _md5_id(path: Path, length: int = 8) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:length]


def _is_uniprot_accession(accession: str) -> bool:
    pattern = (
        r"[OPQ][0-9][A-Z0-9]{3}[0-9](-[0-9]{1,2})?"
        r"|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-[0-9]{1,2})?"
    )
    return bool(re.fullmatch(pattern, accession))


def check_uniprot_accession_nr(accession: str) -> None:
    if not _is_uniprot_accession(accession):
        raise ValueError(
            f"Accession '{accession}' is not a valid UniProt accession.",
        )


def _format_frankenfield_header(header: str) -> str:
    """
    Validate Frankenfield2022 headers; enforce three pipe-separated
    fields and UniProt-style accession.
    """
    parts = header.split(maxsplit=1)
    id_part = parts[0]
    desc = parts[1] if len(parts) > 1 else ""

    segments = id_part.split("|")
    if len(segments) != 3:
        raise ValueError(
            f"Header '{header}' must have exactly three "
            "pipe-separated fields.",
        )

    database, accession_number, protein_id = segments
    if accession_number.startswith("Cont_"):
        accession_number = accession_number[len("Cont_") :]
    if accession_number not in _FRANKENFIELD_MANUAL_IDS:
        check_uniprot_accession_nr(accession_number)

    new_id = f"{database}|{accession_number}|{protein_id}"
    return f"{new_id} {desc}".strip()


def _format_fasta(
    source_path: Path,
    destination_path: Path,
    formatter: Callable[[str], str],
) -> None:
    """
    Rewrite FASTA headers using a formatter callable.
    """
    with (
        open(source_path, encoding="utf-8") as src,
        open(
            destination_path,
            "w",
            encoding="utf-8",
        ) as dest,
    ):
        for line in src:
            if line.startswith(">"):
                header = line[1:].strip()
                formatted = formatter(header)
                dest.write(f">{formatted}\n")
            else:
                dest.write(line)


def _download(url: str, destination: Path) -> None:
    """
    Stream ``url`` to ``destination`` with a bounded timeout.
    """
    with (
        urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
        open(
            destination,
            "wb",
        ) as out,
    ):
        shutil.copyfileobj(response, out)


def _validate_fasta(path: Path) -> None:
    """
    Verify ``path`` is non-empty and starts (after blank lines) with a
    FASTA header line beginning with ``>``.
    """
    with open(path, "rb") as src:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith(b">"):
                raise ValueError(
                    f"Downloaded file at {path} does not look like a "
                    "FASTA: first non-blank line does not start with '>'.",
                )
            return
    raise ValueError(f"Downloaded file at {path} is empty.")


def _resolve_destination(
    base_destination: Path,
    candidate_path: Path,
    use_digest: bool,
) -> Path:
    """
    Resolve final destination, optionally appending an MD5 digest.
    """
    if not use_digest:
        return base_destination
    digest = _md5_id(candidate_path)
    return base_destination.with_name(
        f"{base_destination.stem}_{digest}{base_destination.suffix}",
    )


def _atomic_move(candidate_path: Path, destination: Path) -> None:
    """
    Move ``candidate_path`` to ``destination`` via same-fs staging.
    """
    staging = destination.parent / f".{destination.name}.tmp"
    try:
        shutil.copy2(candidate_path, staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def _check_no_existing(path: Path, force: bool) -> None:
    """
    Raise ``FileExistsError`` if ``path`` exists and ``force`` is False.
    """
    if path.exists() and not force:
        raise FileExistsError(
            f"File already exists at {path}. Use force=True to overwrite.",
        )


def _fetch_candidate(
    url: str,
    tmp_dir: str,
    formatter: Callable[[str], str] | None,
    verbose: bool,
) -> Path:
    """
    Download FASTA into ``tmp_dir`` and apply ``formatter`` if given.
    """
    raw_path = Path(tmp_dir) / "raw"
    _download(url, raw_path)
    _validate_fasta(raw_path)
    if formatter is None:
        return raw_path
    formatted_path = Path(tmp_dir) / "formatted"
    _format_fasta(raw_path, formatted_path, formatter)
    if verbose:
        print("Formatting contaminants.")
    return formatted_path


_FRANKENFIELD_MANUAL_IDS = {"AAAA1", "AAAA2"}

_SOURCE_MAP = {
    "gpm_crap": {
        "url": "ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta",
        "default_path": "contaminants_gpm-crap.fasta",
    },
    "frankenfield2022": {
        "url": (
            "https://raw.githubusercontent.com/HaoGroup-ProtContLib/"
            "Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics/"
            "refs/heads/main/Universal%20protein%20contaminant%20FASTA/"
            "0602_Universal%20Contaminants.fasta"
        ),
        "default_path": "contaminants_frankenfield2022.fasta",
        "formatter": _format_frankenfield_header,
    },
}


def contaminants(
    source: str = "frankenfield2022",
    path: str | Path | None = None,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """Download a contaminant FASTA file from a supported source.

    Fetches a protein contaminant database in FASTA format and
    writes it to disk. Two sources are supported:

    - ``"frankenfield2022"``: Universal contaminant library for DDA
      and DIA proteomics from Frankenfield et al. [1]_ Headers are
      reformatted to standard ``db|accession|id`` notation and
      ``Cont_`` prefixes are stripped from accession numbers. See
      `database description
      <https://proteopy.readthedocs.io/en/latest/api/manual/
      frankenfield2022.html>`_.
    - ``"gpm_crap"``: The GPM common Repository of Adventitious
      Proteins (cRAP) [2]_, a community-curated list of common
      laboratory contaminants. See `database description
      <https://proteopy.readthedocs.io/en/latest/api/manual/
      gpm-crap.html>`_.

    Parameters
    ----------
    source : str, optional
        Identifier of the contaminant database to download.
        Supported values: ``"frankenfield2022"``, ``"gpm_crap"``.
    path : str | Path | None, optional
        Destination path for the downloaded FASTA file. When
        ``None``, a default file name is written in the current
        working directory with an MD5 digest appended to the stem
        for reproducible identification.
    force : bool, optional
        If ``True``, overwrite an existing file at the resolved
        destination path.
    verbose : bool, optional
        Print download URL, formatting status, and final save
        path to stdout.

    Returns
    -------
    Path
        Absolute path to the written FASTA file.

    Raises
    ------
    ValueError
        If ``source`` is not one of the supported source keys,
        or if a FASTA header from the ``"frankenfield2022"``
        source does not contain exactly three pipe-separated
        fields or carries an invalid UniProt accession number.
    FileExistsError
        If a file already exists at the resolved destination
        and ``force`` is ``False``.

    Examples
    --------
    Download the Frankenfield 2022 library to the default path:

    >>> import proteopy as pr
    >>> path = pr.download.contaminants()

    Download the GPM cRAP database to the default path:

    >>> path = pr.download.contaminants(source="gpm_crap")

    Save to a specific location:

    >>> path = pr.download.contaminants(
    ...     source="frankenfield2022",
    ...     path="my_project/contaminants.fasta",
    ... )

    References
    ----------
    .. [1] Frankenfield AM, Ni J, Ahmed M, and Hao L.
       "Protein Contaminants Matter: Building Universal Protein
       Contaminant Libraries for DDA and DIA Proteomics."
       *Journal of Proteome Research*, 21(9):2104-2113, 2022.
       DOI: 10.1021/acs.jproteome.2c00145.
    .. [2] The Global Proteome Machine Organization. "Common
       Repository of Adventitious Proteins (cRAP)."
       https://www.thegpm.org/crap/
    """
    if source not in _SOURCE_MAP:
        raise ValueError(f"Unsupported source '{source}'.")

    # -- Resolve base destination
    meta = _SOURCE_MAP[source]
    base_destination = (
        Path(meta["default_path"]) if path is None else Path(path)
    )
    base_destination.parent.mkdir(parents=True, exist_ok=True)

    # -- Short-circuit when explicit path already exists
    if path is not None:
        _check_no_existing(base_destination, force)

    # -- Download (and optionally reformat) into a temp directory
    if verbose:
        print(f"Downloading from {meta['url']}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        candidate_path = _fetch_candidate(
            meta["url"],
            tmp_dir,
            meta.get("formatter"),
            verbose,
        )

        destination = _resolve_destination(
            base_destination,
            candidate_path,
            use_digest=path is None,
        )
        if verbose:
            print(f"Destination: {destination}")

        _check_no_existing(destination, force)
        _atomic_move(candidate_path, destination)
        if verbose:
            print(f"Saved to {destination}")

    return destination
