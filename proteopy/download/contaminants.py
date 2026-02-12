"""
Utilities for downloading contaminant FASTA files.
"""

from pathlib import Path
from urllib.request import urlretrieve
from datetime import date
from typing import Callable
import re
import tempfile
import warnings



def _is_uniprot_accession(accession: str) -> bool:
    pattern=r"[OPQ][0-9][A-Z0-9]{3}[0-9](-[0-9]{1,2})?|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-[0-9]{1,2})?"
    return bool(re.fullmatch(pattern, accession))

def check_uniprot_accession_nr(accession: str) -> None:
    if not _is_uniprot_accession(accession):
        raise ValueError(
            f"Accession '{accession}' is not a valid UniProt accession.",
        )

def _format_frankenfield_header(header: str) -> str:
    """
    Validate Frankenfield2022 headers; enforce three pipe-separated fields and
    UniProt-style accession.
    """
    parts = header.split(maxsplit=1)
    id_part = parts[0]
    desc = parts[1] if len(parts) > 1 else ""

    segments = id_part.split("|")
    if len(segments) != 3:
        raise ValueError(
            f"Header '{header}' must have exactly three pipe-separated fields.",
        )

    database, accession_number, protein_id = segments
    if accession_number.startswith("Cont_"):
        accession_number = accession_number[len("Cont_"):]
    _FRANKENFIELD_MANUAL_IDS = {"AAAA1", "AAAA2"}
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
    with open(source_path, "r", encoding="utf-8") as src, open(
        destination_path,
        "w",
        encoding="utf-8",
    ) as dest:
        for line in src:
            if line.startswith(">"):
                header = line[1:].strip()
                formatted = formatter(header)
                dest.write(f">{formatted}\n")
            else:
                dest.write(line)



_SOURCE_MAP = {
    "gpm_crap": {
        "url": "ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta",
        "default_path": "data/contaminants_gpm-crap.fasta",
    },
    "frankenfield2022": {
        "url": (
            "https://raw.githubusercontent.com/HaoGroup-ProtContLib/"
            "Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics/"
            "refs/heads/main/Universal%20protein%20contaminant%20FASTA/"
            "0602_Universal%20Contaminants.fasta"
        ),
        "default_path": "data/contaminants_frankenfield2022.fasta",
        "formatter": _format_frankenfield_header,
    },
}


def contaminants(
    source: str = "frankenfield2022",
    path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download contaminant FASTA files from putative sources.

    - ``frankenfield2022``: Frankenfield et al., 2022
      (doi:10.1021/acs.jproteome.2c00145).
    - ``gpm_crap``: The Global Proteome Machine (GPM) common Repository of
      Adventitious Proteins (cRAP).

    Parameters
    ----------
    source
        Contaminant FASTA source. Supported: ``"frankenfield2022"``,
        ``"gpm_crap"``.
    path
        Destination file path for the downloaded FASTA. If ``None``, a default
        path is chosen based on the ``source``; URL downloads append the current
        date (YYYY-MM-DD) to the filename.
    force
        If ``True``, overwrite an existing file at ``path``.

    Returns
    -------
    Path
        Path to the downloaded FASTA file.
    """
    if source is None:
        raise ValueError("Missing 'source' parameter.")

    if source not in _SOURCE_MAP:
        raise ValueError(f"Unsupported source '{source}'.")

    meta = _SOURCE_MAP[source]
    if path is None:
        destination = Path(meta["default_path"])
        today_suffix = date.today().strftime("%Y-%m-%d")
        destination = destination.with_name(
            f"{destination.stem}_{today_suffix}{destination.suffix}",
        )
    else:
        destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        warnings.warn(
            f"File already exists at {destination}. Use force=True to overwrite.",
        )
        return destination

    formatter = meta.get("formatter")
    if formatter is None:
        urlretrieve(meta["url"], destination)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urlretrieve(meta["url"], tmp_path)
            _format_fasta(tmp_path, destination, formatter)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return destination
