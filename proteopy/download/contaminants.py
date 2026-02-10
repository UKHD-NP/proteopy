"""
Utilities for downloading contaminant FASTA files.
"""

from pathlib import Path
from urllib.request import urlretrieve
from datetime import date
from typing import Callable
import re
import shutil
import tempfile
import warnings


_PACKAGE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _is_uniprot_accession(accession: str) -> bool:
    pattern=r"[OPQ][0-9][A-Z0-9]{3}[0-9](-[0-9]{1,2})?|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-[0-9]{1,2})?"
    return bool(re.fullmatch(pattern, accession))

def check_uniprot_accession_nr(accession: str) -> None:
    if not _is_uniprot_accession(accession):
        raise ValueError(
            f"Accession '{accession}' is not a valid UniProt accession.",
        )

def _format_cpp_crap_header(header: str) -> str:
    """
    Format ccp_crap headers to unified layout.

    Input example:
    sp|cRAP001|P00330|ADH1_YEAST Alcohol dehydrogenase 1 ...
    Output example:
    sp|P00330|ADH1_YEAST Alcohol dehydrogenase 1 ...
    """
    parts = header.split(maxsplit=1)
    id_part = parts[0]
    desc = parts[1] if len(parts) > 1 else ""

    segments = id_part.split("|")
    if len(segments) < 4:
        raise ValueError(
            f"Header '{header}' does not have expected ≥4 pipe segments.",
        )

    database = segments[0]
    accession_number = segments[2]
    protein_id = segments[3]
    check_uniprot_accession_nr(accession_number)
    new_id = f"{database}|{accession_number}|{protein_id}"
    return f"{new_id} {desc}".strip()


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


_DB_MAP_MAXQUANT = {
    "SWISS-PROT": "sp",
    "TREMBL": "tr",
    "ENSEMBL": "ensembl",
    "REFSEQ": "refseq",
}


def _format_maxquant_header(header: str) -> str:
    """
    Format MaxQuant contaminant headers to unified layout.

    Example input:
    P00761 SWISS-PROT:P00761|TRYP_PIG Trypsin - Sus scrofa (Pig).
    Q32MB2 TREMBL:Q32MB2;Q86Y46 Tax_Id=9606 Gene_Symbol=KRT73 Keratin-73
    Example output:
    sp|P00761|TRYP_PIG Trypsin - Sus scrofa (Pig).
    tr|Q32MB2|Q32MB2 Tax_Id=9606 Gene_Symbol=KRT73 Keratin-73
    """
    parts = header.split(maxsplit=2)
    if not parts:
        raise ValueError(
            f"Header '{header}' missing required fields for MaxQuant format.",
        )

    accession_number = None
    if ":" in parts[0]:
        db_and_protein = parts[0]
        description = parts[1] if len(parts) > 1 else ""
    else:
        if len(parts) < 2:
            raise ValueError(
                f"Header '{header}' missing database/accession segment.",
            )
        maxquant_id = parts[0]
        db_and_protein = parts[1]
        description = parts[2] if len(parts) > 2 else ""

    db_split = db_and_protein.split(":", 1)
    if len(db_split) != 2:
        raise ValueError(
            f"Header '{header}' must contain database and accession separator ':'.",
        )
    database_raw, accession_and_protein = db_split

    accession_split = accession_and_protein.split("|", 1)
    accession_number_and_aliases = accession_split[0]
    protein_id = accession_split[1] if len(accession_split) == 2 else ""

    accession_number = accession_number_and_aliases.split(";")[0]

    if accession_number is None:
        raise ValueError(
            f"Error with header '{header}'. Could not parse uniprot accession number.",
        )

    # Only UniProt accessions require strict UniProt validation.
    if database_raw in ("SWISS-PROT", "TREMBL"):
        check_uniprot_accession_nr(accession_number)

    database = _DB_MAP_MAXQUANT.get(database_raw)
    if database is None:
        raise ValueError(f"Unrecognized database '{database_raw}' in header '{header}'.")

    new_id = f"{database}|{accession_number}|{protein_id}".rstrip("|")
    return f"{new_id} {description}".strip()


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
    "ccp_crap": {
        "kind": "package",
        "value": _PACKAGE_DATA_DIR / "contaminants_ccp-crap_2025-12-12.fasta",
        "default_path": "data/contaminants_ccp-crap_2025-12-12.fasta",
        "formatter": _format_cpp_crap_header,
    },
    "gpm_crap": {
        "kind": "url",
        "value": "ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta",
        "default_path": "data/contaminants_gpm-crap.fasta",
    },
    "frankenfield2022": {
        "kind": "url",
        "value": (
            "https://raw.githubusercontent.com/HaoGroup-ProtContLib/"
            "Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics/"
            "refs/heads/main/Universal%20protein%20contaminant%20FASTA/"
            "0602_Universal%20Contaminants.fasta"
        ),
        "default_path": "data/contaminants_frankenfield2022.fasta",
        "formatter": _format_frankenfield_header,
    },
    "maxquant": {
        "kind": "package",
        "value": _PACKAGE_DATA_DIR / "contaminants_maxquant_v2.7.5.0.fasta",
        "default_path": "data/contaminants_maxquant_v2.7.5.0.fasta",
        "formatter": _format_maxquant_header,
    },
}


def contaminants(
    source: str = "frankenfield2022",
    path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download contaminant FASTA files from known sources.
    - ``ccp_crap``: Cambridge Centre for Proteomics, cRAP (via camprotR
      download_ccp_crap).
    - ``gpm_crap``: The Global Proteome Machine (GPM) common Repository of
      Adventitious Proteins (cRAP).
    - ``frankenfield2022``: Frankenfield et al., 2022
      (doi:10.1021/acs.jproteome.2c00145).
    - ``maxquant``: MaxQuant bundled contaminant list (v2.7.5.0).

    Parameters
    ----------
    source
        Contaminant FASTA source. Supported: ``"ccp_crap"``,
        ``"frankenfield2022"``, ``"maxquant"``, ``"gpm_crap"``.
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
        if meta["kind"] == "url":
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

    if meta["kind"] == "url":
        formatter = meta.get("formatter")
        if formatter is None:
            urlretrieve(meta["value"], destination)
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                urlretrieve(meta["value"], tmp_path)
                _format_fasta(tmp_path, destination, formatter)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    elif meta["kind"] == "package":
        src = meta["value"]
        if not src.exists():
            raise FileNotFoundError(
                f"Packaged contaminant FASTA not found at {src}.",
            )
        formatter = meta.get("formatter")
        if formatter is None:
            shutil.copyfile(src, destination)
        else:
            _format_fasta(src, destination, formatter)
    else:
        raise ValueError(f"Unsupported source kind '{meta['kind']}'.")

    return destination
