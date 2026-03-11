import warnings
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pooch

from proteopy.utils.anndata import check_proteodata

_KNOWN_HASH = (
    "sha256:"
    "58c2ea5cfdda5dc1bc91eec2d9c3fb1f56ccadcccd81ae3980877f6710c5a96d"
)


def williams_2018(
    fill_na: float | int | None = None,
) -> ad.AnnData:
    """Load Williams 2018 mouse multi-tissue proteomics dataset.

    Download, process and format as an
    :class:`~anndata.AnnData` object the peptide-level
    SWATH-MS dataset from Williams et al. (2018) [1]_
    quantifying protein expression across five tissues
    in eight genetically diverse BXD mouse strains. Only
    the whole cell fraction is included; peptide
    intensities from different charge states are summed
    per peptide sequence. By default, missing values
    are represented as ``np.nan``.

    Sample annotation (``.obs``) includes:
        - ``sample_id``: Unique sample identifier
        - ``tissue``: Tissue type (Brain, BAT, Heart, Liver, Quad)
        - ``mouse_id``: BXD mouse strain identifier

    Variable annotation (``.var``) includes:
        - ``peptide_id``: Peptide sequence (matches ``.var_names``)
        - ``protein_id``: UniProt protein identifier
        - ``gene_symbol``: Gene symbol

    Data are sourced from the Elsevier supplementary archive
    (DOI: 10.1074/mcp.RA118.000554).

    Parameters
    ----------
    fill_na : float | int | None, optional
        If not ``None``, replace ``np.nan`` in ``.X``
        with this value.

    Returns
    -------
    ad.AnnData
        AnnData object with peptide-level quantification data.
        ``.X`` contains peptide intensities (samples x peptides).

    Raises
    ------
    urllib.error.URLError
        If download from the Elsevier CDN fails.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.williams_2018()
    >>> adata
    AnnData object with n_obs x n_vars
        obs: 'sample_id', 'tissue', 'mouse_id'
        var: 'peptide_id', 'protein_id', 'gene_symbol'

    References
    ----------
    .. [1] Williams EG, Wu Y, Wolski W, Kim JY, Lan J, Hasan M,
       Halter C, Jha P, Ryu D, Auwerx J, and Aebersold R.
       "Quantifying and Localizing the Mitochondrial Proteome
       Across Five Tissues in A Mouse Population." Molecular &
       Cellular Proteomics, 2018, 17(9):1766-1777.
       DOI: 10.1074/mcp.RA118.000554.
    """
    if fill_na is not None and not isinstance(
        fill_na, (int, float),
    ):
        raise TypeError(
            f"fill_na must be float, int, or None, "
            f"got {type(fill_na).__name__}"
        )

    url = (
        "https://ars.els-cdn.com/content/image/"
        "1-s2.0-S1535947620320569-mmc1.zip"
    )
    zip_path = pooch.retrieve(
        url=url,
        known_hash=_KNOWN_HASH,
        fname="williams_2018_mmc1.zip",
        path=pooch.os_cache("proteopy"),
    )

    cache_dir = Path(zip_path).parent
    xlsx_name = "134784_1_supp_121511_p7byjt.xlsx"
    xlsx_path = cache_dir / xlsx_name

    if not xlsx_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(xlsx_name, path=cache_dir)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported",
            category=UserWarning,
            module="openpyxl",
        )
        df = pd.read_excel(xlsx_path)

    # Select metadata columns
    meta_cols = {
        "Unnamed: 0": "peptide_id",
        "Unnamed: 3": "protein_id",
        "Unnamed: 4": "gene_symbol",
    }

    # Select intensity columns: named cols where row 0 == "Intensity",
    # excluding _mito fractions
    intensity_cols = [
        c for c in df.columns
        if "Unnamed" not in str(c)
        and df[c].iloc[0] == "Intensity"
        and "_mito" not in str(c)
    ]

    df = df[list(meta_cols.keys()) + intensity_cols]

    # Remove _WholeCell suffix from sample column names
    df = df.rename(columns={
        c: c.replace("_WholeCell", "")
        for c in intensity_cols
    })
    df = df.rename(columns=meta_cols)

    # Drop the first row (secondary header)
    df = df.iloc[1:].reset_index(drop=True)

    # Extract peptide sequence (remove prefixes and suffixes)
    df["peptide_id"] = (
        df["peptide_id"].str.split("_").str[1]
    )

    # Verify protein_id and gene_symbol are consistent
    # across charge states of the same peptide
    meta_check = (
        df.groupby("peptide_id")[["protein_id", "gene_symbol"]]
        .nunique()
    )
    inconsistent = meta_check[
        (meta_check["protein_id"] > 1)
        | (meta_check["gene_symbol"] > 1)
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Inconsistent protein_id or gene_symbol "
            "across charge states for peptides:\n"
            f"{inconsistent.index.tolist()}"
        )

    # Sum intensities across charge states of the same peptide
    sample_cols = [
        c for c in df.columns
        if c not in ("peptide_id", "protein_id", "gene_symbol")
    ]
    df[sample_cols] = df[sample_cols].astype(float)
    var = (
        df.groupby("peptide_id")[["protein_id", "gene_symbol"]]
        .first()
    )
    var["peptide_id"] = var.index
    X = (
        df.groupby("peptide_id")[sample_cols]
        .sum()
        .values.T
    )

    # Build obs annotation with tissue and mouse_id
    obs = pd.DataFrame({"sample_id": sample_cols})
    parts = obs["sample_id"].str.split(
        "_", n=1, expand=True,
    )
    parts.columns = ["p1", "p2"]
    tissue_first = parts["p1"].str.fullmatch(
        r"Brain|BAT|Heart|Liver|Quad"
    )
    obs["tissue"] = np.where(
        tissue_first, parts["p1"], parts["p2"],
    )
    obs["mouse_id"] = np.where(
        tissue_first, parts["p2"], parts["p1"],
    )
    obs = obs.set_index("sample_id")
    obs.index.name = None
    obs["sample_id"] = obs.index

    # Construct anndata
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.X[adata.X == 0] = np.nan

    if fill_na is not None:
        adata.X[np.isnan(adata.X)] = fill_na

    check_proteodata(adata)

    return adata
