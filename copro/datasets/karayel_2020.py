"""Karayel 2020 human erythropoiesis proteomics dataset.

This module provides access to the protein-level DIA-MS proteomics dataset
from Karayel et al. (2020) studying dynamic phosphosignaling networks
during human erythropoiesis. The study quantified ~7,400 proteins from
CD34+ hematopoietic stem/progenitor cells (HSPCs) isolated from healthy
donors, across five differentiation stages of erythroid development.

Cells were FACS-sorted using CD235a, CD49d, and Band 3 surface markers.
The data is sourced from the PRIDE archive (PXD017276) and includes
measurements from the following erythroid differentiation stages:

- Progenitor: CFU-E progenitor cells (CD34+ HSPCs, negative fraction)
- ProE&EBaso: Proerythroblasts and early basophilic erythroblasts
- LBaso: Late basophilic erythroblasts
- Poly: Polychromatic erythroblasts
- Ortho: Orthochromatic erythroblasts

Reference
---------
Karayel et al. (2020) Integrative proteomics reveals principles of
dynamic phosphosignaling networks in human erythropoiesis.
Molecular Systems Biology 16: e9813.
DOI: 10.15252/msb.20209813
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

import copro as cp

def _parse_sample_id(col: str) -> str:
    """Parse and clean sample identifiers from raw column names.

    Remove technical prefixes, suffixes, and file extensions from column
    names in the downloaded PRIDE data to extract meaningful sample
    identifiers.

    Parameters
    ----------
    col : str
        Raw column name from the PRIDE CSV file containing sample
        identifier and technical annotations.

    Returns
    -------
    str
        Cleaned sample identifier with technical metadata removed.

    Examples
    --------
    >>> col = "[1] 20181222_QX0_OzKa_SA_CD34pos_DIA_P1.raw.PG.Quantity"
    >>> _parse_sample_id(col)
    'P1'
    """
    col = re.sub(r"^\[\d+\]\s*", "", col)
    col = col.replace(".PG.Quantity", "")
    col = re.sub(r"\.raw$", "", col)
    col = Path(col).stem
    col = col.replace("20181222_QX0_OzKa_SA_CD34pos_", "")
    col = col.replace("DIA_", "")
    col = col.replace("_181226121547", "")
    return col

def karayel_2020():
    """Load Karayel 2020 erythropoiesis proteomics dataset.

    Download and process the protein-level DIA-MS dataset from Karayel
    et al. (2020) studying CD34+ hematopoietic stem cell differentiation
    during erythropoiesis. The dataset contains quantitative proteomics
    measurements across five cell types representing sequential stages
    of erythroid development.

    The function downloads data from the PRIDE archive (PXD017276),
    processes sample identifiers, maps technical names to biological
    cell types, and excludes day 7 samples. Protein quantities marked
    as 'Filtered' in the original data are converted to ``np.nan``.

    Sample annotation (``.obs``) includes:
        - ``sample_id``: Unique sample identifier (cell_type_replicate)
        - ``cell_type``: Differentiation stage (Progenitor, ProE&EBaso,
          LBaso, Poly, Ortho)
        - ``replicate``: Technical replicate identifier

    Variable annotation (``.var``) includes:
        - ``protein_id``: Protein group identifier (matches
          ``.var_names``)
        - ``gene_name``: Associated gene name(s)

    Returns
    -------
    ad.AnnData
        AnnData object containing protein-level quantification data.
        ``.X`` contains protein intensities (samples × proteins) with
        missing values as ``np.nan``. Day 7 samples are excluded from
        the dataset.

    Raises
    ------
    urllib.error.URLError
        If download from PRIDE archive fails.

    Examples
    --------
    >>> import copro as cp
    >>> adata = cp.datasets.karayel_2020()
    >>> adata
    AnnData object with n_obs × n_vars
        obs: 'sample_id', 'cell_type', 'replicate'
        var: 'protein_id', 'gene_name'

    >>> adata.obs['cell_type'].unique()
    ['Progenitor', 'ProE&EBaso', 'LBaso', 'Poly', 'Ortho']

    Notes
    -----
    The dataset represents five stages of erythroid differentiation:

    1. Progenitor: CD34+ hematopoietic stem cells
    2. ProE&EBaso: Proerythroblasts and early basophilic erythroblasts
    3. LBaso: Late basophilic erythroblasts
    4. Poly: Polychromatic erythroblasts
    5. Ortho: Orthochromatic erythroblasts

    Samples collected at day 7 (_D7) are filtered out during processing.

    Reference
    ---------
    Karayel et al. (2020) Integrative proteomics reveals principles of
    dynamic phosphosignaling networks in human erythropoiesis.
    Molecular Systems Biology 16: e9813.
    DOI: 10.15252/msb.20209813
    """
    url = (
        "https://ftp.pride.ebi.ac.uk/pride/data/archive/2020/10/"
        "PXD017276/20190213_CD34_Phospho_study_DIA_proteome_Report.csv"
        )
    file_path = pooch.retrieve(
        url=url,
        known_hash=None,  # TODO
        fname="karayel_2020_proteome_report.csv",
        path=pooch.os_cache("copro"),
        )
    df = pd.read_csv(file_path)

    quant_cols = [c for c in df.columns if c.endswith(".PG.Quantity")]
    # Replace 'Filtered' with np.nan before melting
    df[quant_cols] = df[quant_cols].replace("Filtered", np.nan).astype(float)

    long = (
        df[["PG.ProteinGroups"] + quant_cols]
        .melt(
            id_vars="PG.ProteinGroups",
            var_name="raw_col",
            value_name="intensity",
            )
        )

    long["sample_id"] = long["raw_col"].map(_parse_sample_id)
    long = long.drop(columns=["raw_col"])
    long = long.rename(columns={"PG.ProteinGroups": "protein_id"})
    long['sample_id'] = (
        long['sample_id']
        .str.replace('Negativefrac', 'Progenitor', regex=False)
        .str.replace('P1andP2', 'ProE&EBaso', regex=False)
        .str.replace('P3', 'LBaso', regex=False)
        .str.replace('P4', 'Poly', regex=False)
        .str.replace('P5', 'Ortho', regex=False)
        )

    Karayel_2020_quant = long[~long["sample_id"].str.contains('_D7')]

    Karayel_2020_meta_obs = (
        Karayel_2020_quant[['sample_id']]
        .drop_duplicates()
        .reset_index(drop=True)
        )
    Karayel_2020_meta_obs["cell_type"] = (
        Karayel_2020_meta_obs["sample_id"].str.split("_").str[0]
        )
    Karayel_2020_meta_obs["replicate"] = (
        Karayel_2020_meta_obs["sample_id"].str.split("_").str[-1]
        )

    Karayel_2020_meta_var = (
        df[['PG.ProteinGroups', 'PG.Genes']]
        .drop_duplicates()
        .reset_index(drop=True)
        )
    Karayel_2020_meta_var = Karayel_2020_meta_var.rename(columns={
        'PG.ProteinGroups': 'protein_id',
        'PG.Genes': 'gene_name'
    })

    adata = cp.read.long(
        intensities=Karayel_2020_quant,
        level='protein',
        sample_annotation=Karayel_2020_meta_obs,
        var_annotation=Karayel_2020_meta_var,
    )

    return adata
