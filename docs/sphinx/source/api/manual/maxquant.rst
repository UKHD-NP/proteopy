MaxQuant -- Contaminant Database
================================

.. bibliography::
   :filter: key == "cox-2008"

.. rubric:: Overview

`MaxQuant <https://maxquant.org/>`__ is a widely used quantitative
proteomics software platform :cite:p:`cox-2008`. It ships with a
built-in contaminant FASTA file that is automatically appended to
search databases during analysis. The contaminant list is not
independently published or documented outside of the MaxQuant
distribution.

The MaxQuant contaminant file contains **246 protein entries**
(MaxQuant_v2.7.5.0) and uses a non-standard FASTA header format (see
:ref:`maxquant-header-format` below).

.. rubric:: Library Composition

The entries can be broadly categorized as:

.. list-table::
   :widths: 50 10
   :header-rows: 1

   * - Category
     - Entries
   * - Bovine serum and tissue proteins (predominantly serum albumin,
       caseins, immunoglobulins, and other plasma proteins)
     - 116
   * - Keratins and keratin-associated proteins (human, mouse)
     - 109
   * - Proteolytic enzymes and laboratory reagents (trypsin,
       chymotrypsinogen, Lys-C, Glu-C, Asp-N, pepsin, nuclease,
       streptavidin)
     - 12
   * - Other (dermokine, hornerin, filaggrin,
       and unannotated Ensembl/RefSeq entries)
     - 7
   * - Fluorescent proteins (GFP, YFP)
     - 2

.. rubric:: Organism Breakdown

.. list-table::
   :widths: 50 10
   :header-rows: 1

   * - Organism
     - Proteins
   * - *Bos taurus* (serum, tissue, and reagent proteins)
     - 121
   * - *Homo sapiens* (keratins, dermokine)
     - 75
   * - *Mus musculus* (mouse keratins)
     - 32
   * - Other organisms (enzymes, fluorescent proteins, viral)
     - 18

.. rubric:: Obtaining the File

ProteoPy does **not** bundle the MaxQuant contaminant file. Users
must obtain it themselves from the MaxQuant distribution:

1. Read and accept the MaxQuant
   `terms and conditions <https://maxquant.org/download_asset/maxquant/latest>`__
   before downloading.
2. Download the latest MaxQuant release from
   https://maxquant.org/download_asset/maxquant/latest
3. Unzip the downloaded archive.
4. Copy the contaminant FASTA file from the extracted directory:

   .. code-block:: text

      MaxQuant_vX.X.X.X/bin/conf/contaminants.fasta

   to your project directory.

.. _maxquant-header-format:

.. rubric:: Header Format and usage with ProteoPy

MaxQuant uses a non-standard FASTA header format where the UniProt
accession is the first whitespace-delimited token:

.. code-block:: text

   >P00761 SWISS-PROT:P00761|TRYP_PIG Trypsin - Sus scrofa (Pig).
   >Q32MB2 TREMBL:Q32MB2;Q86Y46 Tax_Id=9606 Gene_Symbol=KRT73 ...

Because this differs from the standard UniProt header format,
you must pass a custom ``header_parser`` to
:func:`pr.pp.remove_contaminants()
<proteopy.pp.filtering.remove_contaminants>`.

.. code-block:: python

   import proteopy as pr

   def maxquant_header_parser(header: str) -> str:
       """Extract the UniProt accession from a MaxQuant FASTA header.
       """
       return header.split()[0]

   pr.pp.remove_contaminants(
       adata,
       contaminant_path="contaminants.fasta",
       header_parser=maxquant_header_parser,
       )

.. rubric:: Entries to Consider Removing

Four entries in the MaxQuant contaminant file use non-standard headers
that cannot be mapped to UniProt accessions. These will cause the
``header_parser`` above to return identifiers that do not match any
UniProt protein. Depending on your workflow, you may want to manually
remove them from the FASTA before use:

- ``Streptavidin (S.avidinii)``
- ``H-INV:HIT000016045`` -- Similar to Keratin, type II cytoskeletal 8
- ``H-INV:HIT000292931`` -- Similar to Keratin, type II cytoskeletal 8
- ``H-INV:HIT000015463`` -- Similar to Keratin 18 (gene symbol PTPN14)

.. rubric:: Resources

- **MaxQuant**: `maxquant.org <https://maxquant.org/>`__
