MaxQuant -- Bundled Contaminant Database
========================================

.. bibliography::
   :filter: key == "cox-2008"

.. rubric:: Overview

`MaxQuant <https://maxquant.org/>`__ is a widely used quantitative
proteomics software platform :cite:p:`cox-2008`. It ships with a
built-in contaminant FASTA file that is automatically appended to
search databases during analysis. The contaminant list is not
independently published or documented outside of the MaxQuant
distribution.

ProteoPy bundles a copy extracted from **MaxQuant v2.7.5.0**,
containing **242 protein entries**. The original MaxQuant file contains
246 entries; four were removed because they use non-standard headers
that cannot be mapped to UniProt accessions (see
:ref:`maxquant-removed-entries` below).

.. rubric:: Library Composition

The 242 entries can be broadly categorized as:

.. list-table::
   :widths: 50 10
   :header-rows: 1

   * - Category
     - Entries
   * - Bovine serum and tissue proteins (predominantly serum albumin,
       caseins, immunoglobulins, and other plasma proteins)
     - 116
   * - Keratins and keratin-associated proteins (human, mouse)
     - 106
   * - Proteolytic enzymes and laboratory reagents (trypsin,
       chymotrypsinogen, Lys-C, Glu-C, Asp-N, pepsin, nuclease)
     - 11
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
     - 72
   * - *Mus musculus* (mouse keratins)
     - 32
   * - Other organisms (enzymes, fluorescent proteins, viral)
     - 17

.. _maxquant-removed-entries:

.. rubric:: Removed Entries

Four entries from the original MaxQuant contaminant file were removed
by ProteoPy because they use non-standard headers incompatible with
ProteoPy's standardized header formatting:

- ``Streptavidin (S.avidinii)`` -- streptavidin with a free-text
  header lacking any database prefix or accession number
- ``H-INV:HIT000016045`` -- Similar to Keratin, type II cytoskeletal 8
  (fragment, 91 aa)
- ``H-INV:HIT000292931`` -- Similar to Keratin, type II cytoskeletal 8
  (502 aa)
- ``H-INV:HIT000015463`` -- Similar to Keratin 18 (gene symbol PTPN14,
  344 aa)

The three H-INV entries use identifiers from H-InvDB (a discontinued human gene
database) and are partial or predicted keratin sequences already well
represented by their canonical UniProt entries. The streptavidin entry lacks a
parseable database/accession structure.

.. note::

   ProteoPy bundles this database as a package resource. Use
   ``pr.download.contaminants(source="maxquant")`` to obtain it.

.. rubric:: Resources

- **MaxQuant**: `maxquant.org <https://maxquant.org/>`__
