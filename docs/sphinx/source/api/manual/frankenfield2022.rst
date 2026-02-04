Frankenfield 2022 -- Universal Protein Contaminant Library
==========================================================

.. bibliography::
   :filter: key == "frankenfield-2022"

.. rubric:: Overview

Mass spectrometry-based proteomics is challenged by the presence of
contaminant protein signals originating from reagents, sample handling,
and the laboratory environment. These contaminants are difficult to
avoid and, if unaccounted for, can lead to false protein identifications
and reduced sensitivity.

Frankenfield et al. (2022) systematically characterized common sources
of protein contamination and compiled a **universal contaminant FASTA
library** containing **381 protein entries**. The library was designed to
be applicable across all proteomics workflows, including both
data-dependent acquisition (DDA) and data-independent acquisition (DIA).

The authors demonstrated that including the contaminant library during
database searching reduces false discoveries and increases true protein
identifications without affecting quantification accuracy. Library-based
DIA analysis showed more than 5% additional protein and peptide
identifications when contaminant libraries were included in a HepG2 human cells
dataset.

.. rubric:: Library Composition

The 381 contaminant proteins are classified into the following source
categories:

.. list-table::
   :widths: 45 10
   :header-rows: 1

   * - Source of Contamination
     - Proteins
   * - Human skin and hair (keratins and keratin-associated proteins)
     - 151
   * - Residual cell culture medium containing fetal bovine serum (FBS)
     - 120
   * - FBS / affinity bead background
     - 39
   * - Mouse skin and hair
     - 26
   * - Sheep keratin (wool clothing)
     - 16
   * - Proteolytic enzymes (trypsin, pepsin, Lys-C, and others)
     - 11
   * - Other contaminants
     - 7
   * - Fluorescent proteins (GFP, YFP)
     - 3
   * - Latex gloves (*Hevea brasiliensis*)
     - 2
   * - Affinity purification reagents (FLAG, HA, streptavidin beads)
     - 3
   * - Bacterial (*Escherichia coli*)
     - 1
   * - Lys-C protease enzyme
     - 1
   * - Trypsin protease enzyme
     - 1

.. rubric:: Organism Breakdown

Entries in the FASTA file originate from the following organisms:

.. list-table::
   :widths: 45 10
   :header-rows: 1

   * - Organism
     - Proteins
   * - *Bos taurus* (bovine, predominantly FBS-derived)
     - 159
   * - *Homo sapiens* (human keratins and skin proteins)
     - 151
   * - *Mus musculus* (mouse keratins)
     - 26
   * - *Ovis aries* (sheep wool keratins)
     - 16
   * - *Sus scrofa* (porcine enzymes)
     - 4
   * - Other organisms (15 species, 1--2 entries each)
     - 25

.. rubric:: Methodology

The authors generated contamination-only samples by introducing known
contaminant sources into lysis buffer:

- **Enzyme contamination**: trypsin, Lys-C, and trypsin/Lys-C mixtures
- **Affinity purification contamination**: streptavidin, FLAG, and HA
  beads
- **Serum contamination**: fetal bovine serum
- **Keratin contamination**: intentional handling of samples with
  ungloved hands

These contamination-only samples were analyzed by LC-MS/MS. The
resulting identifications were combined with proteins from existing
contaminant databases (cRAP, MaxQuant) to build the universal library,
adding 166 previously uncharacterized contaminant entries.

The library was validated using HEK293 cell lysates and mouse brain
tissue across multiple software platforms (MaxQuant, Proteome Discoverer,
Spectronaut, DIA-NN).

.. rubric:: Sample-Type Specific Libraries

In addition to the universal library, the authors provide sample-type
specific contaminant FASTA files for:

- Cell culture
- Mouse tissue
- Rat tissue
- Neuron culture
- Stem cell culture

These are available from the `GitHub repository
<https://github.com/HaoGroup-ProtContLib/Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics>`__.

.. note::

   ProteoPy downloads the **universal** contaminant library via
   ``pr.download.contaminants(source="frankenfield2022")``.

.. rubric:: Resources

- **Paper**: `doi:10.1021/acs.jproteome.2c00145
  <https://doi.org/10.1021/acs.jproteome.2c00145>`__
- **GitHub**: `HaoGroup-ProtContLib
  <https://github.com/HaoGroup-ProtContLib/Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics>`__
- **FASTA download**: `Universal Contaminants (.fasta)
  <https://raw.githubusercontent.com/HaoGroup-ProtContLib/Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics/refs/heads/main/Universal%20protein%20contaminant%20FASTA/0602_Universal%20Contaminants.fasta>`__
- **Contaminant protein descriptions**: `Supplemental Table S1 (.xlsx)
  <https://github.com/HaoGroup-ProtContLib/Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics/blob/main/Universal%20protein%20contaminant%20FASTA/Contaminant%20protein%20information%20in%20the%20FASTA%20library%20and%20the%20potential%20source%20of%20contaminations.xlsx>`__
  (UniProt IDs, protein names, organisms, and contamination sources
  for all 381 entries)
- **ProteomeXchange**: `PXD031139
  <http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD031139>`__
  (raw LC-MS data for contaminant-only samples)
