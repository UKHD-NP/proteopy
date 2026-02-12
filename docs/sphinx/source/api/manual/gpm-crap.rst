GPM cRAP -- common Repository of Adventitious Proteins
=======================================================

.. bibliography::
   :filter: key == "gpm-crap"

.. rubric:: Overview

The **cRAP** (common Repository of Adventitious Proteins, pronounced
"cee-RAP") is a curated database of protein sequences commonly found in
proteomics experiments through either accidental contamination or
deliberate use as laboratory reagents. The database contains **116 protein
entries** and is maintained by `The Global Proteome Machine Organization
<https://www.thegpm.org/>`__.

.. rubric:: Library Composition

The website documents 115 entries organized into five categories. The
downloadable FASTA contains one additional unlisted entry (see note
below).

.. list-table::
   :widths: 50 10
   :header-rows: 1

   * - Category
     - Entries
   * - Sigma-Aldrich Universal Protein Standard (UPS) reference proteins
     - 48
   * - Dust and contact proteins (human skin, hair, saliva, sheep wool,
       latex gloves)
     - 38
   * - Laboratory reagent proteins (BSA, trypsin, chymotrypsinogen,
       pepsin, Lys-C, and others)
     - 19
   * - Molecular weight markers and standard proteins (horse cytochrome
       C, *E. coli* beta-galactosidase, rabbit aldolase, and others)
     - 9
   * - Common viral contaminants (*S. cerevisiae* virus L-A coat
       protein)
     - 1
   * - Unlisted in website (KKA1_ECOLX, *E. coli* kanamycin
       nucleotidyltransferase -- antibiotic resistance marker)
     - 1

.. note::

   The 48 UPS entries are human proteins from the Sigma-Aldrich
   Universal Protein Standard, a commercial mixture used for absolute
   quantification. These are not contaminants per se but are included
   because they frequently appear in proteomics experiments as
   spike-in standards.

.. rubric:: Organism Breakdown

Entries in the FASTA file originate from the following organisms:

.. list-table::
   :widths: 50 10
   :header-rows: 1

   * - Organism
     - Proteins
   * - *Homo sapiens* (keratins, UPS standards, saliva proteins)
     - 68
   * - *Ovis aries* (sheep wool keratins)
     - 16
   * - *Bos taurus* (BSA, caseins, trypsin, and other reagents)
     - 13
   * - *Sus scrofa* (porcine pepsin, trypsin)
     - 4
   * - *Equus caballus* (cytochrome C, myoglobin)
     - 2
   * - *Hevea brasiliensis* (latex glove proteins)
     - 2
   * - *Gallus gallus* (lysozyme, ovalbumin)
     - 2
   * - Other organisms (9 species, 1--2 entries each)
     - 9

.. rubric:: Category Details

**Laboratory proteins** include enzymes and reagents routinely used in
sample preparation:

- Bovine serum albumin (BSA)
- Bovine trypsin variants (TRY1, TRY2)
- Porcine trypsin and pepsin isoforms (A, B, C)
- Bovine chymotrypsinogen variants (A, B)
- Lysyl endopeptidase (Lys-C)
- *Staphylococcus aureus* V8 protease (Glu-C)

**Dust and contact proteins** represent environmental contaminants from
human skin and hair (keratins, keratin-associated proteins), sheep wool
(from clothing), and latex gloves (*Hevea brasiliensis* rubber elongation
factor and small rubber particle protein).

**Molecular weight markers** are commonly used calibration standards
such as horse cytochrome C, horse myoglobin, chicken ovalbumin and
lysozyme, rabbit aldolase, yeast alcohol dehydrogenase, *E. coli*
beta-galactosidase, and bovine glutamate dehydrogenase.

.. note::

   ProteoPy downloads the cRAP database via
   ``pr.download.contaminants(source="gpm_crap")``.

.. rubric:: Resources

- **Website**: `thegpm.org/crap <https://www.thegpm.org/crap/>`__
- **FASTA download**: `ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta
  <ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta>`__
