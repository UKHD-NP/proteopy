CCP cRAP -- Cambridge Centre for Proteomics cRAP
=================================================

.. rubric:: Overview

The **CCP cRAP** database is a contaminant protein collection maintained
by the `Cambridge Centre for Proteomics
<https://cambridgecentreforproteomics.github.io/camprotR/>`__ (CCP) as
part of their ``camprotR`` R package. It contains **125 protein
entries** and is largely based on the :doc:`GPM cRAP <gpm-crap>`
database, with several additions.

.. rubric:: Obtaining the File

ProteoPy does **not** bundle the CCP cRAP file. Users must obtain it
themselves using the ``camprotR`` R package. See the
`cRAP vignette
<https://cambridgecentreforproteomics.github.io/camprotR/articles/crap.html>`__
for full details.

.. code-block:: r

   library(camprotR)

   ccp_crap <- download_ccp_crap(tempfile(fileext = ".fasta"), is_crap = TRUE)

   library(Biostrings)
   writeXStringSet(ccp_crap, filepath = "/path/to/ccp_crap.fasta")

.. _ccp-crap-entries-to-consider-removing:

.. rubric:: Entries to Consider Removing

Two entries in the CCP cRAP use placeholder accession numbers
(``000000``) that are not valid UniProt accessions. These will not
match any protein in a UniProt-based search database. Depending on
your workflow, you may want to manually remove them from the FASTA
before use:

- ``cRAP126|000000|ENDOP_GLUC`` -- Endoproteinase Glu-C (NEB, P8100S)
- ``cRAP127|000000|RECOM_LYSC`` -- recombinant Lys-C (Promega, V167A)

Both are commercial protease products added by CCP as extensions to the
GPM cRAP base.

.. rubric:: License

The camprotR R package (and its bundled cRAP FASTA) is released under the
`MIT License <https://opensource.org/license/mit>`__ by the Cambridge Centre
for Proteomics.

.. rubric:: Resources

- **camprotR package**: `GitHub
  <https://github.com/CambridgeCentreForProteomics/camprotR>`__
- **cRAP vignette**: `crap.html
  <https://cambridgecentreforproteomics.github.io/camprotR/articles/crap.html>`__
