CCP cRAP -- Cambridge Centre for Proteomics cRAP
=================================================

.. rubric:: Overview

The **CCP cRAP** database is a contaminant protein collection maintained
by the `Cambridge Centre for Proteomics
<https://cambridgecentreforproteomics.github.io/camprotR/>`__ (CCP) as
part of their ``camprotR`` R package. It contains **123 protein
entries**. The original CCP cRAP has 125 entries; two were removed by
ProteoPy (see :ref:`ccp-crap-removed-entries` below).

The CCP cRAP is largely based on the :doc:`GPM cRAP <gpm-crap>`
database, with several additions:

.. _ccp-crap-removed-entries:

.. rubric:: Removed Entries

Two entries from the original CCP cRAP were removed by ProteoPy
because they use placeholder accession numbers (``000000``) that are
not valid UniProt accessions:

- ``cRAP126|000000|ENDOP_GLUC`` -- Endoproteinase Glu-C (NEB, P8100S)
- ``cRAP127|000000|RECOM_LYSC`` -- recombinant Lys-C (Promega, V167A)

Both are commercial protease products added by CCP as extensions to the
GPM cRAP base.

.. note::

   ProteoPy bundles a copy downloaded on 2025-12-12. Use
   ``pr.download.contaminants(source="ccp_crap")`` to obtain it.
   This is the **default** source for ``pr.download.contaminants()``.

.. rubric:: License

The camprotR R package (and its bundled cRAP FASTA) is released under the
`MIT License <https://opensource.org/license/mit>`__ by the Cambridge Centre
for Proteomics. ProteoPy includes the full MIT license text and copyright
notice in the ``THIRD_PARTY_LICENSES`` file at the repository root.

.. rubric:: Resources

- **camprotR package**: `GitHub
  <https://github.com/CambridgeCentreForProteomics/camprotR>`__
- **cRAP vignette**: `crap.html
  <https://cambridgecentreforproteomics.github.io/camprotR/articles/crap.html>`__
