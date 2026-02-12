``.download``
=============

.. module:: proteopy.download
   :synopsis: Download external resources

The ``proteopy.download`` module provides functions to download external
resources such as contaminant protein databases.

.. rubric:: Functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.download.contaminants

.. rubric:: Contaminant Databases (automatic download via API)

The ``contaminants`` function downloads protein contaminant databases commonly
used in proteomics quality control:

- **Frankenfield 2022** *(default)*: Universal contaminant library (381
  proteins) from Frankenfield et al. --
  :doc:`details <manual/frankenfield2022>`
- **GPM cRAP**: Global Proteome Machine common Repository of
  Adventitious Proteins (116 proteins) --
  :doc:`details <manual/gpm-crap>`

.. rubric:: Contaminant Databases (manual download guide, not in API)

- **CCP cRAP**: Cambridge Centre for Proteomics cRAP (125 proteins,
  requires ``camprotR`` R package) --
  :doc:`details <manual/ccp-crap>`
- **MaxQuant**: Contaminant database from MaxQuant
  (246 proteins) -- :doc:`details <manual/maxquant>`

.. toctree::
   :hidden:

   manual/frankenfield2022
   manual/ccp-crap
   manual/gpm-crap
   manual/maxquant
