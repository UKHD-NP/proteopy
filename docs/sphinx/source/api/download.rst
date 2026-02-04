``.download``
=============

.. module:: proteopy.download
   :synopsis: Download external resources

The ``proteopy.download`` module provides functions to download external resources
such as contaminant protein databases.

.. rubric:: Functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.download.contaminants

.. rubric:: Contaminant Databases

The ``contaminants`` function downloads protein contaminant databases commonly
used in proteomics quality control:

- **Frankenfield 2022**: Universal contaminant library (381 proteins) from
  Frankenfield et al. -- :doc:`details <manual/frankenfield2022>`
- **CCP cRAP** *(default)*: Cambridge Centre for Proteomics cRAP
  (123 proteins) -- :doc:`details <manual/ccp-crap>`
- **GPM cRAP**: Global Proteome Machine common Repository of
  Adventitious Proteins (116 proteins) --
  :doc:`details <manual/gpm-crap>`
- **MaxQuant**: Bundled contaminant database from MaxQuant v2.7.5.0
  (242 proteins) -- :doc:`details <manual/maxquant>`

.. toctree::
   :hidden:

   manual/frankenfield2022
   manual/ccp-crap
   manual/gpm-crap
   manual/maxquant
