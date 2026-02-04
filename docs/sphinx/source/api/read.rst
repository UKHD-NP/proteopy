``.read``
=========

.. module:: proteopy.read
   :synopsis: Data import from various proteomics pipelines

The ``proteopy.read`` module provides functions to import quantitative proteomics
data from various analysis pipelines into :class:`~anndata.AnnData` objects.

Supported formats:

- **DIA-NN**: Native report format
- **Long format**: Generic tabular format with sample, feature, and intensity columns

.. rubric:: Functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.read.diann
   proteopy.read.long
