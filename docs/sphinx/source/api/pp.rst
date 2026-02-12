``.pp``
=======

.. module:: proteopy.pp
   :synopsis: Preprocessing functions for proteomics data

The ``proteopy.pp`` module provides preprocessing functions for quality control,
filtering, normalization, and imputation of proteomics data.

.. rubric:: Filtering

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pp.filter_samples
   proteopy.pp.filter_samples_completeness
   proteopy.pp.filter_var
   proteopy.pp.filter_var_completeness
   proteopy.pp.filter_proteins_by_peptide_count
   proteopy.pp.filter_samples_by_category_count
   proteopy.pp.remove_zero_variance_vars
   proteopy.pp.remove_contaminants

.. rubric:: Normalization

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pp.normalize_median

.. rubric:: Imputation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pp.impute_downshift

.. rubric:: Quantification

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pp.extract_peptide_groups
   proteopy.pp.summarize_overlapping_peptides
   proteopy.pp.quantify_proteins
   proteopy.pp.quantify_proteoforms

.. rubric:: Statistics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pp.calculate_cv
