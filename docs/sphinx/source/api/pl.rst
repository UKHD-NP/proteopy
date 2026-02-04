``.pl``
=======

.. module:: proteopy.pl
   :synopsis: Visualization functions

The ``proteopy.pl`` module provides publication-ready visualizations for quality
control, exploratory analysis, and statistical results.

.. rubric:: Intensity Distributions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.peptide_intensities
   proteopy.pl.proteoform_intensities
   proteopy.pl.intensity_hist
   proteopy.pl.intensity_box_per_sample
   proteopy.pl.box

.. rubric:: Quality Control

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.abundance_rank
   proteopy.pl.completeness
   proteopy.pl.completeness_per_sample
   proteopy.pl.completeness_per_var

.. rubric:: Metadata exploration

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.n_samples_per_category
   proteopy.pl.n_peptides_per_sample
   proteopy.pl.n_proteins_per_sample
   proteopy.pl.n_cat1_per_cat2_hist
   proteopy.pl.n_peptides_per_protein
   proteopy.pl.n_proteoforms_per_protein
   proteopy.pl.cv_by_group
   proteopy.pl.sample_correlation_matrix

.. rubric:: Proteoform Inference

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.proteoform_scores

.. rubric:: Differential Analysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.volcano_plot
   proteopy.pl.differential_abundance_box

.. rubric:: Clustering

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.pl.hclustv_profiles_heatmap
   proteopy.pl.hclustv_silhouette
   proteopy.pl.hclustv_elbow
   proteopy.pl.hclustv_profile_intensities
