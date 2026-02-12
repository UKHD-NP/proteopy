``.tl``
=======

.. module:: proteopy.tl
   :synopsis: Analysis tools and algorithms

The ``proteopy.tl`` module provides analysis tools including differential abundance
testing, proteoform inference via the COPF algorithm, and clustering methods.

.. rubric:: Differential Analysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.tl.differential_abundance

.. rubric:: Clustering

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.tl.hclustv_tree
   proteopy.tl.hclustv_cluster_ann
   proteopy.tl.hclustv_profiles

.. rubric:: Proteoform Inference (COPF)

The COPF (COrrelation based functional ProteoForm assessment) algorithm enables
detection of functional proteoform groups from peptide-level quantitative data
:cite:p:`bludau-2021`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.tl.pairwise_peptide_correlations
   proteopy.tl.peptide_dendograms_by_correlation
   proteopy.tl.peptide_clusters_from_dendograms
   proteopy.tl.proteoform_scores

