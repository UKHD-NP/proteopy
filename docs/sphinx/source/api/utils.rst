``.utils``
==========

.. module:: proteopy.utils
   :synopsis: Utility functions

.. note::

   Only the functions listed below are part of the public API and
   officially supported. All other functions in ``proteopy.utils``
   are internal helpers and may change without notice.

.. rubric:: Data Validation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.utils.is_proteodata
   proteopy.utils.check_proteodata

.. rubric:: Array Utilities

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.utils.is_log_transformed

.. rubric:: Proteoform Group Simulation

Simulate proteoform groups to evaluate how well a proteoform inference
method recovers them in a dataset, optionally with simulated batch
effects to evaluate batch-aware inference.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.utils.proteoform_group_simulator
   proteopy.utils.proteoform_simulation_metrics
   proteopy.utils.apply_peptide_batch_effects

.. rubric:: Plotting

.. autosummary::
   :toctree: generated/
   :nosignatures:

   proteopy.utils.volcano_plot
