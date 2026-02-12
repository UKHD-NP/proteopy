``proteopy``
============

This section provides complete API documentation for ProteoPy. The library is
organized into the following modules:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`read`
     - **Data import** from DIA-NN, MaxQuant, and tabular formats
   * - :doc:`pp`
     - **Preprocessing**: filtering, normalization, imputation
   * - :doc:`tl`
     - **Tools**: differential analysis, proteoform inference, clustering
   * - :doc:`pl`
     - **Plotting**: QC visualizations, volcano plots, heatmaps
   * - :doc:`get`
     - **Data retrieval**: extract results as DataFrames
   * - :doc:`ann`
     - **Annotation**: add metadata to samples and variables
   * - :doc:`datasets`
     - Built-in example datasets
   * - :doc:`download`
     - Download external resources (contaminant databases)
   * - :doc:`utils`
     - **Utility** functions for data validation

.. attention::

   The ``utils`` module is under active development. Its API may change
   without notice and we do not guarantee backwards compatibility.

.. toctree::
   :maxdepth: 2
   :hidden:

   read
   ann
   pp
   tl
   pl
   get
   datasets
   download
   utils
