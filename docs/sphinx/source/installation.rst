Installation
============

ProteoPy requires Python 3.10 or later. We recommend installing ProteoPy in a
dedicated virtual environment.

Creating a Virtual Environment
------------------------------

.. tab-set::

   .. tab-item:: venv

      .. code-block:: bash

         python -m venv proteopy-env
         source proteopy-env/bin/activate  # Linux/macOS
         # proteopy-env\Scripts\activate   # Windows

   .. tab-item:: conda

      .. code-block:: bash

         conda create -n proteopy-env "python>=3.10"
         conda activate proteopy-env

   .. tab-item:: uv

      .. code-block:: bash

         uv venv proteopy-env
         source proteopy-env/bin/activate

Installing ProteoPy
-------------------

Basic Installation
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install proteopy

With Notebook Support
^^^^^^^^^^^^^^^^^^^^^

For notebook-centric workflows, the ``[usage]`` extra installs ipykernel,
jupyterlab, and scanpy (for extended analysis functionality such as batch
control, PCA, UMAP and more):

.. code-block:: bash

   pip install proteopy[usage]

Don't forget to make your environment accessible via jupyter kernels.

.. code-block:: bash

    python -m ipykernel install --user --name=proteopy-env

Development Version
^^^^^^^^^^^^^^^^^^^

To install the development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/UKHD-NP/proteopy.git

Dependencies
------------

ProteoPy is built on established open-source scientific libraries:

- `NumPy <https://numpy.org/>`_ - Numerical computing
- `SciPy <https://scipy.org/>`_ - Scientific computing
- `pandas <https://pandas.pydata.org/>`_ - Data manipulation
- `AnnData <https://anndata.readthedocs.io/>`_ - Annotated data structures
- `Matplotlib <https://matplotlib.org/>`_ / `Seaborn <https://seaborn.pydata.org/>`_ - Visualization
- `scikit-learn <https://scikit-learn.org/>`_ - Machine learning utilities

Optional dependencies for extended functionality:

- `scanpy <https://scanpy.readthedocs.io/>`_ - Single-cell analysis (batch correction, PCA, UMAP)
- `MuData <https://mudata.readthedocs.io/>`_ / `MUON <https://muon.readthedocs.io/>`_ - Multi-omics integration
