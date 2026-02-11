ProteoPy
========

**An AnnData-based framework for integrated proteomics analysis**

.. image:: https://img.shields.io/pypi/v/proteopy
   :target: https://pypi.org/project/proteopy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/l/proteopy
   :target: https://github.com/UKHD-NP/proteopy/blob/main/LICENSE
   :alt: License

.. image:: https://github.com/UKHD-NP/proteopy/actions/workflows/format-code_perform-tests_on_push-pr.yaml/badge.svg
   :target: https://github.com/UKHD-NP/proteopy/actions/workflows/format-code_perform-tests_on_push-pr.yaml
   :alt: Tests

ProteoPy is a Python library that brings quantitative proteomics analysis into
the :doc:`AnnData <anndata:index>` ecosystem. It provides a unified framework
for protein- and peptide-level analysis — from data import
through quality control, preprocessing, and differential abundance testing —
while storing all data and metadata in a single portable object.

ProteoPy provides simplified yet extensible functions for common
proteomics workflows, seamlessly integrating with the
:doc:`scanpy <scanpy:index>`, `MUON <https://muon.readthedocs.io/>`_,
and the broader single-cell Python ecosystems for reproducible and
scalable multi-omics analysis.

Key features
------------

- **Flexible data import** from DIA-NN, MaxQuant, and generic tabular formats
- **Quality control & filtering** with completeness metrics, CV analysis, and
  contaminant removal
- **Preprocessing** including normalization, batch
  correction (via scanpy), and missing-value imputation
- **Peptide-level analysis** with overlapping peptide grouping, peptide-to-
  protein quantification, and per-protein peptide intensity visualization
- **Differential abundance analysis** with t-test, Welch's test and multiple
  testing correction
- :blue-bold:`Proteoform inference` via a reimplementation of the COPF
  algorithm for detecting functional proteoform groups from peptide-level data
- **Exploratory analysis** via direct access to scanpy routines for PCA, UMAP,
  and clustering
- **Publication-ready visualizations** for QC, exploratory analysis, and
  statistical results

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      How to install ProteoPy and its dependencies.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step notebooks for protein-level and peptide-level workflows.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete reference for all modules and functions.

   .. grid-item-card:: News
      :link: news/index
      :link-type: doc

      What's new in ProteoPy.

   .. grid-item-card:: Changelog
      :link: changelog
      :link-type: doc

      Full release history.

   .. grid-item-card:: Community
      :link: community
      :link-type: doc

      Get help, report issues, and connect with other users.

Source code
-----------

ProteoPy is open source and available on GitHub under an Apache 2.0 license: https://github.com/UKHD-NP/proteopy.

Citation
--------
ProteoPy is developed and maintained by the BludauLab in the Department of
Computational `Neuropathology
<https://www.klinikum.uni-heidelberg.de/pathologisches-institut/neuropathologie>`_,
`University Hospital Heidelberg <https://www.klinikum.uni-heidelberg.de/>`_.

If you use ProteoPy in your research, please cite:

.. bibliography::
   :filter: key == "fichtner-2026"

.. code-block:: bibtex

    @article{fichtner2026proteopy,
        title={ProteoPy: an AnnData-based framework for integrated proteomics analysis},
        author={Fichtner, Ian Dirk and Sahm, Felix and Gerstung, Moritz and Bludau, Isabell},
        journal={UNPUBLISHED},
        year={2026}
    }

If you use the COPF proteoform inference functionality, please also cite:

.. bibliography::
   :filter: key == "bludau-2021"

.. code-block:: bibtex

    @article{bludau2021systematic,
        title={Systematic detection of functional proteoform groups from bottom-up proteomic datasets},
        author={Bludau, Isabell and Frank, Max and D{\"o}rig, Christian and Cai, Yujia and Heusel, Moritz and Rosenberger, George and Picotti, Paola and Collins, Ben C. and R{\"o}st, Hannes and Aebersold, Ruedi},
        journal={Nature Communications},
        volume={12},
        pages={3810},
        year={2021},
        doi={10.1038/s41467-021-24030-x}
    }

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API reference

   changelog

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: About

   news/index
   community
   contributors
   acknowledgements
