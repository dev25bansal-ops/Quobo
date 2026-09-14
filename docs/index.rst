Quobo — QUBO-based Quantum Feature Selection for Environmental Image Classification
====================================================================================

.. toctree::
   :maxdepth: 2

   api

Quobo is the IDP3 pipeline for resource-constrained waste-image classification:
TACO crops -> MobileNetV2 embeddings -> leak-free per-split PCA -> QUBO feature
selection (8-of-50) -> Quantum SVM (QSVC) + classical baselines -> pollution
analytics (cleanliness score + Pollution Severity Index).
