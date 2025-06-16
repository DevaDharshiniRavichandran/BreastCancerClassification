# Breast Cancer Genetic Mutation Classification using 3D MRI

This project aims to enhance the classification of **breast cancer genetic mutations** using **3D MRI data** — a domain with limited prior research and few publicly available datasets. The complex nature of MRI signals and typically small sample sizes make this a challenging problem.

> Genetic mutations such as **BRCA1/2** cannot be directly identified from MRI scans alone and must be confirmed through genetic testing. Mutation labels used in this study are derived from verified genetic test results.

Traditional methods often rely on handcrafted features or transfer learning from non-medical or 2D modalities, which may not generalize well to 3D medical imaging [1], [2].

To address these challenges, this project:
- Evaluates multiple **3D convolutional neural networks (CNNs)**, including architectures pretrained on medical imaging tasks.
- Implements comprehensive **preprocessing pipelines** to normalize orientations, remove background noise, and enhance image contrast.
- Applies advanced **data augmentation techniques** to mitigate limited sample size.
- Benchmarks models using metrics such as **precision**, **recall**, **AUC**, and **F1-score**.

This study bridges the gap between medical imaging and genetic mutation prediction by leveraging recent advancements in **3D vision models** and **deep learning for medical diagnostics**.
