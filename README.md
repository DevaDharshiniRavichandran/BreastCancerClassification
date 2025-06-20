# Breast Cancer Genetic Mutation Classification using 3D MRI

## Overview

This project focuses on the classification of breast cancer genetic mutations—such as **BRCA1/2**—using **3D MRI** data. Due to the complexity of MRI signals and the small sample sizes in medical imaging, this remains a challenging and underexplored research area.

## Motivation

Genetic mutations like BRCA1/2 cannot be directly visualized through imaging alone. Mutation status used in this study is derived from **confirmed genetic tests**, not from imaging labels. This approach allows us to train models that learn subtle imaging correlates of genetic mutations.

While previous methods have relied on:
- **Handcrafted features**, or
- **Transfer learning** from 2D or non-medical image datasets,

these techniques often **fail to generalize** to 3D medical data.

## Goals

To overcome these limitations, the project aims to:

- Evaluate multiple **3D Convolutional Neural Networks (3D CNNs)**, including models pretrained on medical imaging tasks.
- Apply **preprocessing techniques** for orientation normalization, noise reduction, and contrast enhancement.
- Use **advanced data augmentation** methods tailored for 3D volumetric data.
- Benchmark performance with **precision, recall, F1-score, and AUC** metrics.

## Methodology

### Data Preprocessing
- Orientation standardization
- Skull stripping and background removal
- Intensity normalization and contrast adjustment

### Models Used
- 3D ResNet
- MedicalNet
- Custom lightweight 3D CNNs

### Evaluation Metrics
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

## Contributions

This study contributes to the growing body of research bridging **medical imaging** and **genetic profiling** by exploring the feasibility of deep learning models to infer genetic risk markers from imaging features.

## References

[1] Litjens et al., A survey on deep learning in medical image analysis  
[2] Tajbakhsh et al., Convolutional neural networks for medical image analysis: Full training or fine tuning?

## Future Work

- Multimodal integration with genomic, histopathology, and clinical data
- Fine-tuning on larger, multi-institutional datasets
- Explainability and model interpretability using Grad-CAM or SHAP
