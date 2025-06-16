This project aims to enhance the classification of breast cancer genetic mutations using
3D MRI data—a domain with limited prior research and few publicly available datasets.
The complex nature of MRI signals and the typically small sample sizes make this a
challenging task. More importantly, it is not possible to directly identify genetic mutations
such as BRCA1/2 solely from breast MRI; such mutations must be confirmed through
genetic testing. Traditional methods often rely on handcrafted features or transfer
learning from non-medical or 2D modalities, which may not generalize well to 3D medical
imaging [1], [2]. In this study, mutation labels are derived from confirmed genetic testing
results. To address existing gaps in the literature, we evaluate multiple 3D convolutional
neural networks (CNNs) and preprocessing pipelines, leveraging recent advancements
in medical vision models and data augmentation techniques to improve classification
accuracy.
