# Closing the Gap: Improving Vision Transformers on Small-Scale Image Classification

This project explores architectural and training optimizations to improve the performance of **Vision Transformers (ViTs)** on small-scale, low-resolution datasets (CIFAR-10 and CIFAR-100), where they traditionally underperform compared to **Convolutional Neural Networks (CNNs)**.

## Project Overview
While ViTs achieve state-of-the-art results on large-scale datasets, they struggle with 32x32 pixel images due to a lack of inherent inductive biases (translation equivariance and locality). Our team implemented a modified, CLS-token-free ViT to narrow the performance gap between Transformers and a ResNet-18 baseline.

### Key Modifications
- **Patch Size Reduction:** Decreased from 16x16 to 8x8 to increase token granularity for low-resolution inputs.
- **CLS-free Architecture:** Replaced the classification token with **Global Average Pooling (GAP)** to enhance training stability and Keras compatibility.
- **Regularization:** Integrated **Stochastic Depth** to mitigate overfitting in data-constrained environments.
- **Optimizer & Schedule:** Utilized AdamW with a **Cosine Decay with Restarts** learning rate schedule.

## Experimental Results
Both models were trained from scratch under a unified pipeline to ensure a fair comparison.

| Model | Dataset | Accuracy |
| :--- | :--- | :--- |
| **Baseline ViT** | CIFAR-10 | 40.1% |
| **Modified ViT** | CIFAR-100 | 44.2% |
| **Modified ViT** | CIFAR-10 | 63.5% |
| **ResNet-18** | CIFAR-10 | 80.0% |
| **ResNet-18** | CIFAR-100 | 46.5% |

### Key Findings
- The modified ViT achieved a **23.4% accuracy gain** on CIFAR-10 compared to the baseline.
- **t-SNE Visualizations** indicate that while ViTs can learn coarse categories (CIFAR-10), they struggle with fine-grained classification (CIFAR-100) due to high semantic noise and overlapping decision boundaries.
- CNNs remain the more robust choice for small-scale datasets, as their hard-coded spatial priors provide a decisive advantage when pretraining data is unavailable.

## Methodology
- **Framework:** TensorFlow / Keras
- **Visualization:** Matplotlib, Scikit-learn (t-SNE)
- **Data Augmentation:** Standard flips and rotations were applied to both architectures for consistency.

## Conclusion & Future Work
The performance gap can be narrowed through token-level adjustments and stability-focused architectural changes. Future research will explore 4x4 patch scales and more aggressive data augmentation strategies to further compensate for the absence of structural geometric priors in Transformer architectures.

## Contributors
- Will Chiappetta
- Dylan Benson
- Nicholas Cavallaro
