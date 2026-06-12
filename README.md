# IT721 — Applied Research Topics in Deep Learning
**Walsh College · Spring 2026 · Lamonte Smith**

![Course](https://img.shields.io/badge/course-IT721-blue) ![Institution](https://img.shields.io/badge/institution-Walsh%20College-darkred) ![Python](https://img.shields.io/badge/python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange) ![Keras](https://img.shields.io/badge/keras-3.x-red) ![Platform](https://img.shields.io/badge/platform-Google%20Colab%20T4-yellow) ![License](https://img.shields.io/badge/use-academic-lightgrey)

---

## Course Overview

IT721 is a doctoral-level deep learning course covering neural network architectures, optimization algorithms, regularization techniques, and state-of-the-art applications. The course progresses from feed-forward foundations through convolutional networks, recurrent networks, generative models, and large-scale capability/safety analysis — culminating in a comparative architecture study capstone.

This repository contains all seven case studies (Weeks 2–7) and the Week 10 Capstone Project. Each notebook includes complete code, executed outputs from a Google Colab T4 GPU environment, methodological commentary, and an inline AI-assistance disclosure block consistent with Walsh College's Academic Conduct Policy and written authorization from Professor Raftari (instructor of record, IT721 VS1).

---

## Case Studies

| Week | Topic | Notebook |
|------|-------|------|
| Week 2 | Neural Networks & Advanced Backpropagation | `CaseStudy2Submission_NeuralNetworks&AdvancedBackpropagation_LSmith-4-15-26.ipynb` |
| Week 3 | CNN Image Classification with Transfer Learning | `IT721Week3CaseStudy_LSmith.ipynb` |
| Week 4 | RNN & LSTM for Sequence Modeling | `IT721Week4CaseStudy_LSmith_4-27-26.ipynb` |
| Week 5 | Autoencoders & Generative Models | `IT721Week5CaseStudy_LSmith.ipynb` |
| Week 6 | Cats vs Dogs CNN Lab | `IT721Week6CaseStudy_LSmith.ipynb` |
| Week 7 | CNN Capabilities & Scaling Risks | `IT721Week7CaseStudy_Capabilities&ScalingRisks_LSmith.ipynb` |
| Week 10 | **Capstone Project** | `IT721_Capstone_LSmith.ipynb` |

---

## Capstone Spotlight — CNN Architecture Comparison for Binary Image Classification

**File:** `IT721_Capstone_LSmith.ipynb`

The Week 10 Capstone implemented and compared four convolutional neural network architectures for binary image classification on the Kaggle Cats vs Dogs dataset (18,610 train / 2,326 validation / 2,326 test images at 224×224 resolution). Models were trained on Google Colab T4 with seed-controlled reproducibility and a unified data pipeline (normalization, augmentation, prefetching).

### Headline Results

| Model | Test Accuracy | F1 Score | Parameters | Train Time |
|---|---|---|---|---|
| Simple Custom CNN | 84.69% | 0.8469 | 12.9M | 723.7s |
| VGG16 (transfer) | 93.68% | 0.9368 | 14.8M | 1,442.3s |
| ResNet50 (transfer) | 64.49% | 0.6436 | 23.9M | 778.0s |
| **EfficientNetB0 (transfer)** | **97.89%** | **0.9789** | **4.2M** | 803.9s |

### Key Findings

- **EfficientNetB0 won on every dimension** — highest accuracy, highest F1, fewest parameters by a significant margin, and best accuracy-per-second of training time. Compound scaling enables effective ImageNet representation transfer for pet classification.
- **ResNet50 underperformed** due to a known frozen-BatchNormalization interaction issue when input resolution differs from ImageNet pretraining statistics (He et al., 2016). Unfreezing BN layers resolved the same issue for EfficientNetB0.
- **Feature-space analysis** via t-SNE confirmed EfficientNetB0 produces dense, well-separated class clusters in its 1,280-dimensional feature space, while the Simple CNN's 128-dimensional features show 55.1% ReLU-induced sparsity, limiting discriminative capacity.
- **Production recommendation:** EfficientNetB0 — strongest candidate for mobile and edge deployment via quantization, optimal accuracy/parameter tradeoff for real-time inference.

### Ethical Considerations

The Capstone includes an ethical analysis of pet-classification deployment in shelter management, veterinary triage, and insurance contexts — specifically the breed-level bias that arises when training corpora over-represent certain breeds, producing systematic misclassification of rare breeds. Per-breed performance auditing is recommended before any production deployment to ensure equitable classification across the full distribution of domestic cat and dog breeds.

### References (Capstone)

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv:1704.04861*.

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.

---

## Tools & Frameworks

- TensorFlow 2.x / Keras 3.x
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- Google Colab (T4 GPU)
- Python 3.x

---

## Models Implemented Across All Case Studies

- Custom CNN — 82.13% CIFAR-10 test accuracy (Week 3 baseline)
- ResNet50 with Transfer Learning — 84.46%
- EfficientNetB0 with Transfer Learning + Fine-tuning — 97.89% Dogs vs Cats (Capstone winner)
- Vanilla RNN, LSTM, GRU — single, stacked, bidirectional configurations (IMDB sentiment)
- LSTM with Bahdanau Attention — 86.31%
- Convolutional Autoencoder — SSIM 0.5844
- Variational Autoencoder (VAE) with reparameterization trick
- DCGAN for face generation (CelebA dataset)
- Cats vs Dogs CNN — 81.13% with save/reload validation (Week 6 lab)

---

## Repository Conventions

- All notebooks include preserved outputs from the original execution run
- Each notebook contains an inline AI-assistance disclosure block at the top, identifying specific parts where AI assistance was applied
- Filenames follow Walsh assignment naming conventions (week, topic, author, submission date where applicable)
- The Capstone notebook is exam-ready: deterministic seeding, unified data pipeline, fully reproducible training calls

---

## Contributors

- **Lamonte Smith** — Walsh College DBA (AI/ML Leadership) and PhD in Technology (Cybersecurity) candidate; primary author and submitter of all case studies and the Capstone Project
- **Claude (Anthropic)** — AI assistant used for code scaffolding, methodological review, written-analysis structure, and APA 7 reference formatting, under written instructor authorization

---

## Author

**Lamonte Smith** — Milford, MI
Senior Systems Design and Release Engineer · General Motors (Advanced Infotainment, Compute & Connectivity)
[LinkedIn](https://www.linkedin.com/in/lamonte-smith-7518b4248/) · [GitHub](https://github.com/LSmithPMP)
