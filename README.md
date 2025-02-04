# 🔥 Advanced Attention Mechanisms Comparison

**[简体中文 🇨🇳](README.md)**

This project implements and compares multiple **attention mechanisms**, covering various variants from **Self-Attention** to **Memory-Augmented Attention**. During the development process, **large language model prompts were referenced**, and the implementations were optimized accordingly.

## 📌 Project Overview
Attention mechanisms are core components in deep learning, widely applied in **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **Graph Neural Networks (GNN)**. This project includes **20+ different attention mechanisms**, providing performance comparisons to help researchers better understand their applicability and advantages.

## 🔧 Supported Attention Mechanisms
This project implements and evaluates the following attention mechanisms:

1. **Multi-Head Attention**: Allows the model to attend to different subspaces of information simultaneously.
2. **Adaptive Attention**: Dynamically adjusts the focus of attention.
3. **Local Attention**: Focuses on local regions of a sequence to improve efficiency.
4. **Global Attention**: Computes attention over the entire sequence.
5. **Hierarchical Attention**: Multi-level attention mechanism suitable for complex structures.
6. **Cross-Attention**: Shares attention between different modalities or processes.
7. **Self-Attention**: Attention mechanism within elements of the same sequence.
8. **Sparse Attention**: Focuses only on important key-value pairs to improve efficiency.
9. **Convolutional Attention**: Combines convolution operations to capture local patterns.
10. **Gated Attention**: Uses gating mechanisms to control information flow.
11. **Adversarial Attention**: Improves robustness through adversarial training.
12. **Graph Attention**: Designed for processing graph-structured data.
13. **Hard Attention**: Based on discrete selection rather than soft weight allocation.
14. **Soft Attention**: Continuous and differentiable attention allocation.
15. **Segment-Level Recurrence in Transformer-XL**: Enhances the capture of long-term dependencies.
16. **Bidirectional Attention in BERT**: Combines left and right context at all layers.
17. **Hybrid Attention**: Combines different types of attention mechanisms.
18. **Co-Attention**: Applies attention simultaneously on two related sequences.
19. **Axial Attention**: Applies attention along specific dimensions for high-dimensional data.
20. **Frequency Domain Attention**: Applies attention in the frequency domain.
21. **Attention Distillation**: Transfers attention patterns from one model to another.
22. **Attention Pooling**: Uses attention weights for feature pooling.
23. **Memory-Augmented Attention**: Introduces external memory to enhance attention.

## 🚀 Installation & Environment Setup
This project is implemented using `Python` and `PyTorch`, with dependencies listed in `requirements.txt`.

### 1️⃣ **Clone this repository**
```sh
git clone https://github.com/py177/attention-models.git
cd attention-models
```

### 2️⃣ **Install dependencies**
Using `pip`:
```sh
pip install -r requirements.txt
```
Using `conda`:
```sh
conda create -n attention_env python=3.8
conda activate attention_env
pip install -r requirements.txt
```

## 📊 Running Models & Performance Comparison
The project provides `main.py` as the entry point to run different attention mechanisms and compare performance.

### 1️⃣ **Run all attention mechanisms**
```sh
python main.py
```
Upon execution, the system will:
- Compute **computational cost, accuracy, and convergence speed** for each attention mechanism.
- Generate **visual analyses**, including **loss curves** and **attention weight distributions**.

### 2️⃣ **Run a specific attention mechanism**
To test a particular attention mechanism, specify it as follows:
```sh
python main.py --mode multihead
```

## 📜 Code References
- During implementation, **large language model prompts (e.g., ChatGPT) were referenced** and adapted using PyTorch and TensorFlow.
- Related literature and research are listed in `docs/references.md`.

## 🔬 Research & Applications
This project can be used for:
- **Natural Language Processing (NLP)**: Machine translation, text summarization, question-answering systems.
- **Computer Vision (CV)**: Object detection, image generation.
- **Reinforcement Learning (RL)**: Adaptive strategy learning.
- **Multi-Modal Learning**: Modeling across different data modalities.

## 🔗 Relevant Research Papers
Some important papers related to this project include:
1. Vaswani et al. (2017) - **"Attention Is All You Need"** (Transformer)
2. Devlin et al. (2019) - **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**
3. Dai et al. (2019) - **"Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"**
4. Yang et al. (2020) - **"XLNet: Generalized Autoregressive Pretraining for Language Understanding"**
5. Brown et al. (2020) - **"Language Models are Few-Shot Learners"** (GPT-3)

## 🤝 Contribution Guidelines
Contributions are welcome! Follow these steps:
1. **Fork this repository**
2. **Create a new branch** (`git checkout -b feature-newattention`)
3. **Commit your changes** (`git commit -m "Added new attention mechanism"`)
4. **Push your branch** (`git push origin feature-newattention`)
5. **Create a Pull Request**

## 📄 License
This project is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for details.
