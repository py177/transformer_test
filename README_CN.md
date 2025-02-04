# 🔥 高级注意力机制比较研究

**[English Version 🇺🇸](README_EN.md)**

本项目实现并比较了多种 **注意力机制（Attention Mechanisms）** 的性能，涵盖从 **自注意力（Self-Attention）** 到 **记忆增强注意力（Memory-Augmented Attention）** 等多个变体。代码编写过程中 **参考了大语言模型的提示**，并在此基础上进行实现和优化。

## 📌 项目简介
注意力机制是深度学习领域的核心组件，广泛应用于 **自然语言处理（NLP）**、**计算机视觉（CV）** 和 **图神经网络（GNN）** 等多个领域。本项目包含 **20+ 种不同的注意力机制**，并提供了性能比较方法，帮助研究者更好地理解不同注意力机制的适用场景及优势。

## 🔧 支持的注意力机制
本项目实现并评估以下注意力机制：

1. **多头注意力（Multi-Head Attention）**：允许模型同时关注不同的信息子空间。
2. **自适应注意力（Adaptive Attention）**：动态调整注意力的聚焦点。
3. **局部注意力（Local Attention）**：关注序列中的局部区域以提高效率。
4. **全局注意力（Global Attention）**：在整个序列上计算注意力。
5. **层次注意力（Hierarchical Attention）**：多级别的注意力机制，适用于复杂结构。
6. **交叉注意力（Cross-Attention）**：在不同模态或流程间共享注意力。
7. **自注意力（Self-Attention）**：一个序列内部元素间的注意力机制。
8. **稀疏注意力（Sparse Attention）**：只关注重要的键值对，提高效率。
9. **卷积注意力（Convolutional Attention）**：结合卷积操作以捕捉局部模式。
10. **门控注意力（Gated Attention）**：通过门控机制控制信息流。
11. **对抗性注意力（Adversarial Attention）**：使用对抗训练来改善注意力的鲁棒性。
12. **图注意力（Graph Attention）**：用于处理图结构数据。
13. **硬注意力（Hard Attention）**：基于离散选择，而非软性权重分配。
14. **软注意力（Soft Attention）**：连续且可微的注意力分配。
15. **Transformer-XL 的段级重复注意力**：增强对长期依赖的捕捉。
16. **BERT 的双向注意力**：在所有层中结合左右两侧的上下文。
17. **混合注意力（Hybrid Attention）**：结合不同类型的注意力机制。
18. **协同注意力（Co-Attention）**：同时在两个相关序列上应用注意力。
19. **轴向注意力（Axial Attention）**：沿特定维度应用注意力，用于高维数据。
20. **频域注意力（Frequency Domain Attention）**：在频域内应用注意力。
21. **注意力蒸馏（Attention Distillation）**：从一个模型到另一个模型转移注意力模式。
22. **注意力池化（Attention Pooling）**：利用注意力权重进行特征池化。
23. **记忆增强注意力（Memory-Augmented Attention）**：引入外部记忆机制以增强注意力。

## 🚀 安装与环境配置
本项目使用 `Python` 及 `PyTorch` 进行实现，并提供 `requirements.txt` 进行环境配置。

### 1️⃣ **克隆本项目**
```sh
git clone https://github.com/py177/attention-models.git
cd attention-models
```

### 2️⃣ **安装依赖**
使用 `pip` 安装必要的依赖：
```sh
pip install -r requirements.txt
```
如果你使用 `conda`，可以创建一个新的环境：
```sh
conda create -n attention_env python=3.8
conda activate attention_env
pip install -r requirements.txt
```

## 📊 运行模型并比较性能
项目提供 `main.py` 入口文件，可运行不同注意力机制并进行性能比较。

### 1️⃣ **运行所有注意力机制**
```sh
python main.py
```
运行后，系统会：
- 计算各注意力机制在不同任务上的 **计算成本、准确率和收敛速度**；
- 生成 **可视化分析**，包括 **损失曲线（Loss Curve）** 和 **注意力权重（Attention Weights）**。

### 2️⃣ **单独测试某个注意力机制**
你可以指定某种注意力机制进行测试，例如：
```sh
python main.py --mode multihead
```

## 📜 代码参考来源
- 本项目的代码编写过程中，**一定程度上参考了大语言模型（如 ChatGPT）的提示**，并基于 PyTorch 和 TensorFlow 进行实现。
- 相关文献和研究工作已在 `docs/references.md` 文件中列出。

## 🔬 研究与应用场景
本项目可用于：
- **自然语言处理（NLP）**：机器翻译、文本摘要、问答系统等。
- **计算机视觉（CV）**：目标检测、图像生成等。
- **强化学习（RL）**：自适应策略学习。
- **多模态学习（Multi-Modal Learning）**：跨模态数据建模。

## 🔗 相关研究论文
以下是与本项目相关的一些重要论文：
1. Vaswani et al. (2017) - **"Attention Is All You Need"** (Transformer)
2. Devlin et al. (2019) - **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**
3. Dai et al. (2019) - **"Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"**
4. Yang et al. (2020) - **"XLNet: Generalized Autoregressive Pretraining for Language Understanding"**
5. Brown et al. (2020) - **"Language Models are Few-Shot Learners"** (GPT-3)

## 🤝 贡献指南
欢迎贡献代码和优化：
1. **Fork 本仓库**
2. **创建新分支** (`git checkout -b feature-newattention`)
3. **提交更改** (`git commit -m "添加新的注意力机制"`)
4. **推送分支** (`git push origin feature-newattention`)
5. **创建 Pull Request**

## 📄 许可证
本项目基于 **MIT 许可证** 开源，详细信息请查看 [LICENSE](LICENSE) 文件。
