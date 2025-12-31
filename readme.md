# Minimal BERT Implementation (Pure NumPy)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy%20Only-orange?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Caravan Research Paper Study Team - Alki**
>
> *An educational implementation of BERT built from scratch using only NumPy, co-developed with GPT-4o.*
>
> *GPT-4o의 도움을 받아 오직 NumPy만을 사용하여 바닥부터(Scratch) 구현한 교육용 BERT 프로젝트입니다.*

---

## 📖 Project Overview (프로젝트 개요)

This repository contains a minimalist implementation of **BERT (Bidirectional Encoder Representations from Transformers)**. Unlike standard implementations relying on deep learning frameworks like PyTorch or TensorFlow, this project builds the entire architecture—including the automatic differentiation (autograd) engine—using **only NumPy**.

This project aims to provide a deep understanding of the mathematical principles behind the Transformer architecture.

이 저장소는 딥러닝 프레임워크 없이 **순수 NumPy**만으로 구현한 BERT 모델을 담고 있습니다. 자동 미분(Autograd) 엔진부터 어텐션 메커니즘까지 직접 구현하여 트랜스포머의 내부 동작 원리를 학습하기 위해 만들어졌습니다.

### ⚠️ Note
* **Computation:** CPU Only (Optimized for educational clarity, not speed).
* **Dependency:** Zero external DL libraries (No Torch, No TF).

---

## 📂 Repository Structure (폴더 구조)

```bash
AlkiGit/
├── computation/        # Custom Autograd Engine & Tensor Operations
│                       # (자동 미분 및 텐서 연산 모듈)
├── dataset/            # Data Loading & Preprocessing Utilities
│                       # (데이터 로더 및 전처리 유틸리티)
├── model/              # BERT Architecture (Layers, Encoder, Attention)
│                       # (BERT 모델 아키텍처 구현체)
├── tokenization/       # WordPiece Tokenizer Implementation
│                       # (WordPiece 토크나이저)
├── bert_model.npz      # Pre-trained Model Weights (NumPy Archive)
│                       # (사전 학습된 모델 가중치 파일)
├── train.py            # Main Training Entrypoint
│                       # (학습 실행 스크립트)
└── LICENSE             # MIT License
```

---

## ✨ Key Features (핵심 기능)

* **🚫 No External Frameworks:** Pure Python & NumPy implementation. (외부 딥러닝 프레임워크 미사용)
* **⚙️ Custom Autograd:** Lightweight reverse-mode automatic differentiation engine. (직접 구현한 자동 미분 엔진)
* **🔤 WordPiece Tokenizer:** Custom subword tokenization logic. (WordPiece 토크나이저 내장)
* **🧠 BERT Components:**
    * Multi-Head Self Attention
    * Layer Normalization & Residual Connections
    * Feed-Forward Networks
    * GELU Activation
* **💾 Model Persistence:** Save and load weights using `.npz` format. (가중치 저장 및 로드 지원)

---

## 🚀 Getting Started (시작하기)

### 1. Prerequisites
All you need is Python and NumPy.

```bash
pip install numpy
```

### 2. Training (학습하기)
To start training the model from scratch using the provided script:

```bash
python train.py
```

### 3. Loading Pre-trained Weights (가중치 로드)
You can load the included `bert_model.npz` to test the model without training.

```python
import numpy as np

# Load the weights
data = np.load('bert_model.npz')

# List all layers/weights stored
print(data.files)
```

---

## 👨‍💻 Developed By

**Caravan Research Paper Study Team - Alki**

This project serves as study material for:
* Computational Graphs & Backpropagation
* Matrix Calculus in Deep Learning
* Transformer Attention Mechanisms

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
