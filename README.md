# Face-Verification-Deep-learning
Face Verification System: Custom CNN achieving 98%+ accuracy using PyTorch and advanced regularization

# Face Verification System with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**High-performance face verification system achieving 98%+ accuracy using custom CNN architecture**

[Features](#features) • [Architecture](#architecture) • [Results](#results) • [Installation](#installation) • [Usage](#usage)

</div>

---

## Project Overview

A deep learning face verification system that determines if two face images belong to the same person. Built from scratch using PyTorch with a custom EfficientNet-inspired architecture, achieving **98.1% verification accuracy** and **2.13% Equal Error Rate (EER)** on a large-scale dataset.

### Key Achievements

- **98.1% Verification Accuracy** on validation set
- **2.13% Equal Error Rate (EER)** - industry-competitive performance  
- **27M parameter model** trained from scratch (no pre-trained weights)
- **Custom architecture** with stochastic depth and progressive expansion ratios
- **80.1% Classification Accuracy** across 8,631 identities
- Robust to variations in lighting, pose, and facial expressions

### What Makes This Special

Unlike traditional face recognition that only works with known individuals, this system learns a **similarity metric** that generalizes to unseen faces. The model learns 512-dimensional embeddings that capture facial features, enabling it to verify identity even for people it has never encountered during training.

---

## Features

- **Custom CNN Architecture**: EfficientNet-inspired design with MBConv blocks
- **Advanced Regularization**: Stochastic depth, dropout, and label smoothing to prevent overfitting
- **Smart Training Strategy**: Cosine annealing LR schedule preventing learning rate collapse
- **Progressive Expansion**: Channel expansion ratios increase gradually (2→3→4→4) across stages
- **Data Augmentation**: Comprehensive augmentation pipeline including rotation, color jitter, and random erasing
- **Production-Ready**: Mixed precision training (FP16) for 35% faster training
- **Experiment Tracking**: Integrated with Weights & Biases for monitoring

---
