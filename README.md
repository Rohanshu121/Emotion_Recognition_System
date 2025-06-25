# MoodX: Multi-Modal Emotion Recognition System

**MoodX** is a lightweight, modular architecture for emotion recognition based on valence-arousal modeling. It integrates visual, audio, and textual modalities through bi-directional cross-modal attention and predicts emotional states in polar coordinates, which are converted to valence and arousal values.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Training](#training)
* [Evaluation](#evaluation)
* [Usage](#usage)
* [License](#license)

---

## Overview

MoodX processes multi-modal inputs in conversational videos using:

* **Visual**: Swin Transformer-based visual encoder
* **Audio**: HuBERT-based audio encoder
* **Text**: RoBERTa-based text encoder

These are fused through **cross-modal attention** and refined with self-attention (BEiT), producing emotion predictions in polar form (θ, intensity), then transformed into valence and arousal scores.

---

## Dataset

* Trained and evaluated on the **Aff-Wild2** dataset
* Contains 594 videos with \~3M frames annotated for **valence** and **arousal**

---

## Architecture

1. **Feature Extractors** (frozen):

   * Swin Transformer for visual
   * HuBERT for audio
   * RoBERTa for text

2. **Cross-Modal Attention** (6 directions between all modality pairs)

3. **Self-Attention Fusion** using BEiT

4. **MLP Head** for polar emotion output → mapped to valence/arousal

---

## Training

* **Optimizer**: Adam (`lr=1e-4`, `weight_decay=1e-4`)
* **Scheduler**: ReduceLROnPlateau (`factor=0.1`, `patience=5`)
* **Epochs**: 100 (early stopping after 10 epochs)
* **Batch Size**: 8

---

## Evaluation

Measured using **Concordance Correlation Coefficient (CCC)**:

$$
P = \frac{CCC_{\text{valence}} + CCC_{\text{arousal}}}{2}
$$

**Baseline (ResNet-50):**

* Valence CCC: 0.24
* Arousal CCC: 0.20
* Average Score: 0.22

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Rohanshu121/Emotion_Recognition_System.git
cd Emotion_Recognition_System
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Ensure the Aff-Wild2 dataset is placed under `data/Aff-Wild2/`.

### 4. Train the Model

```bash
python embeddings.py
python Train_BEiT.py
python Train_MLP.py
```

### 5. Evaluate the Model

```bash
python Test.py
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.