# LLM-Transformers-NER-FineTuning-HuggingFace-BERT-XLM

<p align="center">
    <a href="https://huggingface.co/xlm-roberta-base">
        <img src="https://huggingface.co/xlm-roberta-base/resolve/main/icon.svg" alt="Model Badge" height="80">
    </a>
</p>

<p align="center">
    <a href="https://pytorch.org">
        <img src="https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg" alt="PyTorch">
    </a>
    <a href="https://huggingface.co">
        <img src="https://img.shields.io/badge/Transformers-4.47.1-blue.svg" alt="Transformers">
    </a>
    <a href="https://huggingface.co">
        <img src="https://img.shields.io/badge/Datasets-3.2.0-yellow.svg" alt="Datasets">
    </a>
    <a href="https://huggingface.co">
        <img src="https://img.shields.io/badge/Tokenizers-0.21.0-red.svg" alt="Tokenizers">
    </a>
</p>

---

## 🌟 Model Overview

`LLM-Transformers-NER-FineTuning-HuggingFace-BERT-XLM` is a fine-tuned version of [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) for multilingual Named Entity Recognition (NER) tasks. The model is fine-tuned for extracting entities such as names, locations, and organizations from texts in **English**, **French**, and **Italian**, leveraging the multilingual capabilities of the base model.

### 🤖 Model Details
- **Base Model**: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- **Dataset**: PAN-X (specific version unspecified)
- **Languages Supported**: English, French, Italian (with multilingual support available)

### 🏆 Results
On the evaluation set:
- **Loss**: `0.3151`

---

## 📈 Training and Evaluation Data

This model was fine-tuned using a multilingual dataset. Details regarding the dataset’s composition or preprocessing steps are not provided but are essential for replication.

---

## 🔧 Training Procedure

### Hyperparameters

The following hyperparameters were used during training:
- **Learning Rate**: `5e-05`
- **Training Batch Size**: `12`
- **Evaluation Batch Size**: `12`
- **Seed**: `42`
- **Optimizer**: AdamW (`betas=(0.9, 0.999)`, `epsilon=1e-08`)
- **Learning Rate Scheduler**: Linear decay
- **Number of Epochs**: `3`

### Training Results

| **Training Loss** | **Epoch** | **Step** | **Validation Loss** |
|:-----------------:|:---------:|:--------:|:-------------------:|
| `0.5243`          | `1`       | `334`    | `0.3466`            |
| `0.2758`          | `2`       | `668`    | `0.3423`            |
| `0.1768`          | `3`       | `1002`   | `0.3151`            |

---

## 🛠️ Framework Versions

The following versions of frameworks were used:
- **Transformers**: `4.47.1`
- **PyTorch**: `2.5.1+cu121`
- **Datasets**: `3.2.0`
- **Tokenizers**: `0.21.0`

---

## 📂 Installation

To reproduce the training or fine-tuning process, install the dependencies via `requirements.txt`. Set up a virtual environment and run:

```bash
pip install -r requirements.txt
