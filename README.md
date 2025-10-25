# Sequential Sales Prediction with PyTorch
**Independent Consulting Project – Industrial Client**

This project builds a sequential recommendation model to predict the next product category a B2B client is likely to purchase. The goal is to support sales teams and inventory planning by anticipating client needs based on their procurement history.

> **Note:** This repository contains **only the research notebook** used for prototyping and experimentation.  
> The full production code (dataset, training pipeline, inference API) was developed privately for the client and is not included here due to confidentiality.


## What the Model Does
- Represents each client’s purchase history as a variable-length sequence
- Uses **categorical embeddings** + normalized numerical features
- Learns temporal patterns with **GRU4Rec and LSTM** models, including an **attention mechanism**
- Handles **class imbalance** via weighted sampling and **Focal Loss**
- Uses **Optuna** to tune hyperparameters and improve generalization

## Results
Best-performing model: **GRU4Rec + Attention**

| Metric | Score |
|-------|------|
| **Hit@1** | ~51% |
| **Hit@3** | ~78% |
| **Hit@5** | ~89% |

This means the correct next product category typically appears within the **top recommended options**, making the model useful for sales assistance rather than strict automation.

## Repository Structure
```
.
├── notebooks/              # Exploratory data analysis and model development
├── src/
│   ├── dataset.py          # Custom Dataset and collate function for variable sequences
│   ├── models.py           # LSTMRec and GRU4Rec implementations with attention
│   ├── train.py            # Training loop with early stopping and validation
│   ├── tune.py             # Optuna hyperparameter optimization pipeline
│   └── inference.py        # Top-k prediction for new industrial clients
├── requirements.txt        # Project dependencies
└── README.md
```
## Business Impact
- Proactive and relevant product recommendations
- More efficient inventory and procurement planning
- Stronger client engagement and retention

## Data Privacy
All client identifiers and product labels have been anonymized. The sample dataset provided is **synthetic** and intended for demonstration only.
