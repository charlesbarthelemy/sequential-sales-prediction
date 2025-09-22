# Sequential Sales Prediction with PyTorch
**Industrial Client Consulting Project - Independent Consultant**

This repository demonstrates a **sequential recommendation and sales prediction system** built with **PyTorch** for an industrial enterprise client. The project leverages advanced sequence models (LSTM and GRU4Rec with attention), robust preprocessing, and automated hyperparameter tuning with **Optuna** to predict future product categories purchased by B2B clients based on their historical procurement patterns.

---
## Project Goal
The aim of this consulting project was to:
- Predict the **next product category** an industrial client is most likely to procure.  
- Handle **heterogeneous features**: categorical (e.g., product family, contract type, supplier segment) and numerical (e.g., order volume, contract value, usage frequency).  
- Manage **variable-length sequences** of client procurement histories using custom datasets and collate functions.  
- Address **class imbalance** in industrial product categories using Focal Loss and weighted sampling.  
- Optimize hyperparameters via cross-validated **Optuna tuning** to maximize business impact.

---
## Key Features
- **Custom Dataset** for sequence modeling with variable-length procurement histories.  
- **Embedding layers** for categorical features + normalized continuous features.  
- **Bidirectional LSTM and GRU4Rec** with attention mechanism for sequential pattern recognition.  
- **Cross-validation** with K-Fold split at the client level to ensure robust generalization.  
- **Weighted sampling** and **Focal Loss** to handle imbalanced industrial product categories.  
- **Optuna integration** for automated hyperparameter optimization.  
- **Top-k evaluation metrics (Hit@k)** and comprehensive class-level performance reports.

---
## Performance Results
After extensive tuning and training on industrial procurement data:
- **Best model**: GRU4Rec + Attention  
- **Primary evaluation metric**: Hit@3 (top-3 category accuracy)  
- **Business-relevant results**:  
  - Hit@1 ≈ **51%** (exact next category prediction)
  - Hit@3 ≈ **78%** (next category in top-3 recommendations)  
  - Hit@5 ≈ **89%** (next category in top-5 recommendations)  
- **F1-scores by category**:  
  - Strong performance on frequent industrial categories (>0.60 F1)  
  - Acceptable performance on specialized/rare categories through class reweighting  

These results demonstrate significant business value for procurement planning and inventory optimization in industrial settings.

---
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

---
## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sequential-sales-prediction.git
   cd sequential-sales-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training**
   ```bash
   python src/train.py
   ```

4. **Run hyperparameter tuning**
   ```bash
   python src/tune.py
   ```

5. **Generate predictions**
   ```bash
   python src/inference.py
   ```

---
## Data Privacy & Confidentiality
- **Full anonymization**: All client identifiers, company names, and business-sensitive variables have been completely anonymized or replaced with synthetic equivalents.
- **Industrial data protection**: Specific product categories, pricing information, and procurement volumes have been masked to protect client confidentiality.
- **Demonstration purposes**: The provided dataset serves solely for code demonstration and pipeline reproducibility - it cannot be used for actual business predictions.
- **Consultant ethics**: This project adheres to strict data privacy standards required in independent consulting engagements.

---
## Industrial Application Context
This solution was developed as part of an independent consulting engagement to help optimize:
- **Procurement planning** and inventory management
- **Sales forecasting** for B2B industrial products  
- **Customer relationship management** through predictive insights
- **Supply chain optimization** via demand anticipation

The model architecture and evaluation framework can be adapted for various industrial contexts including manufacturing, construction, and specialized equipment sectors.

---
## Business Impact
The deployed model enables:
- **Proactive inventory management** based on predicted client needs
- **Targeted sales recommendations** for account managers
- **Optimized procurement cycles** reducing stockouts and overstock
- **Enhanced client satisfaction** through anticipatory service

---
*Developed as an independent consultant for industrial enterprise optimization*
