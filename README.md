# GS-ChemDRP
## Gene Set-guided ChemBERTa for Drug Response Prediction

약물 표적 경로 기반 대조 학습과 Cross-Attention을 이용한 항암제 민감도 예측 모델

---

## Overview

GS-ChemDRP는 항암제의 세포주별 약물 반응(IC50)을 예측하는 딥러닝 모델입니다. 
사전학습된 화학 언어 모델 ChemBERTa와 유전자 발현 데이터를 결합하며, 
GDSC에서 제공하는 약물 표적 경로(drug target pathway) 정보를 대조 학습으로 활용합니다.

Key Features:
- Lightweight model: No graph neural networks required, uses only frozen ChemBERTa and gene embeddings
- Biological interpretability: Drug-pathway interactions interpretable through gene set gating
- Pathway information integration: Supervised contrastive learning based on drug target pathways
- Strong performance: Test PCC 0.9220 (GDSC dataset, 949 cell lines)

---

## Performance Results

GDSC Dataset (949 cell lines):

| Model | Test PCC | Test RMSE | Notes |
|-------|----------|-----------|-------|
| GS-ChemDRP (Final) | 0.9220 | 1.0682 | This work |
| + Gene Set Gating | 0.9198 | 1.0812 | Gating only |
| + Contrastive Learning | 0.9148 | 1.1090 | Contrastive only |
| Baseline (ChemBERTa) | 0.9112 | 1.1327 | Baseline |

Ablation Study:
- ChemBERTa Baseline: PCC 0.9112
- + Contrastive Learning (+0.0036): PCC 0.9148
- + Gene Set Gating (+0.0050): PCC 0.9220

---

## Model Architecture

### Overall Model Structure

![Model Overview](figures/figure1.png)

The model consists of two main branches:
- **Drug Encoder**: ChemBERTa-based encoder for drug representation
- **Cell Encoder**: Gene set-based representation for cell lines
- **Interaction Module**: Gating mechanism and cross-attention for drug-cell interaction
- **Prediction Head**: MLP for IC50 prediction

### Detailed Architecture

![Detailed Architecture](figures/figure2.png)

**Drug Branch:**
- Input: SMILES strings
- ChemBERTa-based drug encoder (pre-trained on 77M SMILES)
- Supervised contrastive learning using drug target pathway information
- Output: Drug embedding [256-dim]

**Cell Branch:**
- Input: Gene expression data (949 landmark genes)
- Gene set representation: 48 Hallmark pathways
- Gene set embedding and aggregation
- Output: Cell representation with pathway-aware features

### Contrastive Learning Framework

![Contrastive Learning](figures/figure3.png)

Supervised contrastive learning encourages drugs targeting the same pathways to have similar embeddings:
- Drug tokens from same target pathway should be close in embedding space
- SupCon loss with drug-pathway relationships from GDSC
- Soft regularization (lambda=0.1) to balance with main IC50 prediction task

Loss Function:
```
Loss = MSE(predicted_IC50, true_IC50) + lambda * SupCon_loss(drug_embeddings, pathway_labels)
```

---

## Installation and Usage

Requirements:
```bash
pip install -r requirements.txt
```

Libraries:
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- pandas, numpy, scikit-learn
- transformers (HuggingFace)

Quick Start (Sample Data):

```bash
# 1. Clone repository
git clone https://github.com/Jongwoongsong/GS-ChemDRP.git
cd GS-ChemDRP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test with sample data
python train.py \
    --data_dir data/sample \
    --epochs 5 \
    --batch_size 32 \
    --output_dir ./outputs_test

# 4. Check results
python -c "
import json
with open('./outputs_test/results.json') as f:
    results = json.load(f)
    print('Test PCC:', results['test_pcc'])
    print('Test RMSE:', results['test_rmse'])
"
```

---

## Data Preparation

GDSC Dataset

GS-ChemDRP uses GDSC (Genomics of Drug Sensitivity in Cancer) dataset.

Download:

```bash
# 1. GDSC official website
# https://www.cancerrxgene.org/downloads/anova/sanger1018_basic_clean.csv

# 2. Drug information (GDSC Release 8.4)
# https://www.cancerrxgene.org/downloads/drug_screens/screened_compounds_rel_8_4.csv

# 3. Gene expression data (949 landmark genes)
# LINCS L1000 or microarray data

# 4. Prepare with following structure
data/gdsc/
├── gdsc_train_landmark_baseline_clean.csv
├── gdsc_valid_landmark_baseline_clean.csv
├── gdsc_test_landmark_baseline_clean.csv
└── screened_compounds_rel_8_4.csv
```

Or request preprocessed dataset from authors.

Drug Target Pathway Information:

```bash
python create_drug_pathway_mapping.py \
    --drug_file screened_compounds_rel_8_4.csv \
    --output_file drug_pathway_mapping.json
```

---

## Model Training

Full Dataset Training (Recommended):

```bash
python train.py \
    --data_dir data/gdsc \
    --gene_sets_file gene_sets/hallmark_949.json \
    --embed_dim 256 \
    --num_heads 8 \
    --batch_size 32 \
    --epochs 100 \
    --early_stop_patience 10 \
    --lambda_con 0.1 \
    --lambda_gate_l1 0.01 \
    --lambda_gate_smooth 0.01 \
    --output_dir outputs_final
```

Hyperparameter Details:

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 256 | Drug/Gene embedding dimension |
| num_heads | 8 | Cross-attention head count |
| batch_size | 32 | Batch size |
| lr | 1e-3 | Learning rate |
| lambda_con | 0.1 | Contrastive loss weight |
| lambda_gate_l1 | 0.01 | Gating L1 regularization |
| lambda_gate_smooth | 0.01 | Gating smoothness regularization |

---

## Results Analysis

Training Curves:

```python
import json
import matplotlib.pyplot as plt

with open('outputs_final/training_log.json') as f:
    log = json.load(f)

epochs = range(1, len(log['val_pcc']) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, log['train_pcc'], label='Train PCC')
plt.plot(epochs, log['val_pcc'], label='Val PCC')
plt.xlabel('Epoch')
plt.ylabel('PCC')
plt.legend()
plt.title('Model Performance')

plt.subplot(1, 2, 2)
plt.plot(epochs, log['train_loss'], label='Train Loss')
plt.plot(epochs, log['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.tight_layout()
plt.savefig('outputs_final/training_curves.png')
```

---

## File Structure

```
GS-ChemDRP/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Dependencies
├── .gitignore
│
├── train.py                               # Main training script
├── dataset.py                             # GDSC dataset loader
├── gs_chemberta_model.py                  # Model definition
│
├── prepare_gene_sets.py                   # Gene set preparation
├── create_drug_pathway_mapping.py         # Drug-pathway mapping
│
├── figures/                               # Figures for README
│   ├── Figure 1.png                       # Model overview
│   ├── Figure 2.png                       # Detailed architecture
│   └── Figure 3.png                       # Contrastive learning framework
│
├── data/
│   ├── sample/                            # Sample data
│   │   ├── sample_train.csv
│   │   ├── sample_valid.csv
│   │   └── sample_test.csv
│   └── gdsc/                              # Download needed
│       ├── gdsc_train_landmark_baseline_clean.csv
│       ├── gdsc_valid_landmark_baseline_clean.csv
│       ├── gdsc_test_landmark_baseline_clean.csv
│       └── screened_compounds_rel_8_4.csv
│
├── gene_sets/                             # Hallmark gene sets
│   ├── hallmark_949.json                  # 48 pathways
│   └── go_clusters_949.json               # GO terms
│
└── outputs_final/                         # Training results
    ├── best_model.pth
    ├── results.json
    └── training_log.json
```

---

## Paper Information

Title: Anti-Cancer Drug Response Prediction via Functional Gene Set Cross-Attention and Supervised Contrastive Learning

Authors: Jongung Song (Chonnam National University)

Corresponding Author: Sunyong Yoo (syyoo@jnu.ac.kr)

Journal: Journal of Digital Contents Society

Status: Under Review

---

## Reference Papers

- ChemBERTa: Chithrananda et al., 2020
- Cross-Attention: Vaswani et al., 2017 (Transformer)
- Contrastive Learning: Chen et al., 2020 (SimCLR)
- Drug Response Prediction: 
  - Daemen et al., 2018 (DeepCDR)
  - Li et al., 2021 (CSG2A)

---

## FAQ

Q: Why is sample data performance low?
A: Sample data is for demonstration. Full GDSC dataset (949 cell lines) achieves PCC 0.922.

Q: Do I need to train the model?
A: Preprocessed checkpoints will be provided (coming soon).

Q: Where to download GDSC data?
A: https://www.cancerrxgene.org/ or request preprocessed version from authors.

Q: Can I use other datasets?
A: Yes! CCLE, GDSC_v2 and similar drug response datasets are supported.

---

## Contributing and Contact

- Bug Report: GitHub Issues
- Feature Request: GitHub Issues
- Pull Requests: Welcome!
- Questions: Discussions or Email

For more information, please contact the author via GitHub Issues.

---

## License

MIT License - Free to use, modify, and distribute.

Copyright (c) 2026 Jongung Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

---

## Acknowledgments

- GDSC team for drug sensitivity screening data
- HuggingFace for ChemBERTa pretrained model
- PyTorch Geometric for GNN utilities

---

Last Updated: 2026-01-08
Version: 1.0.0