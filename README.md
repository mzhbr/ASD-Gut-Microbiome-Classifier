# 🦠 ASD Gut Microbiome Classifier

A machine learning pipeline that classifies Autism Spectrum Disorder (ASD)
from gut microbiome composition — exploring the gut-brain axis through 
computational biology.

## The Problem

ASD diagnosis today relies entirely on behavioral observation:
- Subjective and time-consuming
- Often delayed until age 3-5
- No biological test exists

Recent research shows gut microbiome composition differs significantly 
between ASD and neurotypical individuals. The gut-brain axis — the 
bidirectional communication between gut bacteria and the brain — may 
play a role in ASD development.

This project asks: **can we classify ASD from gut bacteria alone?**

## Answer: 77% Cross-Validated Accuracy

Using Random Forest on 16S rRNA OTU data from 60 subjects 
(30 ASD, 30 Control), the model achieved 77% ± 6% accuracy 
across 5-fold cross-validation.

## Pipeline

Raw OTU Table (5619 bacteria × 60 samples)
↓
Transpose → (60 samples × 5619 bacteria)
↓
Relative Abundance Normalization
↓
Variance-based Feature Selection (top 100 bacteria)
↓
Random Forest + Stratified 5-Fold Cross-Validation
↓
Feature Importance Analysis

## Key Biological Finding

The model identified clinically meaningful bacterial signatures:

| Bacteria | Known ASD Association |
|----------|----------------------|
| Fusobacterium mortiferum | Elevated in ASD gut studies |
| Sutterella spp. | One of the most replicated ASD microbiome findings |
| Prevotella spp. | Consistently reduced in ASD children |
| Clostridium spp. | Linked to gut-brain axis and behavior |

The model learned these patterns from data — without being told 
which bacteria matter.

## Visualizations

### Top 20 Most Important Bacteria
![Bacteria Importance](bacteria_importance.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix_asd.png)

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas | Data loading and transposition |
| NumPy | Relative abundance normalization |
| scikit-learn | Random Forest, cross-validation, feature selection |
| matplotlib / seaborn | Visualization |

## How to Run

```bash
git clone https://github.com/mzhbr/asd-microbiome-classifier
cd asd-microbiome-classifier
pip install -r requirements.txt
python main.py
```

## Dataset

Human Gut Microbiome with ASD — available on Kaggle.  
60 subjects (30 ASD, 30 neurotypical controls).  
16S rRNA gene sequencing, OTU-level resolution.

## Why This Matters

This project is a miniature version of my MSc thesis, which applies 
similar ML pipelines to gut microbiome data across ASD, ADHD, and 
GI disorders. The goal: find shared and disease-specific microbial 
signatures that could support earlier, more objective diagnosis.

---
*MSc Neuroscience, Bahçeşehir University |
Computational Biology & Machine Learning*
