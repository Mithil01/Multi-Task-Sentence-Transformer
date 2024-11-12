# Multi-Task-Sentence-Transformer

## Overview
This project implements a multi-task transformer that performs:

Text Classification: Using AG News dataset (4 classes).
Named Entity Recognition: Using CoNLL2003 dataset (9 NER tags).

## Directory Structure
src/
├── data_connector.py         # Dataset and collator implementations
├── MultiTaskmodel.py         # Multi-task transformer architecture
├── training.py              # Training logic and layer-wise optimization
├── training_application.py  # Training execution and monitoring
├── test.py                 # Model evaluation and testing
├── app.py                  # Streamlit web interface
└── requirements.txt        # Project dependencies

## Datasets
 - AG News (Classification)

<p> 4 classes: World, Sports, Business, Technology </p>
<p> </p>Training samples: ~120,000 </p>
Test samples: ~7,600

CoNLL2003 (NER)

9 labels (B-PER, I-PER, B-ORG, I-ORG, etc.)
Training sentences: ~14,000
Test sentences: ~3,453

## Model Architecture

- Base Model: BERT-base-uncased

- Tasks:

-- Classification Head (4 classes)
-- NER Head (9 labels)


### Features:

-- Shared transformer backbone
-- Layer-wise learning rates
-- Task-specific heads

## 2. Model Training
# Run training
python src/training_application.py
