# Multi-Task-Sentence-Transformer

## Overview
This project implements a multi-task transformer that performs:

- Text Classification: Using AG News dataset (4 classes).
- Named Entity Recognition: Using CoNLL2003 dataset (9 NER tags).

## Directory Structure
![Images Alt text](https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/dir_struct.png)

## Datasets
 1. AG News (Classification)

 - 4 classes: World, Sports, Business, Technology </p>
- Training samples: ~120,000 
- Test samples: ~7,600

2. CoNLL2003 (NER)

- 9 labels (B-PER, I-PER, B-ORG, I-ORG, etc.)
- Training sentences: ~14,000
- Test sentences: ~3,453

## Sentence Transformer
<p align="left">
  <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/model.png" width="400">
</p>
   - **Model Configuration: **
       - ‘Bert-base-uncased’ as backbone.
       - Configurable embedding size.

   - **Pooling strategy: **
     - Implemented three different pooling strategies
       1. Mean: averages pooling across all token embeddings.
       2. Cls: Using [Cls] token out from pooler_output. 
       3. Max: max pooling across token embeddings.

   - **Embeddings Normalization:**
     - Optional embeddings normalization.
     - Used L1-norm.


## Model Architecture
<p align="left">
  <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/model.png" width="400">
</p>
- Base Model: BERT-base-uncased

- Tasks:
  - Classification Head (4 classes)
  - NER Head (9 labels)


- Features:

  - Shared transformer backbone
  - Layer-wise learning rates
  - Task-specific heads

## 2. Model Training
# Run training
``` python
python src/training_application.py
```

## 2. Evaluate on test samples
# Run training
``` python
python src/test.py
```

## 3. Streamlit application
# Run training
``` python
streamlit run src/app.py
```
