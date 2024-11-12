# Multi-Task-Sentence-Transformer

## Overview
This project implements a multi-task transformer that performs:

Text Classification: Using AG News dataset (4 classes)
Named Entity Recognition: Using CoNLL2003 dataset (9 NER tags)

## Directory Structure
src/
├── data_connector.py         # Dataset and collator implementations
├── MultiTaskmodel.py         # Multi-task transformer architecture
├── training.py              # Training logic and layer-wise optimization
├── training_application.py  # Training execution and monitoring
├── test.py                 # Model evaluation and testing
├── app.py                  # Streamlit web interface
└── requirements.txt        # Project dependencies
