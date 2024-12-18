# Multi-Task-Sentence-Transformer
<p align="left">
      <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/model.png" width="500">
</p>
This project implements a multi-task transformer that performs:

- Text Classification: Using AG News dataset (4 classes).
- Named Entity Recognition: Using CoNLL2003 dataset (9 NER tags).

## Installation
- Clone the repository
  ```bash  
   git clone https://github.com/Mithil01/Multi-Task-Sentence-Transformer.git
   cd Multi-Task-Sentence-Transformer
  ```

- Install dependencies
  ```python
  pip install -r requirements.txt
  ```
## Docker Image
```bash
# Pull the image
docker pull mithilg1/multi-task-sentence-transformer

# Run Streamlit app
docker run -p 8502:8502 mithilg1/multi-task-sentence-transformer
```
## Directory Structure
```bash
SENTENCETRANSFORMER/
├── src/
│   ├── app.py                    # Streamlit interface
│   ├── data_connector.py         # Data processing
│   ├── MultiTaskmodel.py         # Multi-task model
│   ├── SentenceTransformer.py    # Base transformer
│   ├── test.py                   # Testing
│   ├── training_application.py   # Training execution  
│   └── training.py               # Training logic
├── model_path/            # Saved models
├── Dockerfile            # Docker config
└── requirements.txt      # Dependencies
```

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
    <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/sentence_transformer.png" width="400">
   </p>

  - **Model Configuration:**
    - ‘Bert-base-uncased’ as backbone.
    - Configurable embedding size.

  - **Pooling strategy:**
     - Implemented three different pooling strategies
       1. Mean: averages pooling across all token embeddings.
       2. Cls: Using [Cls] token out from pooler_output. 
       3. Max: max pooling across token embeddings.
          
  - **Embeddings Normalization:**
     - Optional embeddings normalization.
     - Used L1-norm.


## Multi-task Sentence Transformer Architecture
   <p align="left">
      <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/model.png" width="400">
   </p>

  - **Shared component - bert-base-uncased:**
    - Bert base transformer remains common.
    - Shared tokenizer.
    - Common hidden size from base transformer.
  - **Task Specific heads:**
    - Classification head: performs hidden_size projection to num_classes.
    - Named Entity Recognition head: performs hidden_size projection to num_ner_labels.

  - **Head Architecture:**
    - Both heads follow same structure i.e. dropout for regularization, layer norm, GELU, task specific final projection.
      
  ### **Mixed Precision Training:**
  - Utilized torch.cuda autocast function and GradScaler() for mixed precision training.
   <p align="left">
    <img src="https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/images/Mixed_precision.png" width="300" height="500">
   </p>
   
  ### Layer-wise learning rate:

  - Embedding layer:
      - Lowest Learning rate.

  - Base Transformer layers:
      - Gradually Increasing Learning rate.

  - Task specific head:
      - Highest Learning rate.
      
## Model Evaluation:
   - **Loss**: Crossentropy
       - **Task weighing**: total_loss = classification_weight * classification_loss+ ner_weight * ner_loss
       
   - **Metric**: Accuracy, Number of epochs = 5
       - Train Classification accuracy: 97.2%
       - Validation Classification accuracy: 92.6%
       - Train NER accuracy: 90.8%
       - Validation NER accuracy: 89.9%
       
## Components:
   -  **Data Processing (`data_connector.py`)**
         - Implements data loading for AG News and CoNLL2003 datasets
         - Custom collator for multi-task learning
           
   -  **Model Architecture (`MultiTaskmodel.py`)**
         - Multi-task transformer implementation
           
   -  **Training Implementation (`training.py`)**
         - Training loop and optimization
         - Layer-wise learning rates
         - Mixed precision training
           
   -  **Training Execution (`training_application.py`)**
         - Configures and runs training
         - Handles model saving and evaluation
        
   -  **Model Testing (`test.py`)**
         - Model evaluation on test datasets
           
   -  **Web Interface (`app.py`)**
         - Streamlit application for model demo
         

## Usage    
- **Model Training**
``` python
python src/training_application.py
```

- **Model Evaluation**
``` python
python src/test.py
```

- **Streamlit application**
``` python
streamlit run src/app.py
```

## Project Tasks Writeup
[Doc](https://github.com/Mithil01/Multi-Task-Sentence-Transformer/blob/main/Sentence_Transformer_tasks_writeup.pdf)
