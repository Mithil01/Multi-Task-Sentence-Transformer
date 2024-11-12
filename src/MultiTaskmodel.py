import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from tqdm import tqdm


class MultitaskSentenceTransformer(nn.Module):
    def __init__(
        self,
        model_name='bert-base-uncased',
        pooling_strategy = 'mean',
        num_classes=4,  #AG News has 4 classes
        num_ner_labels=9, #CoNLL2003 NER tags
        max_length=128,
        dropout=0.1):
      
        super().__init__()
        #Shared base transformer(bert-uncased in this case)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.basetransformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        hidden_size = self.basetransformer.config.hidden_size

        #Sentence classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes))

        #Named Entity Recognition head
        self.ner = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_ner_labels))
        
    def apply_pooling(self, hidden_states, attention_masks=None):
        """Apply pooling strategy to transformer outputs"""
        if self.pooling_strategy == 'cls':
            # Use CLS token
            return hidden_states[:, 0]
            
        elif self.pooling_strategy == 'mean':
            # Mean pooling
            if attention_masks is not None:
                # Consider padding tokens
                mask_expanded = attention_masks.unsqueeze(-1)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(hidden_states, dim=1)
            
        elif self.pooling_strategy == 'max':
            # Max pooling
            if attention_masks is not None:
                hidden_states[attention_masks == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
    def forward(self, batch):
        """Classification forward pass function"""

        #Classification forward pass
        classification_outputs = self.basetransformer(
            input_ids=batch['classification_input_ids'],
            attention_mask=batch['classification_attention_mask']
        )
        # Pool classification outputs
        pooled_output = self.apply_pooling(
            hidden_states=classification_outputs[0],
            attention_masks=batch['classification_attention_mask'])

        #NER forward pass
        ner_outputs = self.basetransformer(
            input_ids=batch['ner_input_ids'],
            attention_mask=batch['ner_attention_mask'])

        #Get outputs
        classification_logits = self.classifier(pooled_output)  #Out correponding to 'CLS' token(pooled_output)
        ner_logits = self.ner(ner_outputs[0])  #All tokens

        return {
            'classification_logits': classification_logits,
            'ner_logits': ner_logits}







            