import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union
import numpy as np
from datasets import load_dataset

class RealMultiTaskDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length

        #Load AG News for Sentence classification
        print("Loading AG News dataset...")
        ag_news = load_dataset('ag_news', split=split)
        self.texts, self.classification_labels = ag_news['text'], ag_news['label']

        #Load CoNLL2003 for Named Entity Recognition
        print("Loading CoNLL2003 dataset...")
        conll = load_dataset('conll2003', split=split)
        self.ner_texts, self.ner_labels = conll['tokens'], conll['ner_tags']

        #Create label mappings for NER
        self.ner_label_map = {
            'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6,'B-MISC': 7, 'I-MISC': 8}

        #Matching both datasets length
        min_length = min(len(self.texts), len(self.ner_texts))
        min_length = 100
        self.texts, self.ner_texts = self.texts[:min_length], self.ner_texts[:min_length]
        self.classification_labels, self.ner_labels  = self.classification_labels[:min_length], self.ner_labels[:min_length]

        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.texts)

    def convert_ner_labels(self, ner_tags):
        return [self.ner_label_map.get(tag, 0) for tag in ner_tags]

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'classification_label': self.classification_labels[idx],
            'ner_text': ' '.join(self.ner_texts[idx]),
            'ner_label': self.ner_labels[idx]}


class RealMultiTaskCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        classification_labels = [item['classification_label'] for item in batch]
        ner_texts = [item['ner_text'] for item in batch]
        ner_labels = [item['ner_label'] for item in batch]

        # Tokenize classification texts
        classification_encodings = self.tokenizer(texts,
            padding='max_length', truncation=True,
            max_length=self.max_length,return_tensors='pt')

        # Tokenize NER texts and align labels
        ner_encodings = self.tokenizer(ner_texts, padding='max_length', truncation=True, 
                        max_length=self.max_length,return_tensors='pt')

        # Prepare NER labels
        padded_ner_labels = self.pad_and_align_ner_labels(ner_labels, ner_encodings['attention_mask'])

        return {
            'classification_input_ids': classification_encodings['input_ids'],
            'classification_attention_mask': classification_encodings['attention_mask'],
            'ner_input_ids': ner_encodings['input_ids'],
            'ner_attention_mask': ner_encodings['attention_mask'],
            'classification_labels': torch.tensor(classification_labels),
            'ner_labels': torch.tensor(padded_ner_labels)}

    def pad_and_align_ner_labels(self, ner_labels, attention_mask):
        batch_size, seq_length = attention_mask.shape
        padded_labels = np.full((batch_size, seq_length), -1000)

        for i, labels in enumerate(ner_labels):
            if len(labels) > seq_length:
                labels = labels[:seq_length]
            padded_labels[i, 1:len(labels)+1] = labels  #Offset by 1 for 'CLS' token
            padded_labels[i, 0] = -1000  #'CLS' token
            padded_labels[i, len(labels)+1:] = -1000  # Padding tokens

        return padded_labels