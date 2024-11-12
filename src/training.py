import torch
import torch.nn as nn
from tqdm import tqdm


class MultitaskTrainer:
    def __init__(self, model, optimizer, classification_weight = 1.0, ner_weight = 1.0, device=None):
        self.model = model
        self.optimizer = optimizer
        self.classification_weight = classification_weight
        self.ner_weight = ner_weight
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.classification_criterion = nn.CrossEntropyLoss()
        self.ner_criterion = nn.CrossEntropyLoss(ignore_index = -1000)

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0
        classification_correct = 0
        ner_correct = 0
        total_samples = 0
        total_ner_tokens = 0

        #For Mixed-precision training
        scaler = torch.amp.GradScaler('cuda')

        for tr_batch in tqdm(train_loader, desc='Training'):

            #Auto-data type casting during forward pass - Mixed Precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                #Move batch to device  
                tr_batch = {k: v.to(self.device) for k,v in tr_batch.items()}

                #Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(tr_batch)
            
                #Classification loss
                classification_loss = self.classification_criterion(
                      outputs['classification_logits'],
                      tr_batch['classification_labels'])

                #NER loss
                ner_logits = outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1))
                ner_labels = tr_batch['ner_labels'].view(-1)
                ner_loss = self.ner_criterion(ner_logits, ner_labels)
            
                #Combined loss
                loss = (self.classification_weight * classification_loss + self.ner_weight * ner_loss)
  
            #Backward pass
            #Loss scaling and gradient unscaling during backward pass - Mixed-Precision
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            #Calculate metrics
            classification_preds = torch.argmax(outputs['classification_logits'], dim=1)
            classification_correct += (classification_preds == tr_batch['classification_labels']).sum().item()

            ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
            valid_tokens = (tr_batch['ner_labels'] != -1000)
            ner_correct += ((ner_preds == tr_batch['ner_labels']) & valid_tokens).sum().item()

            total_samples += tr_batch['classification_labels'].size(0)
            total_ner_tokens += valid_tokens.sum().item()
            total_loss += loss.item()

        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_classification_accuracy': classification_correct/total_samples,
            'train_ner_accuracy': ner_correct/ total_ner_tokens}

        if val_loader:
            val_metrics = self.evaluate(val_loader)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        classification_correct,ner_correct = 0, 0
        total_samples, total_ner_tokens  = 0, 0

        for batch in tqdm(dataloader, desc='Validation'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)

            #Classification metrics
            classification_preds = torch.argmax(outputs['classification_logits'], dim=1)
            classification_correct += (classification_preds == batch['classification_labels']).sum().item()

            #NER metrics
            ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
            valid_tokens = (batch['ner_labels'] != -1000)
            ner_correct += ((ner_preds == batch['ner_labels']) & valid_tokens).sum().item()

            total_samples += batch['classification_labels'].size(0)
            total_ner_tokens += valid_tokens.sum().item()

        return {
            'classification_accuracy': classification_correct / total_samples,
            'ner_accuracy': ner_correct / total_ner_tokens}