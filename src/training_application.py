import torch
import torch.nn as nn
from data_connector import RealMultiTaskDataset, RealMultiTaskCollator
from MultiTaskmodel import MultitaskSentenceTransformer
from training import MultitaskTrainer
from torch.utils.data import  DataLoader

def train_and_evaluate(adaptive_lr = False, save_path = 'model_path/MultiTaskTrainedModel.path'):
    #Initialize model 
    model = MultitaskSentenceTransformer()

    #Create datasets
    train_dataset = RealMultiTaskDataset(model.tokenizer, split='train')
    val_dataset = RealMultiTaskDataset(model.tokenizer, split='test')

    #Create train dataloader
    train_loader = DataLoader(train_dataset,
        batch_size = 64,
        shuffle = True,
        collate_fn = RealMultiTaskCollator(model.tokenizer))
    
    #Create validation dataloader
    val_loader = DataLoader(val_dataset,
        batch_size = 64,
        shuffle = False,
        collate_fn = RealMultiTaskCollator(model.tokenizer))

    #Initializing optimizer
    if not adaptive_lr:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        save_path = 'model_path/MultiTaskTrainedModel.path'

        #Initializing trainer
        trainer = MultitaskTrainer(model=model,
        optimizer=optimizer,
        classification_weight=1.0,
        ner_weight=1.0)
    else:
        optimizer = LayerwiseOptimizer(
        model,
        base_lr=2e-5,
        layer_decay=0.85,
        classifier_lr=5e-5,
        ner_lr=5e-5)

        #Initializing trainer
        trainer = MultitaskTrainer(model=model,
        optimizer=optimizer.optimizer,
        classification_weight=1.0,
        ner_weight=1.0)

        save_path = 'model_path/MultiTaskTrained_AdaptiveLR.path'

    #Running training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"\nEpoch : {epoch+1}/{num_epochs}")
        metrics = trainer.train_epoch(train_loader, val_loader)

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # Save model
    torch.save(model, save_path)

class LayerwiseOptimizer:
    def __init__(self, model, base_lr = 2e-4,
        layer_decay = 0.85, classifier_lr = 5e-4,
        ner_lr = 5e-4, weight_decay = 0.01):
      
        self.model = model
        param_groups = list()
        
        #Get number of layers in base transformer
        num_layers = len(model.basetransformer.encoder.layer)
        
        #Embeddings layer(lowest learning rate)
        param_groups.append({
            'params': model.basetransformer.embeddings.parameters(),
            'lr': base_lr * (layer_decay**(num_layers + 1)),
            'name': 'embeddings'})
        
        #Transformer layers(gradually increasing learning rates)
        for i in range(num_layers):
            layer = model.basetransformer.encoder.layer[i]
            lr = base_lr * (layer_decay ** (num_layers - i))
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr,
                'name': f'encoder_layer_{i}'})
        
        #Task-specific heads(highest learning rates)
        param_groups.extend([
            {
                'params': model.classifier.parameters(),
                'lr': classifier_lr,
                'name': 'classifier_head'
            },
            {
                'params': model.ner.parameters(),
                'lr': ner_lr,
                'name': 'ner_head'
            }])
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay = weight_decay)
    
    def get_lr_info(self):
        return {group['name']: group['lr'] for group in self.optimizer.param_groups}

def train_and_evaluate_AdaptiveLearningRate(save_path='TrainedModel/AdaptiveLearning_model.path'):
    #Initialize model and datasets
    model = MultitaskSentenceTransformer()
    
    #Create datasets
    train_dataset = RealMultiTaskDataset(model.tokenizer, split='train')
    val_dataset = RealMultiTaskDataset(model.tokenizer, split='test')
    
    #Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=RealMultiTaskCollator(model.tokenizer))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=RealMultiTaskCollator(model.tokenizer))
    
    #Initialize layerwise optimizer
    optimizer = LayerwiseOptimizer(
        model,
        base_lr=2e-5,
        layer_decay=0.85,
        classifier_lr=5e-5,
        ner_lr=5e-5)
    
    #learning rates for different layers
    lr_info = optimizer.get_lr_info()
    print("\nLayer-wise Learning Rates:")
    for name, lr in lr_info.items():
        print(f"{name}: {lr:.2e}")
    
    #Initialize trainer with layerwise optimizer
    trainer = MultitaskTrainer(
        model=model,
        optimizer=optimizer.optimizer,  #AdamW optimizer inside LayerwiseOptimizer
        classification_weight = 1.0,
        ner_weight = 1.0)
    
    #Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        metrics = trainer.train_epoch(train_loader, val_loader)
        
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    #Save the model
    torch.save(model, save_path)

if __name__ == "__main__":
    #Train & Evaluate model w/o adaptive learning rate
    train_and_evaluate(adaptive_lr = False)
    #Train & Evaluate model With learning rate
    train_and_evaluate(adaptive_lr = True)