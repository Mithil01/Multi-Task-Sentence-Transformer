import torch

def predict_single_sentence(model, text: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    #AG News classes
    class_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
    
    #NER labels
    ner_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
    
    model.eval()
    model = model.to(device)
    
    #Tokenize
    encoded = model.tokenizer(
        text, padding='max_length', truncation=True, 
        max_length=128, return_tensors='pt')
    
    #Prepare inputs
    batch = {
        'classification_input_ids': encoded['input_ids'].to(device),
        'classification_attention_mask': encoded['attention_mask'].to(device),
        'ner_input_ids': encoded['input_ids'].to(device),
        'ner_attention_mask': encoded['attention_mask'].to(device)}
    
    with torch.no_grad():
        outputs = model(batch)
        
        # Get classification prediction
        class_logits = outputs['classification_logits']
        class_probs = torch.softmax(class_logits, dim=1)
        class_pred = torch.argmax(class_logits, dim=1)
        predicted_class = class_map[class_pred.item()]
        class_confidence = class_probs[0][class_pred].item()
        
        # Get NER predictions
        ner_logits = outputs['ner_logits']
        ner_probs = torch.softmax(ner_logits, dim=-1)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0]
        
        # Get tokens and align with NER predictions
        tokens = model.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Format entities
        entities = []
        current_entity = []
        current_label = None
        
        for token, pred in zip(tokens, ner_preds):
            label = ner_map[pred.item()]
            if label.startswith('B-'):
                if current_entity:
                    entities.append((' '.join(current_entity), current_label[2:]))
                current_entity = [token]
                current_label = label
            elif label.startswith('I-') and current_label is not None:
                current_entity.append(token)
            elif label == 'O':
                if current_entity:
                    entities.append((' '.join(current_entity), current_label[2:]))
                current_entity = []
                current_label = None
        
        if current_entity:
            entities.append((' '.join(current_entity), current_label[2:]))
        
        #Clean up entity text(remove special tokens)
        cleaned_entities = []
        for entity_text, entity_type in entities:
            cleaned_text = entity_text.replace('#', '').replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            if cleaned_text:
                cleaned_entities.append((cleaned_text, entity_type))
        
    return {
        'text': text,
        'classification': predicted_class,
        'classification_confidence': f"{class_confidence:.2%}",
        'entities': cleaned_entities}


def test_samples():
    #Loading model
    model = torch.load('src/model_path/MultiTaskTrainedModel.path') 

    #Test samples
    test_samples = [
        #Tech/Business samples
        "Google and Microsoft announced their latest AI models yesterday in Silicon Valley",
        
        #World news samples
        "The United Nations Security Council met in New York to discuss global security",
        
        #Sports samples
        "Liverpool defeated Manchester City 2-1 at Anfield in Premier League",
        "Roger Federer announced his retirement from professional tennis",
        
        #Science samples
        "SpaceX successfully launched Starship from their Texas facility",
        "Researchers at MIT developed a new quantum computing breakthrough" ]
    
    print("\n=== Model Predictions ===\n")
    
    for text in test_samples:
        result = predict_single_sentence(model, text)
        
        print(f"Text: {result['text']}")
        print(f"Classification: {result['classification']} (Confidence: {result['classification_confidence']})")
        print("Entities: ", end=" ")
        for entity, entity_type in result['entities']:
            print(f"{entity}({entity_type})", end=" | ")
        print("\n" + "-"*50+ "\n")


if __name__ == "__main__":
    test_samples()