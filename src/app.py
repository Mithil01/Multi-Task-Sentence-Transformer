import streamlit as st
import os
import torch
import pandas as pd
from MultiTaskmodel import MultitaskSentenceTransformer

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
        'entities': cleaned_entities
    }

def main():
    st.set_page_config(
        page_title="MultiTask Text Analyzer",
        page_icon="📝",
        layout="wide")
    
    st.title("📝 MultiTask Text Analyzer")
    st.markdown("""
    This app analyzes text for:
    1. Text Classification (News Category)
    2. Named Entity Recognition (NER)
    """)
    
    # Load model
    @st.cache_resource
    def load_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Trained model on Colab T4 GPU, so locally just loading trained weights.
        model = MultitaskSentenceTransformer()
        # Load weights
        model.load_state_dict(torch.load('model_path/ST_Model_dict.path', map_location='cpu'))
        model.eval()
        return model
    
    try:
        model = load_model()
        print('done')
        
        # Input text area
        text = st.text_area(
            "Enter your text:",
            height=100,
            placeholder="Enter news text here..."
        )
        
        if st.button("Analyze"):
            if text:
                with st.spinner("Analyzing text..."):
                    results = predict_single_sentence(model, text)
                
                col1, col2 = st.columns(2)
                
                # Classification results
                with col1:
                    st.subheader("📊 Classification Results")
                    st.markdown(f"""
                    **Category**: {results['classification']}  
                    **Confidence**: {results['classification_confidence']}
                    """)
                
                # NER results
                with col2:
                    st.subheader("🏷️ Named Entities")
                    if results['entities']:
                        # Create DataFrame for entities
                        entities_df = pd.DataFrame(results['entities'], columns=['Text', 'Type'])
                        
                        # Color code different entity types
                        def highlight_entities(s):
                            colors = {
                                        'PER': 'background-color: #FF7F7F',    # Brighter red
                                        'ORG': 'background-color: #90EE90',    # Brighter green
                                        'LOC': 'background-color: #87CEFA',    # Brighter blue
                                        'MISC': 'background-color: #FFD700'    # Bright gold
                                    }
                            return [colors.get(s['Type'], '') for _ in range(len(s))]
                        
                        st.dataframe(
                            entities_df.style.apply(highlight_entities, axis=1),
                            hide_index=True
                        )
                    else:
                        st.info("No named entities found.")
                    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file is in the correct location.")

if __name__ == '__main__':
    main()
