import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling_strategy = 'mean', normalize = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize

    def forward(self, input_ids, attn_masks):
        # pass inputs through bert-uncased model 
        last_hidden_state, pooler_output = self.base_transformer(input_ids, attention_mask = attn_masks, return_dict=True)

        # Apply different pooling strategy
        if self.pooling_strategy == 'mean':
            last_hidden_state = torch.mean(last_hidden_state, dim=1)
        elif self.pooling_strategy == 'cls':
            last_hidden_state = pooler_output
        elif self.pooling_strategy == 'max':
            last_hidden_state = torch.max(last_hidden_state, dim=1)
        else:
            raise Exception('Unknown pooling strategy')
        
        # Normalize final embeddings(L1 Norm)
        if self.normalize:
           norm_last_hidden_state = F.normalize(last_hidden_state, p=1, dim=1)
           return norm_last_hidden_state
        return last_hidden_state

    def encode(self, sentences, batch_size = 2):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            # Tokenize the sentences batch
            encoded_input = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
            encoded_input_ids = encoded_input['input_ids']
            encoded_attn_masks = encoded_input['attention_mask']
            # Get embeddings w/o gradient computation
            with torch.no_grad():
                model_output = self.base_transformer(encoded_input_ids, encoded_attn_masks)
            all_embeddings.append(model_output[1])
        # Concatenate batch embeddings
        all_embeddings = torch.cat(all_embeddings, dim = 0)
        return all_embeddings


def test_model():
    #initialize model
    model = SentenceTransformer(
        model_name='bert-base-uncased',
        pooling_strategy='max',
        normalize = True
    )

    #Sample sentences
    sentences = [
        "The cat sat on the mat.",
        "The cat sat on the mat and watched the dogs playing in the park.",
        'I love Manchester United',
        'I love chelsea',
        'I love Manchester United']

    #Get embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    embeddings = model.encode(sentences)

    #Print shapes 
    print(f"Embedding shape: {embeddings.shape}")

    #calculate cosine similarities between sentences
    similarities = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)

    print("\n Cosine Similarities:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"\nSentence {i+1} vs Sentence {j+1}")
            print(f"'{sentences[i]}' vs '{sentences[j]}'")
            print()
            print(f"Similarity: {similarities[i][j].item():.4f}")

if __name__ == "__main__":
   test_model()