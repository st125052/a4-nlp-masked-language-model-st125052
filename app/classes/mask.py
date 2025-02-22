import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from classes.all_classes import *

def get_prerequisites():
    with open("./helper/classifier_head.pkl", 'rb') as f:
        classifier_head = torch.load(f, weights_only=False)

    with open("./helper/tokenizer.pkl", 'rb') as f:
        tokenizer = torch.load(f, weights_only=False)

    with open("./models/BERT/model.pkl", 'rb') as f:
        bert = torch.load(f, weights_only=False)

    return bert, classifier_head, tokenizer

def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def get_torch_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def predict_nli_class_with_similarity(model, classifier_head, premise, hypothesis, tokenizer, device):
    # Tokenize and convert to input IDs and attention masks
    inputs_a = tokenizer(premise, return_tensors='pt', truncation=True, padding=True).to(device)
    inputs_b = tokenizer(hypothesis, return_tensors='pt', truncation=True, padding=True).to(device)
    
    inputs_ids_a = inputs_a['input_ids']
    attention_a = inputs_a['attention_mask']
    inputs_ids_b = inputs_b['input_ids']
    attention_b = inputs_b['attention_mask']
    
    segment_ids_a = torch.zeros_like(inputs_ids_a).to(device)
    segment_ids_b = torch.zeros_like(inputs_ids_b).to(device)
    
    # Get BERT embeddings
    with torch.no_grad():
        u_last_hidden_state = model.get_last_hidden_state(inputs_ids_a, segment_ids_a).to(device)
        v_last_hidden_state = model.get_last_hidden_state(inputs_ids_b, segment_ids_b).to(device)
    
    # Mean-pooling
    u_mean_pool = mean_pool(u_last_hidden_state, attention_a).detach().cpu().numpy()
    v_mean_pool = mean_pool(v_last_hidden_state, attention_b).detach().cpu().numpy()
    
    # Create the feature vector for classification
    uv_abs = torch.abs(torch.sub(torch.tensor(u_mean_pool).to(device), torch.tensor(v_mean_pool).to(device)))
    x = torch.cat([torch.tensor(u_mean_pool).to(device), torch.tensor(v_mean_pool).to(device), uv_abs], dim=-1)
    
    # Get logits from the classifier head
    logits = classifier_head(x)
    
    # Compute class probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Map probabilities to classes
    class_labels = ['contradiction', 'neutral', 'entailment']
    predicted_class = class_labels[torch.argmax(probs).item()]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(u_mean_pool.reshape(1, -1), v_mean_pool.reshape(1, -1))[0, 0]
    
    return {
        'predicted_class': predicted_class,
        'class_probabilities': probs.detach().cpu().numpy(),
        'cosine_similarity': cosine_sim
    }

def predict_nli_class(premise, hypothesis):
    device = get_torch_device()
    bert, classifier_head, tokenizer = get_prerequisites()
    result = predict_nli_class_with_similarity(bert, classifier_head, premise, hypothesis, tokenizer, device)
    return result