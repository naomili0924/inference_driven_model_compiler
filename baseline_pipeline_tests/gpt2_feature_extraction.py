import torch

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

MODEL_ID = "gpt2"

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    export=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = "The future of artificial intelligence is"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

position_ids = (
    inputs["attention_mask"].cumsum(-1) - 1
).clamp(min=0)

inputs["position_ids"] = position_ids

outputs = model(**inputs)

# shape:
# [batch_size, seq_len, hidden_size]
hidden_states = outputs.last_hidden_state

print("Hidden state shape:")
print(hidden_states.shape)

# sentence embedding by mean pooling
attention_mask = inputs["attention_mask"]

masked_hidden = hidden_states * attention_mask.unsqueeze(-1)

sentence_embedding = (
    masked_hidden.sum(dim=1)
    / attention_mask.sum(dim=1, keepdim=True)
)

print("\nSentence embedding shape:")
print(sentence_embedding.shape)

print("\nFirst 10 values:")
print(sentence_embedding[0, :10])