from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L12-v2')
#model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L12-v2')
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
    'sentence-transformers/paraphrase-MiniLM-L12-v2',
    from_transformers=True,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "intermediate_size","type_vocab_size", "vocab_size"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("outputs: ", outputs)
#print("last_hidden_state shape:", outputs.last_hidden_state.shape)
