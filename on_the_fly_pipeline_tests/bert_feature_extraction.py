from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "ONNX Runtime accelerates machine learning inference."
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
