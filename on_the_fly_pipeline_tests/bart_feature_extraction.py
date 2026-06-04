from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction

ckpt = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "The quick brown fox jumps over the lazy dog."
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["d_model"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
