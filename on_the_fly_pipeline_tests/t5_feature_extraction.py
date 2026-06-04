from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction

ckpt = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "translate English to French: Hello, how are you?"
# Only encoder inputs needed — encoder-decoder path exports the encoder only
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["d_model", "d_ff"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
