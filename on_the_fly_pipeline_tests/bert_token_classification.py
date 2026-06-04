from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForTokenClassification

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "Hugging Face is based in New York City."
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForTokenClassification.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("logits shape:", outputs.logits.shape)
tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
print("Tokens:", tokens)
