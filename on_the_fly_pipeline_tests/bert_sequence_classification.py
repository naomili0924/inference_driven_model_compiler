from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForSequenceClassification

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "I love using ONNX Runtime for inference."
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForSequenceClassification.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
print("logits shape:", outputs.logits.shape)
print("logits:", outputs.logits)
