from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForMaskedLM

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "Paris is the [MASK] of France."
encoded_input = tokenizer(text, return_tensors="pt")

model = OnTheFlyORTModelForMaskedLM.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
mask_index = (encoded_input["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_id = outputs.logits[0, mask_index].argmax(dim=-1)
print("logits shape:", outputs.logits.shape)
print("Predicted token:", tokenizer.decode(predicted_id))
