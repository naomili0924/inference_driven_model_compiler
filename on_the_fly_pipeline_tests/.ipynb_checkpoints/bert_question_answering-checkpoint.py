from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForQuestionAnswering

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
question = "What is the capital of France?"
context = "Paris is the capital of France and a major European city."
encoded_input = tokenizer(question, context, return_tensors="pt")

model = OnTheFlyORTModelForQuestionAnswering.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded_input),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    skip_random_generation=False,
)
outputs = model(**encoded_input)
start = outputs.start_logits.argmax()
end = outputs.end_logits.argmax() + 1
answer = tokenizer.decode(encoded_input["input_ids"][0][start:end])
print("start_logits shape:", outputs.start_logits.shape)
print("end_logits shape  :", outputs.end_logits.shape)
print("Answer:", answer)
