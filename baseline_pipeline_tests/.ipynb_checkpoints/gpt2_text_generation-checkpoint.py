from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

MODEL_ID = "gpt2"

# Export GPT-2 to ONNX on first run
model = ORTModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "The future of artificial intelligence is"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)

text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print(text)