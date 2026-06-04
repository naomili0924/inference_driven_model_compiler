from transformers import GPT2Tokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForCausalLM

ckpt = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
inference_kwargs = dict(encoded_input)
module_fixed_axis_fields = {
	"transformer": ["n_ctx", "n_embd"]
}
model = OnTheFlyORTModelForCausalLM.from_pretrained(
	ckpt,
	inference_kwargs=inference_kwargs,
	export_by_inference=True,
    export=True,
	module_fixed_axis_fields=module_fixed_axis_fields,
	skip_random_generation=False,
)
output_ids = model.generate(**encoded_input)
text_output = tokenizer.decode(output_ids[0])
print(text_output)
