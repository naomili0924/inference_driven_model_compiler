import torch
from transformers import AutoFeatureExtractor
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction

ckpt = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(ckpt)

# Dummy RGB image: (1, 3, 224, 224)
dummy_image = torch.randint(0, 256, (224, 224, 3)).numpy()
encoded_input = extractor(images=dummy_image, return_tensors="pt")

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
