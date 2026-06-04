import numpy as np
from transformers import WhisperProcessor
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForFeatureExtraction

ckpt = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(ckpt)

# 1 second of silence at 16 kHz
audio = np.zeros(16000, dtype=np.float32)
encoded_input = processor(audio, sampling_rate=16000, return_tensors="pt")

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
