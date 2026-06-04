from setuptools import setup, find_packages

setup(
    name="inference_driven_model_compiler",
    version="0.1.0",
    description="On-the-fly model compilation and inference pipeline",
    author="naomili0924",

    # 核心：自动找你的 package
    packages=find_packages(
        include=[
            "inference_driven_model_compiler",
            "inference_driven_model_compiler.*",
            "optimum",
            "optimum.*",
        ]
    ),

    python_requires=">=3.8",

    install_requires=[
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "numpy",
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },

    include_package_data=True,
    zip_safe=False,
)