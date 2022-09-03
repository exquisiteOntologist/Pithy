# Pithy
An abridged becoming for lengthy and burdensome texts


## Create optimized models

We want ONNX models &mdash; optimized to run on the [ONNX Runtime](https://onnxruntime.ai/) for ultra-fast cross-system inference.

### For summarization

[Bart](https://huggingface.co/facebook/bart-large-cnn) is a popular and reliable model for summarization. 

1. Open the python script and for each import `pip install <dependency>`
2. Generate the model with the script:

```bash
python create-bart-onnx.py
```
