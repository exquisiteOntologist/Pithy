# Pithy Inference
Pithy inference programs.

## Dependencies

1. [ONNX Runtime](https://onnxruntime.ai/docs/install/) must be installed on the target machine
    - On OSX I:
        1. Installed correct version of `onnxruntime` with homebrew package manager
        2. Then copied the dylib to where the Rust ONNX lib expects:
            - `cp /opt/homebrew/Cellar/onnxruntime/1.12.1/lib/libonnxruntime.dylib /usr/local/lib/libonnxruntime.1.12.1.dylib`

## Usage

Selectively loading a model:

1. Get the ONNX model's directory with its supporting files ready
    - `model.onnx`
    - `config.json`
    - `vocab.json`
    - `merges.txt`
2. Have a text file to summarize handy

```bash
./pithy ./model-directory sample-text.txt
```

## TODO

- [x] Tokenization
- [] Inference
    - [x] Pass input text into tokenizer and get input IDs from it
    - [x] Pass input IDs through encoder model
    - [] Decode inputs to decoded IDs
    - [] Decode decoded IDs with tokenizer to summarized strings outputs
- [x] Model argument change to model directory argument for where model and related files reside
