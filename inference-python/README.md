# Pithy - Python Inference

## Summarisation using Facebook's Bart Large CNN model

### Installation

- Install Tensorflow for your system
    - Careful with this as some systems have specific installs (CUDA, Apple, AWS)
- Install the dependencies the script uses `./inference.py`

### Perform Summarisation

1. Create a `.txt` file which contains the text you wish to summarise
2. Run the command `time python ./inference.py <./path-to-text.txt>`
    - Substituting `<./path-to-text.txt>` for the path to the `.txt` you wish to summarise

