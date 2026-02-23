# LLM Model/Dataset Evaluation Framework

### **Table of Contents**
- [Rundown](#rundown)
- [Setup](#setup)
- [TODOs](#todos)

### Rundown

This project combines the [qwen](https://huggingface.co/Qwen) family of models against computation intensive datasets like [cais/mmlu/elementary_mathematics](https://huggingface.co/datasets/cais/mmlu/viewer/elementary_mathematics) in a generlized framework that allows for further customization.

### Setup

It is highly recommended to run this project in a custom environment with `pip3` installed.

Simply give exection permissions and run the script: `./run.sh`.

### TODOs

- read things in from a config file
- automate running multiple models at different max tokens
