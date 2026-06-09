#!/bin/bash

# this script starts the server for the LLaMA model
# you need to run this before running the agentic and the llm code
# make sure to change the path to the model and the port if needed

python -m llama_cpp.server \
  --model ../Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  --n_gpu_layers -1 \
  --flash_attn true \
  --port 8000