#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

lm_eval --model vllm \
--model_args pretrained=allenai/OLMo-7B-0724-hf,revision=$1,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
--tasks $2 \
--num_fewshot $3 \
--batch_size auto \
--log_samples \
--output_path results