#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

lm_eval --model vllm \
--model_args pretrained=allenai/OLMo-2-1124-7B,revision=$1,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.6 \
--tasks $2 \
--num_fewshot $3 \
--batch_size auto \
--log_samples \
--output_path results_olmo2