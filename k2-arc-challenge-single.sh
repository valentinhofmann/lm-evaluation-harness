#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

lm_eval --model vllm \
--model_args pretrained=LLM360/K2,revision=ckpt_087,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
--tasks arc_challenge \
--num_fewshot 25 \
--batch_size auto \
--log_samples \
--output_path results_k2