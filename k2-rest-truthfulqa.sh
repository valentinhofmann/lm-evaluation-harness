#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

while read r
do
    lm_eval --model vllm \
    --model_args pretrained=LLM360/K2,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks truthfulqa_mc2 \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path results_k2
done <k2-revisions-truthfulqa.txt