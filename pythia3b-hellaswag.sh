#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

while read r
do
    lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-2.8b,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.8 \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size auto \
    --log_samples \
    --output_path results_pythia3b
done <pythia-revisions.txt