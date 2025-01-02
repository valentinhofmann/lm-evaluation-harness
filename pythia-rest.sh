#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

while read r
do
    lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-6.9b,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size auto \
    --log_samples \
    --output_path results_pythia

    lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-6.9b,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --log_samples \
    --output_path results_pythia

    lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-6.9b,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks truthfulqa_mc2 \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path results_pythia

    lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-6.9b,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks winogrande \
    --num_fewshot 5 \
    --batch_size auto \
    --log_samples \
    --output_path results_pythia
done <pythia-revisions.txt