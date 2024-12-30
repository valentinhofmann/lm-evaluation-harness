#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

while read r
do
    lm_eval --model vllm \
    --model_args pretrained=allenai/OLMo-7B-0724-hf,revision=$r,tensor_parallel_size=$gpu_count,dtype=auto,gpu_memory_utilization=0.7 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --log_samples \
    --output_path results_mmlu
done <olmo1-revisions-$1.txt