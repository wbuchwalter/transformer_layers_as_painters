#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

start_layer=7 # start layer index, 14 for Llama-2-7b
model=google/gemma-3-4b-it #meta-llama/Llama-2-7b-hf #mistralai/Mistral-7B-v0.1
method='baseline' # middle_repeat, skip, reverse, baseline, random, loop_parallel
repeat_time_or_seed=3 # repeat time if method is loop_parallel, seed if method is random
tasks='gpqa_diamond_zeroshot' # arc_challenge,hellaswag,winogrande,gsm8k

lm_eval --model hf \
    --model_args pretrained=$model \
    --tasks $tasks \
    --batch_size 8 \
    --method $method \
    --start_layer $start_layer \
    --repeat_time_or_seed $repeat_time_or_seed

# google/gemma-3-4b-it
# https://ai.google.dev/gemma/docs/core/model_card_3?hl=es-419
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.2677|±  |0.0315|
# |                     |       |none  |     0|acc_norm|↑  |0.2677|±  |0.0315|


# with layer 8 repeated 2 times
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.2424|±  |0.0305|
# |                     |       |none  |     0|acc_norm|↑  |0.2424|±  |0.0305|