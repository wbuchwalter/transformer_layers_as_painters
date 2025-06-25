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



# gemma-3-4b-it, 1x, 
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.3485|±  |0.0339|
# |                     |       |none  |     0|acc_norm|↑  |0.3485|±  |0.0339|

# finetuned, 1x
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.3434|±  |0.0338|
# |                     |       |none  |     0|acc_norm|↑  |0.3434|±  |0.0338|

# gemma-3-4b-it, 2x 
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.3384|±  |0.0337|
# |                     |       |none  |     0|acc_norm|↑  |0.3384|±  |0.0337|

# finetuned, 2x
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.3333|±  |0.0336|
# |                     |       |none  |     0|acc_norm|↑  |0.3333|±  |0.0336|


# gemma-3-4b-it, 5x on layers 8, 9, 10
# |        Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |gpqa_diamond_zeroshot|      1|none  |     0|acc     |↑  |0.2626|±  |0.0314|
# |                     |       |none  |     0|acc_norm|↑  |0.2626|±  |0.0314|