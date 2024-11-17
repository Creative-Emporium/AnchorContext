export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'

# export DEBUGPY=1
# export DYNAMO_CACHE_SIZE_LIMIT=1024


# accelerate launch \
# --config_file  accelerate_configs/single_node_4gpu.yaml \
# train.py \
# --wandb AnchorContext \
# --output-dir ./output/test \
# --batch-size 1 \
# --gradient-accumulate-every 8 \
# --max-train-steps 4000  \
# --saving_interval 500 \
# --min_steps_for_save 3000 \
# --save_per_hours 3 \
# --dataset ./data/raw_data/slimpajama_packed_256001_5b_small.jsonl \
# --data-format tokenized \
# --attn_engine 'flash' \
# --model_name_or_path meta-llama/Llama-2-7b-hf \
# --model_max_position_embeddings 4096 \
# --rope_scaling_type default \
# --rope_scaling_factor 16 \
# --seq-length 65536 \
# --rope-theta 100000 \
# --parallel_mode ulysses_attn

# # the *flex attn* version:
accelerate launch \
--config_file  accelerate_configs/single_node_4gpu.yaml \
train.py \
--wandb AnchorContext \
--output-dir ./output/test \
--batch-size 1 \
--gradient-accumulate-every 8 \
--max-train-steps 4000  \
--saving_interval 500 \
--min_steps_for_save 3000 \
--save_per_hours 3 \
--dataset ./data/raw_data/slimpajama_packed_256001_5b_small.jsonl \
--data-format tokenized \
--attn_engine 'flex' \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--model_max_position_embeddings 4096 \
--rope_scaling_type default \
--rope_scaling_factor 16 \
--seq-length 65536 \
--rope-theta 100000 \
--parallel_mode ulysses_attn