#!/bin/bash
# PrunePEFT training script
# Author: zzh
# Modify parameters below as needed
export CUDA_VISIBLE_DEVICES=2,3

# Adapter types: comma-separated list like "lora" or "lora,bottleneck"
ADAPTER_TYPES="lora,bottleneck"

# LoRA parameters (used when ADAPTER_TYPES includes "lora")
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Bottleneck parameters (used when ADAPTER_TYPES includes "bottleneck")
BOTTLENECK_SIZE=64
BOTTLENECK_DROPOUT=0.1
INIT_BOTTLENECK_WEIGHTS=true

# Layer selection: specify which layers to apply adapters (comma-separated integers, e.g., "0,1,2")
# If empty, adapters will be applied to all layers
ADAPTER_LAYERS="0,1,2"  # Layers to apply Bottleneck adapter (block-level), e.g., "0,1,2"
LORA_LAYERS="2,3,4"  # Layers to apply LoRA adapter (linear-level), e.g., "0,1,2"

# Training parameters
TARGET_MODULES="q_proj,v_proj"
SAMPLE_SIZE=128
SEED=42
BIAS="none"
MAX_LENGTH=256
MAX_STEPS=10
ZERO_PAD_TO_MAX_RANK=true   # set false for baseline
ZERO_PAD_RANK=0             # 0 -> use max(rank_pattern, lora_rank); >0 -> force
ZERO_PAD_MODE="bucket"      # none|max|bucket
ZERO_PAD_BUCKET_RANKS="8,16,32,64"

# Heterogeneous LoRA ranks (regex -> rank). Note: keys are regex strings, so we escape '.' as '\\.'.
# NOTE: This is a JSON string. Use JSON escaping (\\.) to represent a single backslash in the final regex.
RANK_PATTERN='{
  "layers\\.2\\.self_attn\\.base_block\\.q_proj": 8,
  "layers\\.2\\.self_attn\\.base_block\\.v_proj": 8,
  "layers\\.3\\.self_attn\\.q_proj": 16,
  "layers\\.3\\.self_attn\\.v_proj": 16,
  "layers\\.4\\.self_attn\\.q_proj": 32,
  "layers\\.4\\.self_attn\\.v_proj": 32
}'

# Build command
CMD="python examples/prunepeft.py \
    --adapter_types=$ADAPTER_TYPES \
    --lora_rank=$LORA_RANK \
    --lora_alpha=$LORA_ALPHA \
    --lora_dropout=$LORA_DROPOUT \
    --bottleneck_size=$BOTTLENECK_SIZE \
    --bottleneck_dropout=$BOTTLENECK_DROPOUT \
    --init_bottleneck_weights=$INIT_BOTTLENECK_WEIGHTS \
    --adapter_layers=$ADAPTER_LAYERS \
    --lora_layers=$LORA_LAYERS \
    --target_modules=$TARGET_MODULES \
    --rank_pattern='$RANK_PATTERN' \
    --max_length=$MAX_LENGTH \
    --max_steps=$MAX_STEPS \
    --zero_pad_to_max_rank=$ZERO_PAD_TO_MAX_RANK \
    --zero_pad_rank=$ZERO_PAD_RANK \
    --zero_pad_mode=$ZERO_PAD_MODE \
    --zero_pad_bucket_ranks=$ZERO_PAD_BUCKET_RANKS \
    --sample_size=$SAMPLE_SIZE \
    --seed=$SEED \
    --bias=$BIAS"

# Execute command
eval $CMD