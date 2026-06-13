#!/bin/bash
# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL deepspeed --num_gpus=2 moe_model.py
# deepspeed --num_gpus=2 moe_model.py

nsys profile \
    -o moe_profile \
    --trace=cuda,nvtx,ucx \
    --stats=true \
    deepspeed --num_gpus=2 moe_model.py
 