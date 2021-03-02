#!/bin/bash

#
# Launches a torch.distributed training run on a single node of a multi-node
# training.
#
# Note that PRIMARY (the primary node hostname) and PRIMARY_PORT (the TCP port
# used to establish communication with the primary node) must be provided as
# environment variables.
#
# (c) 2021, Brian J. Stucky
# UF Research Computiing
#

CHECKPOINT_PATH=checkpoints_2_node
VOCAB_FILE=../data/vocab.txt
DATA_PATH=../data/uf1_TEXT_sentence

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters 200000 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
           --micro-batch-size 4 \
           --global-batch-size 256 \
           --vocab-file $(realpath $VOCAB_FILE) \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 20000 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

TRAINING_SCRIPT="$(realpath Megatron-LM/pretrain_bert.py)"
TRAINING_CMD="$TRAINING_SCRIPT \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $(realpath $CHECKPOINT_PATH) \
       --load $(realpath $CHECKPOINT_PATH) \
       --data-path $(realpath $DATA_PATH)"


# This should be the complete command to launch the per-node training run.
LAUNCH_CMD="singularity exec --nv \
    /apps/nvidia/containers/pytorch/20.12-py3.sif python \
        -m torch.distributed.launch \
              --nproc_per_node=$SLURM_GPUS_PER_TASK \
              --nnodes=$SLURM_JOB_NUM_NODES \
              --node_rank=$SLURM_NODEID \
              --master_addr=$PRIMARY \
              --master_port=$PRIMARY_PORT \
            $TRAINING_CMD"

source pt_multinode_helper_funcs.sh
run_with_retry "$LAUNCH_CMD"

