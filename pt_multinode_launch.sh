#!/bin/bash

#
# Script to launch a multi-node pytorch.distributed training run.
#
# (c) 2021, Brian J. Stucky
# UF Research Computing
#

##### resource allocation
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER@DOMAIN
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=96gb
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI
# Enable the following to limit the allocation to a single SU.
## SBATCH --constraint=su7
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=test_%j.out


# The script that runs torch.distributed on each node. It will receive the
# primary node hostname and TCP port to contact the primary node as the
# environment variables PRIMARY and PRIMARY_PORT.
PT_LAUNCH_SCRIPT=run_pt_on_node.sh

source pt_multinode_helper_funcs.sh
#export NCCL_DEBUG=INFO
init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

echo "Running $(realpath $PT_LAUNCH_SCRIPT) on each node..."
srun $(realpath $PT_LAUNCH_SCRIPT)

