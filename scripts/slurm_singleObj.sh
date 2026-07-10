#!/bin/bash
#
# --- SLURM DIRECTIVES ---
# SLURM directives configure the resources and settings for the job.

#SBATCH --job-name=bsw_single     # Name of the job (easy to identify in the queue)
#SBATCH --partition=LocalQ        # Specify the resource queue or partition to run on
#SBATCH -c 48                     # Request 48 CPU cores (or threads) for the task
#SBATCH --mem=64G                 # Request 64 Gigabytes of RAM (memory)
#SBATCH -t 14-00:00:00            # Time limit for the job (14 days)
#SBATCH --output=logs/slurm-%j-%x.out   # File to which STDOUT will be written (relative to submission dir)
#SBATCH --error=logs/slurm-%j-%x.err    # File to which STDERR will be written (relative to submission dir)

# --- SLURM Best Practices ---

# Create the logs directory if it doesn't exist to prevent write errors
mkdir -p logs

# Start job and log environment details
echo "Starting Single-Objective job on host $(hostname) at $(date)" 

# --- Environment Setup ---
# ATTENZIONE: Decommenta queste righe e inserisci il nome esatto del tuo ambiente conda
# source ~/miniconda3/etc/profile.d/conda.sh  # (Oppure il percorso della tua installazione conda)
# conda activate bsw-optimization

# --- Optimization Parameters ---
NUM_BILAYERS_VAL=5
N_GENERATION_VAL=500
POP_SIZE_VAL=4000
THETA_VAL=60.0
DATASET_PATH="./data"
VARIANT_TYPE="DE/target-to-best/1/bin"

# NOTA: Per passare una lista a argparse da bash, separa i numeri con uno spazio
SEEDS_LIST="22 34 56 78 90 123 145 167 189 210"

# --- Python Program Execution ---
PYTHON_SCRIPT="src/SingleObj_sensitivity.py"

# Costruiamo gli argomenti per lo script Python
PYTHON_ARGS="--num-bilayers $NUM_BILAYERS_VAL \
             --n-generation $N_GENERATION_VAL \
             --pop-size $POP_SIZE_VAL \
             --theta $THETA_VAL \
             --dataset-path $DATASET_PATH \
             --variant $VARIANT_TYPE \
             --seeds $SEEDS_LIST"

echo "Executing: python $PYTHON_SCRIPT $PYTHON_ARGS"

# Esecuzione
python $PYTHON_SCRIPT $PYTHON_ARGS

# Check the exit status of the Python process (Crucial for robust jobs)
if [ $? -ne 0 ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT' failed with exit status $?."
    exit 1
fi

echo "Job completed successfully at $(date)"

# --- SLURM Command Guide for Launching and Monitoring This Script ---
#
# 1. LAUNCHING THE JOB: sbatch scripts/run_single_obj.sh
# 2. MONITORING THE JOB: squeue -u $USER
# 3. CLUSTER INFORMATION: sinfo -p LocalQ
# 4. CANCELLING THE JOB: scancel <JOB_ID>