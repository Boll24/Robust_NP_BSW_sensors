#!/bin/bash
#
# --- SLURM DIRECTIVES ---
# SLURM directives configure the resources and settings for the job.

#SBATCH --job-name=bsw_opt      # Name of the job (easy to identify in the queue)
#SBATCH --partition=LocalQ      # Specify the resource queue or partition to run on
#SBATCH -c 48                   # Request 48 CPU cores (or threads) for the task
#SBATCH --mem=64G               # Request 64 Gigabytes of RAM (memory)
#SBATCH -t 14-00:00:00          # Time limit for the job (14 days)
#SBATCH --output=logs/slurm-%j-%x.out   # File to which STDOUT will be written (relative to submission dir)
#SBATCH --error=logs/slurm-%j-%x.err    # File to which STDERR will be written (relative to submission dir)

# --- SLURM Best Practices ---

# Create the logs directory if it doesn't exist to prevent write errors
mkdir -p logs

# Start job and log environment details
echo "Starting job on host $(hostname) at $(date)" 

# --- Environment Setup ---
# ATTENZIONE: Decommenta queste righe e inserisci il nome esatto del tuo ambiente conda
# source ~/miniconda3/etc/profile.d/conda.sh  # (Oppure il percorso della tua installazione conda)
# conda activate bsw-optimization

# --- Optimization Parameters ---
NUM_BILAYERS_VAL=5
N_GENERATION_VAL=2000
POP_SIZE_VAL=4000
DATASET_PATH="./data"
INIT_POP_FILE="./data/initialPOP_4000_60_5bi.nc"

# --- Python Program Execution ---
PYTHON_SCRIPT="src/MultiObj_robustness_sampling.py"

# Costruiamo gli argomenti passando anche il numero di core assegnati da SLURM
PYTHON_ARGS="--num-bilayers $NUM_BILAYERS_VAL \
             --n-generation $N_GENERATION_VAL \
             --pop-size $POP_SIZE_VAL \
             --dataset-path $DATASET_PATH \
             --init-pop $INIT_POP_FILE \
             --n-process $SLURM_CPUS_PER_TASK"

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
# 1. LAUNCHING THE JOB: sbatch scripts/run_slurm_optimization.sh
# 2. MONITORING THE JOB: squeue -u $USER
# 3. CLUSTER INFORMATION: sinfo -p LocalQ
# 4. CANCELLING THE JOB: scancel <JOB_ID>