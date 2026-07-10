# %%
# ------ Load libraries ------
import os

# ENV VARIABLES: MUST be set before importing numpy/torch to prevent thread oversubscription
NUM_THREADS_TO_SET = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS_TO_SET)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS_TO_SET)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS_TO_SET)


import numpy as np
import torch
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pathlib import Path
import argparse
import pickle
from pymoo.core.callback import Callback
import json

# local functions 
from my_libs import (
    MultiObjectiveProblem_unified,
    dSmax_dthickness_randVar,
    SaveToPickleCallback_noPOP,
    default_idx,
)

# parallel
from pymoo.parallelization import StarmapParallelization
import multiprocessing


# ==============================================================================
# SUPPORT FUNCTIONS
# ==============================================================================

def save_optimization_results(res, pop_size, obj_kwargs, dataset_path, n_bilayers, seed=0):
    """
    Save the final optimization results (Pareto front) to NetCDF using xarray.
    (History saving is delegated to the Pickle callback to save RAM).
    """
    if res.X is not None and res.F is not None:
        print("✅ Optimization results available, preparing to save final Pareto front to NetCDF...")
        
        X = np.array(res.X)
        F = np.array(res.F)

        n_solutions, n_params = X.shape

        # Create Dataset
        results_ds = xr.Dataset(
            data_vars={
                "parameters": (["solution", "param"], X),
                "objective_value": (["solution", "objective"], F.reshape(n_solutions, -1)),
            },
            coords={
                "solution": np.arange(n_solutions),
                "param": [f"x{i}" for i in range(n_params)],
                "objective": [f"f{i}" for i in range(F.reshape(n_solutions, -1).shape[1])]
            },
        )
        
        n_gen = getattr(res, "n_gen", 0)

        # --- Add metadata ---
        # Save parameter string to JSON, converting tensors or non-serializable objects
        clean_kwargs = {k: (v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else str(v)) for k, v in obj_kwargs.items()}
        
        results_ds.attrs.update({
            "population_size": pop_size,
            "generations": n_gen,
            "seed": seed,
            "obj_kwargs": json.dumps(clean_kwargs)
        })

        # --- Save to NetCDF ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(dataset_path) / f"OptSampling_bi{n_bilayers}_pop_{pop_size}_{timestamp}.nc"
        results_ds.to_netcdf(output_file)
        print(f"✅ Final Pareto Front saved to {output_file}")

    else:
        print("⚠️ No optimization results to save.")


# ==============================================================================
# MAIN OPTIMIZATION FUNCTION
# ==============================================================================

def genetic_optimization(seed=7, n_gen=1250, pop_size=800, n_bilayers=9, dataset_path="./data", runner=None, theta=60.0, pop_init=None, n_idx=None): 
    t_start = datetime.now() 
        
    # --- UNIFIED PARAMETER DEFINITION FOR THE NEW FUNCTION ---
    obj_params = {
        "theta0": theta,
        "range_theta": 0.5,
        "wl": torch.tensor([550.0]),
        "pol": 's',
        "n_sampling": 1000,
        "n_fine_sampl": 1000,
        "threshold": 0.95,
        "error_perc": 0.05,
        "test_size": 40,
        "seed": seed,
        "n_idx": n_idx
    }

    # Optimization setup
    n_layers = n_bilayers * 2 + 1 # Each bilayer has two layers

    # --- Define variable bounds ---
    xl = np.ones(n_layers) * 10.0 
    xu = np.ones(n_layers) * 400.0

    # --- Instantiate the Problem ---
    problem = MultiObjectiveProblem_unified(
        unified_func=dSmax_dthickness_randVar,
        func_kwargs=obj_params,
        n_vars=n_layers,
        xl=xl,
        xu=xu,
        elementwise_runner=runner,
    )
    
    if pop_init is not None:
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=pop_init,
            crossover=SBX(eta=15),
            mutation=PM(eta=15),
            eliminate_duplicates=True,
        )
    else:
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15),
            mutation=PM(eta=15),
            eliminate_duplicates=True,
        )

    # --- Define Termination Condition ---
    termination = get_termination("n_gen", n_gen)

    # --- Setup Callback ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_output_file = Path(dataset_path) / f"OptSampling_bi{n_bilayers}_pop_{pop_size}_{timestamp}.pkl"

    history_callback = SaveToPickleCallback_noPOP(
        filename=str(pkl_output_file),
        save_every=25,
        n_max_gen=n_gen
    )

    # --- Execute Minimization ---
    # Keep save_history=False to avoid accumulating data in RAM.
    # Incremental saving is handled by the callback.
    res = minimize(
        problem, 
        algorithm, 
        termination,
        seed=seed, 
        callback=history_callback,
        save_history=False, 
        verbose=True
    )

    # ==============================================================================
    # DISPLAY THE RESULTS
    # ==============================================================================
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)

    if res.X is not None and res.F is not None:
        X = res.X
        F = res.F

        print(f"Found {len(X)} optimal solutions.")
        print("First 5 solutions in variable space (X):")
        print(np.round(X[:5], 4))
        print("\nFirst 5 solutions in objective space (F):")
        print(np.round(F[:5], 4))
        print("-" * 50)
        
        # ==============================================================================
        # SAVE RESULTS TO XARRAY (Final NetCDF)
        # ==============================================================================
        save_optimization_results(
            res=res,
            pop_size=pop_size,
            obj_kwargs=obj_params,
            dataset_path=dataset_path,
            n_bilayers=n_bilayers,
            seed=seed
        )

    t_stop = datetime.now() 
    return t_start, t_stop      


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic Optimization for BSW sensors using Sampling.")

    parser.add_argument('--num-bilayers', type=int, required=True,
                        help='The number of coupled layers (bilayers) for the simulation.')
    parser.add_argument('--n-generation', type=int, required=True, 
                        help='The number of generations for the genetic algorithm.')
    parser.add_argument('--pop-size', type=int, required=True, 
                        help='The size of the population (number of individuals) per generation.')
    parser.add_argument('--theta', type=float, required=True, 
                        help='The value of theta parameter for the genetic algorithm.')
    parser.add_argument('--dataset-path', type=str, default="./data", 
                        help='Relative path to the data folder (default: ./data).')
    parser.add_argument('--init-pop', type=str, default=None, 
                        help='Relative path to a pre-computed initial population (.nc file).')
    parser.add_argument('--n-process', type=int, default=max(1, (os.cpu_count())), 
                        help='Number of CPU cores to use. Default is the maximum available cores.')

    # Parse arguments
    args = parser.parse_args()

    num_bilayers = args.num_bilayers
    n_generation = args.n_generation
    population_size = args.pop_size
    theta_value = args.theta
    dataset_path = args.dataset_path
    n_process = args.n_process

    print("--- OPTIMIZATION PARAMETERS RECEIVED ---")
    print(f"Number of Bilayers: {num_bilayers}")
    print(f"Generations to run: {n_generation}")
    print(f"Population Size: {population_size}")
    print(f"Theta Value: {theta_value}")
    print(f"CPU Cores in use: {n_process}")
    print(f"Dataset Path: {dataset_path}")

    # --- parallel comp ----
    pool = multiprocessing.Pool(n_process)
    runner = StarmapParallelization(pool.starmap)

    # ------------------ Local variables definition ------------------
    seed_value = 55
    legend_info = []

    # Handle Initial Population Loading
    initial_population = None
    if args.init_pop is not None:
        try:
            ds = xr.open_dataset(args.init_pop)
            initial_population = ds['X'].values 
            print(f"Initial population successfully loaded. Shape: {initial_population.shape}")
        except Exception as e:
            print(f"Error: Unable to load initial population from {args.init_pop}.")
            print(f"Details: {e}")
            assert False, "Initial population loading failed. Check the path and file format."

    # Run optimization
    t_start, t_stop = genetic_optimization(
        seed=seed_value,
        n_gen=n_generation,
        pop_size=population_size,
        n_bilayers=num_bilayers,
        dataset_path=dataset_path,
        runner=runner,
        theta=theta_value,
        pop_init=initial_population,
        n_idx=default_idx
    )

    required_time = t_stop - t_start
    legend_info.append((str(required_time), str(t_stop), str(t_start)))

    print("\nOptimization completed.")
    print(f"Total execution time: {required_time}")
    print(f"Execution Log: {legend_info}")