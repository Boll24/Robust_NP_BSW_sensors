# %%
# ------ Load libraries ------
import os

# ENV VARIABLES: MUST be set before importing numpy/torch to prevent thread oversubscription
NUM_THREADS_TO_SET = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS_TO_SET)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS_TO_SET)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS_TO_SET)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS_TO_SET)

import numpy as np
import torch
import xarray as xr
from datetime import datetime
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import argparse
import json
from pathlib import Path

# ------ Load imported functions ------
from my_libs import SingleObjectiveProblemVectorized
from my_libs import Finite_Difference_4or_multiple


# ==============================================================================
# SUPPORT FUNCTIONS
# ==============================================================================

def save_optimization_results(res, variant, pop_size, seed, obj_kwargs, dataset_path, n_bilayers, theta=60.0):
    """
    Save full or partial optimization results to NetCDF using xarray,
    with a descriptive filename containing the number of bilayers and population size.

    Parameters
    ----------
    res : object
        Result object from the optimization algorithm (e.g. pymoo result).
    variant : str
        Name of the optimization algorithm used (e.g. "DE/best/1/bin").
    pop_size : int
        Population size used during optimization.
    seed : int
        Random seed of the run.
    obj_kwargs : dict
        Additional parameters used in the objective function.
    dataset_path : str or Path
        Directory where the results will be saved.
    n_bilayers : int
        Number of bilayers (used in file naming).

    Returns
    -------
    output_file : Path
        Full path to the saved NetCDF file.
    """
    if res.X is not None and res.F is not None:

        # ✅ Check for full history (KEPT AS REQUESTED)
        if hasattr(res, "history") and len(res.history) > 0:
            print("Saving full optimization history...")

            generations = []
            for h in res.history:
                X = h.pop.get("X")
                F = h.pop.get("F").flatten()
                G = h.pop.get("G") if h.pop.get("G") is not None else np.full((X.shape[0], 0), np.nan)
                generations.append((X, F, G))

            n_gen = len(generations)
            n_individuals = generations[0][0].shape[0]
            n_params = generations[0][0].shape[1]
            n_constraints = generations[0][2].shape[1] if generations[0][2].ndim > 1 else 0

            # --- Build arrays ---
            X_all = np.zeros((n_gen, n_individuals, n_params))
            F_all = np.zeros((n_gen, n_individuals))
            G_all = np.full((n_gen, n_individuals, max(1, n_constraints)), np.nan)

            for i, (X, F, G) in enumerate(generations):
                X_all[i, :, :] = X
                F_all[i, :] = F
                if n_constraints > 0:
                    G_all[i, :, :G.shape[1]] = G

            # --- Create Dataset ---
            coords = {
                "generation": np.arange(n_gen),
                "individual": np.arange(n_individuals),
                "param": [f"x{i}" for i in range(n_params)]
            }

            data_vars = {
                "population_X": (["generation", "individual", "param"], X_all),
                "population_F": (["generation", "individual"], F_all),
            }

            if n_constraints > 0:
                data_vars["population_G"] = (["generation", "individual", "constraint"], G_all)
                coords["constraint"] = [f"g{i}" for i in range(n_constraints)]

            results_ds = xr.Dataset(data_vars, coords=coords)

        else:
            # 🧭 Fallback — save only best solution
            print("No optimization history available, saving only best result...")

            results_da = xr.DataArray(
                np.array(res.X),
                dims=["param"],
                coords={"param": [f"x{i}" for i in range(len(res.X))]},
                name="optimized_parameters",
            )

            results_ds = xr.Dataset({"parameters": results_da})
            results_ds["objective_value"] = xr.DataArray(res.F, dims=["objective"])
            n_gen = getattr(res, "n_gen", 0)  # fallback if missing

        # --- Add metadata ---
        results_ds.attrs.update({
            "algorithm": variant,
            "population_size": pop_size,
            "generations": n_gen,
            "seed": seed,
            "obj_kwargs": json.dumps(obj_kwargs, default=str)
        })

        # --- Save to NetCDF with descriptive filename ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(dataset_path) / f"OptSingleObj_bi{n_bilayers}_t{theta}_{timestamp}.nc"
        results_ds.to_netcdf(output_file)
        print(f"✅ Optimization results saved to {output_file}")

    else:
        print("⚠️ No optimization results to save.")


# ==============================================================================
# MAIN OPTIMIZATION FUNCTIONS
# ==============================================================================

def obj_func(
    x,
    n_sub=1.57,
    n_sup=1.33,
    n_l=1.45 + 1j * 1e-5,
    n_h=2.15 + 1j * 1e-4,
    theta=60.0,
    wl=550.0,
    pol="s",
):
    assert x.shape[1] % 2 == 1, "Number of layers must be odd."
    x_tmm = torch.from_numpy(np.array(x, dtype=np.float64))
    n_idx = torch.tensor([n_sub, n_l, n_h, n_sup], dtype=torch.complex128)

    # Compute objective
    _, obj = Finite_Difference_4or_multiple(x_tmm, torch.tensor([theta*np.pi/180]), torch.tensor([wl]), n_idx=n_idx, pol=pol)
    return -obj.squeeze(-1).detach().numpy().real


def genetic_optimization(seeds, variant, n_gen=1250, pop_size=800, save=True, n_bilayers=9, theta=60.0, dataset_path="./data"): 
    t_start = datetime.now() 
    
    for seed in seeds:
        print(f"\n--- Starting Optimization for Seed {seed} ---")
        
        obj_kwargs = {
            "n_sub": 1.57,
            "n_sup": 1.33,
            "n_l": 1.45 + 1j * 1e-5,
            "n_h": 2.15 + 1j * 1e-4,
            "theta": theta,
            "wl": 550.0,
            "pol": "s",
        }

        # --- Configure constraints ---
        constr_funcs = []
        constr_kwargs_list = None

        # --- Configure bounds ---
        n_params = n_bilayers * 2 + 1
        xl = np.array([10.0] * n_params)
        xu = np.array([400.0] * n_params)

        # --- Instantiate the Problem ---
        problem = SingleObjectiveProblemVectorized(
            obj_func=obj_func,
            obj_kwargs=obj_kwargs,
            constr_funcs=constr_funcs,
            constr_kwargs_list=constr_kwargs_list,
            n_vars=n_params,
            xl=xl,
            xu=xu,
        )

        CR = 0.9
        F = 0.5

        algorithm = DE(pop_size=pop_size, variant=variant, CR=CR, F=F)

        # --- Define Termination Condition ---
        termination = get_termination("n_gen", n_gen)

        # --- Run the Optimization ---
        res = minimize(problem, algorithm, termination, seed=seed, verbose=True, save_history=save)

        # ==============================================================================
        # DISPLAY THE RESULTS
        # ==============================================================================
        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        if res.X is not None:
            print(f"Best solution found (X): {np.round(res.X, 4)}")
        else:
            print("No solution found.")
            
        if res.F is not None:
            print(f"Objective value (F): {np.round(res.F, 4)}")
        else:
            print("No solution found.")
            
        if res.G is not None:
            print(f"Constraint values (G): {np.round(res.G, 4)}")
        
        print("-" * 50)
        print("Parameters used during evaluation (stored in Problem object):")
        print(f"  Objective function: {problem.obj_func.__name__}")
        print(f"  Constraint functions: {[f.__name__ for f in problem.constr_funcs]}")
        
        # ==============================================================================
        # SAVE RESULTS TO XARRAY
        # ==============================================================================
        save_optimization_results(
            res=res,
            variant=variant,
            pop_size=pop_size,
            seed=seed,
            theta=theta,
            obj_kwargs=obj_kwargs,
            dataset_path=dataset_path,
            n_bilayers=n_bilayers
        )

    t_stop = datetime.now() 
    return t_start, t_stop      


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Objective Optimization (Differential Evolution) for BSW sensors.")

    parser.add_argument('--num-bilayers', type=int, required=True,
                        help='The number of coupled layers (bilayers) for the simulation.')
    parser.add_argument('--n-generation', type=int, required=True, 
                        help='The number of generations for the optimization algorithm.')
    parser.add_argument('--pop-size', type=int, required=True, 
                        help='The size of the population (number of individuals) per generation.')
    parser.add_argument('--theta', type=float, required=True, 
                        help='The value of the incidence angle theta.')
    parser.add_argument('--dataset-path', type=str, default="./data", 
                        help='Relative path to the data folder (default: ./data).')
    parser.add_argument('--variant', type=str, default="DE/target-to-best/1/bin", 
                        help='Differential Evolution variant to use.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[22, 34, 56, 78, 90, 123, 145, 167, 189, 210], 
                        help='List of random seeds to run (e.g., --seeds 22 34 56).')

    # Parse arguments
    args = parser.parse_args()

    num_bilayers = args.num_bilayers
    n_generation = args.n_generation
    population_size = args.pop_size
    theta_value = args.theta
    dataset_path = args.dataset_path
    variant_type = args.variant
    seed_values = args.seeds

    print("--- OPTIMIZATION PARAMETERS RECEIVED ---")
    print(f"Number of Bilayers: {num_bilayers}")
    print(f"Generations to run: {n_generation}")
    print(f"Population Size: {population_size}")
    print(f"Theta Value: {theta_value}")
    print(f"DE Variant: {variant_type}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Seeds to process: {seed_values}")

    # For Single Objective DE in this script, history is kept enabled
    save_history_flag = True

    # Run optimization
    t_start, t_stop = genetic_optimization(
        seeds=seed_values,
        variant=variant_type,
        n_gen=n_generation,
        pop_size=population_size,
        theta=theta_value,
        save=save_history_flag,
        n_bilayers=num_bilayers,
        dataset_path=dataset_path
    )

    required_time = t_stop - t_start
    print("\n" + "=" * 50)
    print("ALL OPTIMIZATIONS COMPLETED")
    print(f"Total execution time: {required_time}")
    print("=" * 50)