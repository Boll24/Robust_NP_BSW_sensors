# libraries
import numpy as np
import torch
from tmm_fast import coh_tmm as tmm
import matplotlib.pyplot as plt

# default_idx is a tensor of refractive indices organized like [n_sub, n_low, n_high, n_sup]
default_idx= torch.tensor([1.57, 1.45 + 1j*1e-5, 2.15 + 1j*1e-4, 1.33 ], dtype=torch.complex128)

def Finite_Difference_4or_multiple(thicknesses, theta, wl, n_idx=default_idx, pol='s', delta_n=3e-7):
    """
    Compute the finite difference derivative of the reflection matrix (S). 
    Suited for multiple stacks.

    Args:
        thicknesses (torch.Tensor): Thickness values of all layers in nanometers.
        theta (torch.Tensor): Incident angle(s) in radians. Can be a single scalar value or a tensor representing an array of angles.
        wl (torch.Tensor): Wavelength(s) in nanometers. Can be a single scalar value or a tensor representing an array of wavelengths.
        n_idx (list): Refractive indices [n_substrate, n_low, n_high, n_superstrate].
        pol (str): Polarization ('s' or 'p').
        delta_n (float): Perturbation step for refractive index.

    Returns:
        R (torch.Tensor): Reflection coefficients for perturbed superstrate refractive index.
        S (torch.Tensor): Finite difference derivative w.r.t refractive index (central difference).
    """
    T = generate_T_4or_multiple(thicknesses)
    M = generate_n_mat_multiple_4or(n_idx, T.shape[1], T.shape[0], delta_n)
    O = tmm(pol, M, T, theta, wl)
    R = O.get('R')
    quarter = R.shape[0] // 4
    S=(+R[:quarter, :, :] - 8*R[quarter:2*quarter, :, :]+8*R[2*quarter:3*quarter, :, :] - R[3*quarter:, :, :]) / (12 * delta_n)
    return R, S


def generate_T_4or_multiple(T):
    """
    Generate a thickness matrix repeated 4 times to enable batched computation 
    of the 4th-order finite difference derivative.

    This function creates a macro-stack by cloning the input thickness tensor 
    4 times. It also adds zero-thickness padding columns at the beginning and 
    end to represent the substrate and superstrate layers, which are required 
    by the transfer matrix method (TMM) solver.

    Args:
        T (torch.Tensor): Thickness values of the intermediate layers.

    Returns:
        torch.Tensor: A padded thickness matrix containing 4 identical copies 
        of the input stack, ready for batched TMM evaluation.
    """

    stacks = [T.clone(), T.clone() , T.clone() , T.clone()]  # reference copies

    mat_T = torch.cat(stacks, dim=0)
    # add zero-thickness padding columns for substrate and superstrate
    zeros_col = torch.zeros((mat_T.shape[0], 1), dtype=mat_T.dtype)
    mat_T = torch.cat([zeros_col, mat_T, zeros_col], dim=1)

    return mat_T


def generate_n_mat_multiple_4or(n_idx, L, num_stack, delta_n):
    """
    Build the refractive index matrix for a multilayer stack,
    assigning n_sup - delta_n to the first half of stacks,
    and n_sup + delta_n to the second half.

    Args:
        n_idx (list): Refractive indices [n_substrate, n_low, n_high, n_superstrate].
        L (int): Number of layers in the stack.
        num_stack (int): Number of stack repetitions.
        delta_n (float): Perturbation applied to the superstrate refractive index.

    Returns:
        torch.Tensor: Refractive index tensor with shape (num_stack, L_total, num_wl).
    """
    n_sub, n_low, n_high, n_sup = n_idx
    num_wl = 1

    # tensors for each layer type
    n_low_tensor = n_low.expand(num_stack, 1, num_wl)
    n_high_tensor = n_high.expand(num_stack, 1, num_wl)
    n_sub_tensor = n_sub.expand(num_stack, 1, num_wl)

    # superstrate split into two halves
    n_sup_expand = n_sup.expand(num_stack, 1, num_wl).clone()
    quarter = num_stack // 4
    n_sup_expand[:quarter, :, :] -= 2*delta_n   # primo quarto
    n_sup_expand[quarter:2*quarter, :, :] -= delta_n   # primo quarto
    n_sup_expand[2*quarter:3*quarter, :, :] += delta_n 
    n_sup_expand[3*quarter:, :, :] += 2*delta_n   # seconda metà

    # build layer sequence
    layers = []
    for i in range(L):
        if i == 0:
            layers.append(n_sub_tensor)      # substrate
        elif i == L - 1:
            layers.append(n_sup_expand)      # superstrate
        elif i % 2 == 1:
            layers.append(n_low_tensor)      # low-index layer
        else:
            layers.append(n_high_tensor)     # high-index layer

    n_mat = torch.cat(layers, dim=1)  # shape: (num_stack, L_total, num_wl)
    return n_mat

# ---------------------------------------------------------------------------------------------

#                       FUNCTION BASED ON FITTING THE REFLECTANCE CURVE

# ---------------------------------------------------------------------------------------------

def generate_n_matrix(n_idx, L, num_stack):
    """
    Build the refractive index matrix for a multilayer stack to evaluate 
    the baseline optical response (no perturbations).

    Args:
        n_idx (list): Refractive indices [n_substrate, n_low, n_high, n_superstrate].
        L (int): Number of layers in the stack.
        num_stack (int): Number of stack repetitions (for batched computation).

    Returns:
        torch.Tensor: Refractive index tensor with shape (num_stack, L, num_wl).
    """
    n_sub = n_idx[0]
    n_low = n_idx[1]
    n_high = n_idx[2]
    n_sup = n_idx[3]

    num_wl = 1

    # tensors for each layer type
    n_low_tensor = n_low.expand(num_stack, 1, num_wl)
    n_high_tensor = n_high.expand(num_stack, 1, num_wl)
    n_sub_tensor = n_sub.expand(num_stack, 1, num_wl)
    n_sup_expand = n_sup.expand(num_stack, 1, num_wl)

    # build layer sequence
    layers = []
    for i in range(L):
        if i == 0:
            layers.append(n_sub_tensor)      # substrate
        elif i == L - 1:
            layers.append(n_sup_expand)      # superstrate
        elif i % 2 == 1:
            layers.append(n_low_tensor)      # low-index layer
        else:
            layers.append(n_high_tensor)     # high-index layer

    mat_n = torch.cat(layers, dim=1)  # shape: (num_stack, L, num_wl)
    return mat_n

def generate_R_spectra(T_array, 
                       theta0=60.0, 
                       range_theta=0.5, 
                       wl=torch.tensor([550.0]),
                       pol='s',
                       n_sampling=2000,
                       n_idx=default_idx):
    """
    Generate reflectivity spectra for a given set of multilayer stacks over a specified angular range.

    Args:
        T_array (torch.Tensor): Thickness values of the intermediate layers (shape: num_stack, L).
        theta0 (float): Central incidence angle in degrees.
        range_theta (float): Angular range (+/- theta0) to simulate, in degrees.
        wl (torch.Tensor): Wavelength in nanometers.
        pol (str): Polarization ('s' or 'p').
        n_sampling (int): Number of angular points to sample.
        n_idx (torch.Tensor): Refractive indices of the materials.

    Returns:
        R (torch.Tensor): Reflectivity spectra (shape: num_stack, n_sampling).
        v_theta_deg (torch.Tensor): Sampled angles in degrees.
    """
    assert torch.is_tensor(T_array), "T_array deve essere un tensore PyTorch"
    assert torch.is_tensor(wl), "wl deve essere un tensore PyTorch"
    
    # Create the angular vector (in radians) centered around theta0
    v_theta = torch.linspace(theta0 - range_theta, theta0 + range_theta, n_sampling) * torch.pi / 180.0 
    
    # Add zero-thickness padding columns for substrate and superstrate required by TMM
    zeros_col = torch.zeros((T_array.shape[0], 1), dtype=T_array.dtype)
    T = torch.cat([zeros_col, T_array, zeros_col], dim=1)
    
    # Generate the nominal refractive index matrix
    M = generate_n_matrix(n_idx, L=T.shape[1], num_stack=T.shape[0])
    
    # Compute the optical response using the Transfer Matrix Method
    O = tmm(pol, M, T, v_theta, wl)
    
    # Extract Reflectivity and squeeze the last dimension
    R = O.get('R').squeeze(-1)  
    
    return R, v_theta * 180 / torch.pi

def get_resonance_parameters(T_array,
                           theta0=60.0,
                           range_theta=1.0,
                           wl=torch.tensor([550.0]),
                           n_sampling=2000,
                           n_idx=default_idx,
                           pol='s',
                           weight_tail=0.1):
    """
    Extract key BSW resonance parameters (centroid, FWHM, optimal working angle, sensitivity) 
    by analytically fitting the reflectivity dip to a Lorentzian model.

    Args:
        T_array (torch.Tensor): Thickness tensor of the stacks.
        theta0, range_theta, wl, n_sampling, n_idx, pol: TMM simulation parameters.
        weight_tail (float): Threshold to mask asymmetric tails during the fitting process.

    Returns:
        dict: Contains the analytical resonance angle, FWHM, minimum value, working angle, and peak sensitivity.
    """
    # Generate the reflectivity spectra curves to be fitted
    R, v_theta = generate_R_spectra(T_array, theta0=theta0, range_theta=range_theta, 
                                    wl=wl, pol=pol, n_sampling=n_sampling, n_idx=n_idx)
    
    ### ANALYTICAL FITTING ###
    
    # 1. Center the X-axis to prevent numerical ill-conditioning during matrix inversion
    theta_mean = v_theta.mean()
    theta_c = v_theta - theta_mean 
    theta_c_sq = theta_c ** 2
    
    # 2. Calculate the dynamic baseline (max reflectivity) and invert the resonance dip
    y_base = R.max(dim=1, keepdim=True).values
    Y = y_base - R  
    Y_max = Y.max(dim=1, keepdim=True).values
    
    # 3. Hard Masking: Ignore asymmetric tails that fall below the specified threshold
    mask = (Y > weight_tail * Y_max).float()
    
    # Apply mask and weight (W) to prioritize the deepest, most symmetric part of the resonance
    W = mask * Y
    
    # 4. Build the weighted design matrix and target vector for the linear system
    col_A = W * (Y * theta_c_sq)
    col_B = W * (Y * theta_c)
    col_C = W * Y
    M = torch.stack([col_A, col_B, col_C], dim=-1)
    V = W.unsqueeze(-1)
    
    # 5. Solve the linear system to find the Lorentzian coefficients
    solution = torch.linalg.lstsq(M, V).solution
    A = solution[:, 0, 0]
    B = solution[:, 1, 0]
    C = solution[:, 2, 0]
    
    # 6. Extract physical parameters and reverse the axis centering
    thetac0 = -B / (2 * A)
    theta_0 = thetac0 + theta_mean 
    
    # Calculate Full Width at Half Maximum (FWHM) with a safety clamp to prevent NaNs
    gamma_sq = torch.clamp(C / A - thetac0**2, min=1e-12)
    gamma = torch.sqrt(gamma_sq)
    FWHM = 2 * gamma
    
    # Calculate the optimal working angle (point of maximum analytical slope)
    theta_w = theta_0 - (gamma / np.sqrt(3.0))
    
    # Calculate the actual depth of the analytical dip using the dynamic baseline
    I0 = 1.0 / (A * gamma_sq)
    min_val = y_base.squeeze(-1) - I0

    ### SENSITIVITY COMPUTATION ###
    
    # Calculate intensity sensitivity at the dynamically fitted working angle
    _, sens = Finite_Difference_4or_multiple(T_array, n_idx=n_idx, wl=wl, theta=theta_w * torch.pi / 180)
    
    # Extract the diagonal to match each specific structure with its own optimal angle
    sens_diag = torch.diagonal(sens)

    return {
        'theta_res': theta_0,
        'FWHM': FWHM,
        'min_val': min_val,
        'theta_W': theta_w,
        'top_sens': sens_diag
    }

def sensitivity_decomposition(T_array,
                           theta0=60.0,
                           range_theta=1.0,
                           wl=torch.tensor([550.0]),
                           n_sampling=2000,
                           n_idx=default_idx,
                           pol='s',
                           weight_tail=0.1,
                           v_dn_sup=torch.linspace(-0.01, 0.01, 5)):
    """
    Decompose the overall sensitivity by isolating the angular sensitivity (S_theta).
    This is achieved by perturbing the superstrate refractive index and tracking the resonance shift.

    Args:
        T_array (torch.Tensor): Thickness tensor of the stacks.
        v_dn_sup (torch.Tensor): Array of refractive index perturbations to apply to the superstrate.
        (Other arguments are passed down to get_resonance_parameters).

    Returns:
        dict: A comprehensive collection of resonance parameters across all perturbations, 
              including the nominal state (suffix '_0') and the computed angular sensitivity.
    """
    total_res = []
    total_res_0 = []
    
    # Matrix to store the resonance angles for each stack under each perturbation
    theta_res = torch.zeros((T_array.shape[0], v_dn_sup.shape[0]), dtype=torch.float64)
    
    # Create a matrix of refractive indices, applying the perturbations to the superstrate (last column)
    n_idx_matrix = n_idx.clone().repeat(v_dn_sup.shape[0], 1)
    n_idx_matrix[:, -1] = n_idx_matrix[:, -1] + v_dn_sup
    
    # Iterate over each perturbation state
    for i, current_n_idx in enumerate(n_idx_matrix):
        res = get_resonance_parameters(T_array, theta0=theta0, range_theta=range_theta,
                                       wl=wl, n_sampling=n_sampling, 
                                       n_idx=current_n_idx, 
                                       pol=pol, weight_tail=weight_tail)
        
        theta_res[:, i] = res['theta_res']
        total_res.append(res)
        
        # Save the nominal state parameters (when perturbation is 0, i.e., the middle index)
        if i == (v_dn_sup.shape[0] - 1) // 2:
            total_res_0.append(res)

    ### BATCHED LINEAR REGRESSION ###
    
    # Build the design matrix for regression: X = [Delta_n, 1]
    X = torch.stack([v_dn_sup, torch.ones_like(v_dn_sup)], dim=-1).to(dtype=theta_res.dtype)
    Y = theta_res.T 
    
    # Solve the least squares problem to find the slope (angular sensitivity) for all stacks simultaneously
    solution = torch.linalg.lstsq(X, Y).solution
    slopes = solution[0, :]  # Extract the angular sensitivity (Delta_theta / Delta_n)
    
    return {
        # Nominal state parameters (unperturbed)
        'theta_res_0': np.array([res['theta_res'].detach().cpu().numpy() for res in total_res_0]),
        'theta_W_0': np.array([res['theta_W'].detach().cpu().numpy() for res in total_res_0]),
        'FWHM_0': np.array([res['FWHM'].detach().cpu().numpy() for res in total_res_0]),
        'top_sens_0': np.array([res['top_sens'].detach().cpu().numpy() for res in total_res_0]),
        'min_val_0': np.array([res['min_val'].detach().cpu().numpy() for res in total_res_0]),
        
        # Parameters across all perturbed states
        'theta_res': np.array([res['theta_res'].detach().cpu().numpy() for res in total_res]),
        'FWHM': np.array([res['FWHM'].detach().cpu().numpy() for res in total_res]),
        'min_val': np.array([res['min_val'].detach().cpu().numpy() for res in total_res]),
        'theta_W': np.array([res['theta_W'].detach().cpu().numpy() for res in total_res]),
        'top_sens': np.array([res['top_sens'].detach().cpu().numpy() for res in total_res]), 
        
        # Final angular sensitivity derived from linear regression
        'angular_sens': slopes.detach().cpu().numpy(),                
    }


#--------------------------- Functions for Robustness calculation ------------------------------

def generate_T_randVar_normal(T, error_perc=0.05, seed=None, test_size=20):
    """
    Generate random thickness variations based on a Gaussian distribution.
    
    Args:
        T (torch.Tensor): Original thickness tensor.
        error_perc (float): Standard deviation as a percentage (e.g., 0.05 for 5%).
        seed (int, optional): Seed for reproducibility.
        test_size (int): Number of perturbed samples to generate.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Ensure T has the correct shape for broadcasting
    if T.dim() == 1:
        T = T.unsqueeze(0)
    
    T_expanded = T.repeat(test_size, 1)
    sigma = T_expanded * error_perc

    gaussian_noise = torch.randn_like(T_expanded) * sigma

    mat_T = T_expanded + gaussian_noise
    
    # Add zero-thickness padding columns for substrate and superstrate
    zeros_col = torch.zeros((test_size, 1), dtype=mat_T.dtype)
    mat_T = torch.cat([zeros_col, mat_T, zeros_col], dim=1)
    
    return mat_T

#---------------------------- Robustness calculation (FOM_R) rely on FITTING -------------------------------

def dS_max_fitting(thicknesses,
                   theta0=60.0,
                   range_theta=1.0,
                   wl=torch.tensor([550.0]),
                   seed=7,
                   test_size=40,
                   n_sampling=2000,
                   error_perc=0.05,
                   weight_tail=0.1,
                   n_idx=default_idx,
                   pol='s'):
    """
    Evaluate the nominal intensity sensitivity and the fabrication robustness (FOM_R) 
    of a given multilayer stack. This function is designed to be used as the objective 
    evaluator in the multi-objective genetic algorithm.

    Args:
        thicknesses (torch.Tensor): Nominal thickness values of the stack.
        theta0, range_theta, wl, n_sampling, n_idx, pol: TMM simulation parameters.
        seed (int): Random seed for reproducibility of the perturbed ensemble.
        test_size (int): Number of perturbed realizations to generate (ensemble size N).
        error_perc (float): Standard deviation of the thickness perturbation (e.g., 0.05 for 5%).
        weight_tail (float): Threshold to mask asymmetric tails during the fitting process.

    Returns:
        tuple: 
            - float: Negative nominal sensitivity (-S_I_0), inverted for minimization algorithms.
            - float: Mean absolute relative error of the sensitivity (FOM_R in percentage).
    """
    # Create the angular vector (in radians) centered around theta0
    v_theta = torch.linspace(theta0 - range_theta, theta0 + range_theta, n_sampling, dtype=torch.float64) 
    
    # Calculate the intensity sensitivity of the nominal (unperturbed) structure
    _ , sens_original = Finite_Difference_4or_multiple(thicknesses.unsqueeze(0), n_idx=n_idx,
                                                       wl=torch.tensor([wl]), theta=torch.tensor([theta0*np.pi/180]))
    
    # Generate a statistical ensemble of thickness-perturbed structures
    T = generate_T_randVar_normal(thicknesses, seed=seed, test_size=test_size, error_perc=error_perc)
    
    # Compute the reflectivity spectra for all perturbed stacks simultaneously
    M = generate_n_matrix(n_idx, L=T.shape[1], num_stack=T.shape[0])
    O = tmm(pol, M, T, v_theta*np.pi/180, torch.tensor([wl]))
    R = O.get('R').squeeze()
    
    ### ANALYTICAL FITTING ###
    # Find the new optimal working angle for each perturbed structure
    x_mean = v_theta.mean()
    xc = v_theta - x_mean 
    xc_sq = xc ** 2

    y_base = R.max(dim=1, keepdim=True).values
    Y = y_base - R  
    Y_max = Y.max(dim=1, keepdim=True).values
    
    mask = (Y > weight_tail * Y_max).float()
    W = mask * Y
    
    col_A = W * (Y * xc_sq)
    col_B = W * (Y * xc)
    col_C = W * Y
    M = torch.stack([col_A, col_B, col_C], dim=-1)
    V = W.unsqueeze(-1)
    
    solution = torch.linalg.lstsq(M, V).solution
    A = solution[:, 0, 0]
    B = solution[:, 1, 0]
    C = solution[:, 2, 0]
    
    xc0 = -B / (2 * A)
    x0 = xc0 + x_mean
    
    gamma_sq = torch.clamp(C / A - xc0**2, min=1e-12)
    gamma = torch.sqrt(gamma_sq)
    
    # Calculate the new optimal working angle for each realization
    theta_w = x0 - (gamma / np.sqrt(3.0))
    
    # Clamp the new working angle to ensure it stays within the simulated physical range
    theta_w = torch.clamp(theta_w, min=theta0-range_theta, max=theta0+range_theta)
    
    ### SENSITIVITY COMPUTATION ###
    # Evaluate the sensitivity of each perturbed stack at its own new optimal working angle
    dS_percent = torch.zeros(T.shape[0])
    for i, working_angle in enumerate(theta_w):
        # Exclude the zero-padding columns (substrate/superstrate) when calling Finite_Difference
        _, sens = Finite_Difference_4or_multiple(T[i, 1:-1].unsqueeze(0), n_idx=n_idx, wl=torch.tensor([wl]), theta=working_angle.unsqueeze(0)*np.pi/180)
        
        # Calculate the absolute relative percentage error w.r.t the nominal sensitivity
        dS_percent[i] = torch.abs(sens.squeeze() - sens_original.squeeze()) / torch.abs(sens_original.squeeze()) * 100.0
        
    # Calculate the mean across the ensemble (this corresponds to the FOM_R metric)
    avg_dS_percent = dS_percent.mean().item()
    
    # Return negative sensitivity (useful for minimization objectives like NSGA-II) and FOM_R
    return -sens_original.squeeze().item(), avg_dS_percent

#---------------------------- Robustness calculation (FOM_R) rely on SAMPLING -------------------------------
def dSmax_dthickness_randVar(x, 
                       theta0=65.0, 
                       range_theta=0.5, 
                       wl=torch.tensor([550.0]),
                       pol='s',
                       n_sampling=500,
                       n_fine_sampl=1000,
                       threshold=0.97,
                       seed=None,
                       error_perc=0.04,
                       test_size=20,
                       n_idx=default_idx):
    """
    Evaluate the nominal intensity sensitivity and fabrication robustness for a NON-PERIODIC stack.
    Uses a coarse-to-fine angular search to extract the maximum numerical sensitivity.

    Args:
        x (torch.Tensor): Input tensor containing the nominal thickness parameters.
        theta0 (float): Central incidence angle in degrees.
        range_theta (float): Angular range (+/- theta0) to search for the resonance dip.
        wl (torch.Tensor): Wavelength in nanometers.
        pol (str): Polarization ('s' or 'p').
        n_sampling (int): Number of points for the initial coarse angular sweep.
        n_fine_sampl (int): Number of points for the fine angular sweep inside the detected dip.
        threshold (float): Reflectivity threshold to define the Region of Interest (ROI).
        seed (int, optional): Random seed for reproducibility.
        error_perc (float): Standard deviation of the thickness perturbation (e.g., 0.04 for 4%).
        test_size (int): Number of perturbed realizations to generate.
        n_idx (list/tensor): Refractive indices.

    Returns:
        tuple: (-Nominal Sensitivity, Mean Absolute Percentage Error of Max Sensitivity).
               Returns a massive penalty if the resonance is completely lost.
    """
    # _______________ DEFINING ROIs (Coarse Search) _______________
    hasDip = True
    # Create the coarse angular grid (in radians)
    v_theta = torch.linspace(theta0 - range_theta, theta0 + range_theta, n_sampling) * torch.pi / 180.0 
    
    # Generate the statistical ensemble of perturbed structures
    T = generate_T_randVar_normal(x, seed=seed, test_size=test_size, error_perc=error_perc)
    M = generate_n_matrix(n_idx, L=T.shape[1], num_stack=T.shape[0])
    
    # Compute the coarse reflectivity spectra
    O = tmm(pol, M, T, v_theta, torch.tensor([wl]))
    R = O.get('R').squeeze(-1)  # shape: (num_stack, n_sampling)
    
    # Compute the nominal sensitivity at the exact target angle
    _ , sens_original = Finite_Difference_4or_multiple(x.unsqueeze(0), n_idx=n_idx, wl=torch.tensor([wl]), theta=torch.tensor([theta0*np.pi/180]))
    
    first_last_theta = []
    # Identify the angular interval (ROI) where reflectivity drops below the threshold
    for i in range(R.shape[0]):
        idx = torch.nonzero(R[i, :] < threshold).squeeze(1)
        if idx.numel() > 0:
            first_last_theta.append((v_theta[idx[0].item()], v_theta[idx[-1].item()]))
        else:
            hasDip = False
            
    if hasDip:
        intervals_theta = torch.tensor(first_last_theta)
        
        # ______________ CALCULATION SENSITIVITIES (Fine Search) _________
        # Create a highly dense angular grid specifically inside the dip for each stack
        v_theta_interval = np.linspace(intervals_theta[:,0], intervals_theta[:,1], n_fine_sampl)
        dSmax_dt_array = []
        
        for i in range(R.shape[0]):
            # Evaluate sensitivity on the fine grid (excluding substrate/superstrate padding)
            _, sens = Finite_Difference_4or_multiple(T[i, 1:-1].unsqueeze(0), n_idx=default_idx, wl=torch.tensor([wl]), theta=v_theta_interval[:,i])
            
            # Extract the absolute maximum numerical sensitivity inside the ROI
            S_max, _ = torch.max(sens, dim=1)
            
            # Calculate percentage variation relative to the nominal structure
            dSmax_dt = (S_max - sens_original.squeeze()) / sens_original.squeeze() * 100.0
            dSmax_dt_array.append(dSmax_dt.item())
            
        return -sens_original.squeeze().item(), np.abs(np.array(dSmax_dt_array)).mean()
    else:
        # Massive penalty if the BSW resonance is lost
        return -sens_original.squeeze().item(), 100000000.0
    

def dSmax_dthickness_randVar_periodic(x, 
                       theta0=60.0, 
                       v_theta=torch.linspace(59.0, 61.0, 1000)*torch.pi/180.0,
                       wl=torch.tensor([550.0]),
                       pol='s',
                       #n_sampling=500,
                       n_fine_sampl=1000,
                       threshold=0.97,
                       seed=None,
                       n_bilayers=5,
                       error_perc=0.04,
                       test_size=40,
                       n_idx=default_idx):
    """
    Evaluate the nominal intensity sensitivity and fabrication robustness for a PERIODIC stack.
    Expands a compressed parameter vector (bilayer + defect) into a full multilayer stack before evaluation.

    Args:
        x (list/array): Compressed parameters [d_low, d_high, d_defect].
        theta0 (float): Central incidence angle in degrees.
        v_theta (torch.Tensor): Pre-computed coarse angular grid in radians.
        wl (torch.Tensor): Wavelength in nanometers.
        pol (str): Polarization ('s' or 'p').
        n_fine_sampl (int): Number of points for the fine angular sweep inside the detected dip.
        threshold (float): Reflectivity threshold to define the Region of Interest (ROI).
        seed (int, optional): Random seed for reproducibility.
        n_bilayers (int): Number of times the [d_low, d_high] bilayer unit is repeated.
        error_perc (float): Standard deviation of the thickness perturbation.
        test_size (int): Number of perturbed realizations to generate.
        n_idx (list/tensor): Refractive indices.

    Returns:
        tuple: (-Nominal Sensitivity, Scaled Absolute Percentage Error).
               Note: The error is multiplied by test_size, effectively returning the sum of errors.
    """
    # _______________ DEFINING ROIs (Coarse Search) _______________
    hasDip = True
    
    # 1. Expand the periodic architecture
    x = torch.from_numpy(np.array(x, dtype=np.float64)).unsqueeze(0)
    bilayer_unit = x[:, 0:2]             # First two parameters are the repeating bilayer
    t_defect = x[:, -1:]                 # Last parameter is the termination layer (defect)
    periodic_part = bilayer_unit.repeat(1, n_bilayers) # Repeat the bilayer n_bilayers times
    x_final = torch.cat([periodic_part, t_defect], dim=1) # Assemble the full nominal stack
    
    # Generate the statistical ensemble of perturbed structures based on the full stack
    T = generate_T_randVar_normal(x_final, seed=seed, test_size=test_size, error_perc=error_perc)
    M = generate_n_matrix(n_idx, L=T.shape[1], num_stack=T.shape[0])
    
    # Compute the coarse reflectivity spectra
    O = tmm(pol, M, T, v_theta, torch.tensor([wl]))
    R = O.get('R').squeeze(-1)  # shape: (num_stack, n_sampling)
    
    # Compute the nominal sensitivity using the full unperturbed stack
    _ , sens_original = Finite_Difference_4or_multiple(x_final, n_idx=n_idx, wl=torch.tensor([wl]), theta=torch.tensor([theta0*np.pi/180]))
    
    first_last_theta = []
    # Identify the angular interval (ROI) for each perturbed stack
    for i in range(R.shape[0]):
        idx = torch.nonzero(R[i, :] < threshold).squeeze(1)
        if idx.numel() > 0:
            first_last_theta.append((v_theta[idx[0].item()], v_theta[idx[-1].item()]))
        else:
            hasDip = False
            
    if hasDip:
        intervals_theta = torch.tensor(first_last_theta)
        
        # ______________ CALCULATION SENSITIVITIES (Fine Search) _________
        # Create fine grids tailored to each stack's dip
        v_theta_interval = np.linspace(intervals_theta[:,0], intervals_theta[:,1], n_fine_sampl)
        dSmax_dt_array = []
        
        for i in range(R.shape[0]):
            # Evaluate sensitivity on the fine grid
            _, sens = Finite_Difference_4or_multiple(T[i, 1:-1].unsqueeze(0), n_idx=n_idx, wl=torch.tensor([wl]), theta=v_theta_interval[:,i])
            
            # Extract numerical maximum sensitivity
            S_max, _ = torch.max(sens, dim=1)
            
            # Compute percentage variation
            dSmax_dt = (S_max - sens_original.squeeze()) / sens_original.squeeze() * 100.0
            dSmax_dt_array.append(dSmax_dt.item())
            
        # Return negative sensitivity and the scaled error (sum of absolute errors across the ensemble)
        return -sens_original.squeeze().item(), np.abs(np.array(dSmax_dt_array)).mean()
    else:
        # Massive penalty if the BSW resonance is lost
        return -sens_original.squeeze().item(), 100000000.0



# ------------------------PROBLEM CLASS------------------------------
import pickle
from pathlib import Path
import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem

class MultiObjectiveProblem_unified(ElementwiseProblem):
    def __init__(
        self,
        unified_func,
        func_kwargs={},
        n_vars=2,
        xl=np.array([0, -5]),
        xu=np.array([5, 3]),
        **kwargs,
    ):
        super().__init__(
            n_var=n_vars,
            n_obj=2,           
            n_ieq_constr=0,    # No constraints
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.unified_func = unified_func
        self.func_kwargs = func_kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        # NOTE: pymoo passes 'x' as a 1D NumPy array. 
        # Since the objective function uses PyTorch, we convert it to a Tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # The evaluation function returns 2 objectives (-Sensitivity and FOM_R)
        res1, res2 = self.unified_func(x_tensor, **self.func_kwargs)

        # Assign the objective values to 'F'
        out["F"] = np.array([res1, res2], dtype=float)


class SaveToPickleCallback(Callback):
    """
    Callback to incrementally save the optimization history.
    It saves the population data every 'save_every' generations and at the final generation.
    """
    def __init__(self, filename, save_every=2, n_max_gen=None):
        super().__init__()
        self.filename = filename
        self.save_every = save_every
        self.n_max_gen = n_max_gen  # Required to ensure the last generation is saved
        
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize/clear the file
        with open(self.filename, "wb") as f:
            pass

    def notify(self, algorithm):
        gen = algorithm.n_gen
        
        # Condition: save if the generation is a multiple of save_every OR if it's the last one
        is_multiple = (gen % self.save_every == 0)
        is_last = (self.n_max_gen is not None and gen == self.n_max_gen)
        
        if is_multiple or is_last:
            pop = algorithm.pop
            
            data_to_save = {
                "gen": gen,
                "F": pop.get("F"),
                "X": pop.get("X")
            }
            
            # Incremental save (append binary)
            with open(self.filename, "ab") as f:
                pickle.dump(data_to_save, f)


class SaveToPickleCallback_noPOP(Callback):
    """
    Callback to incrementally save the optimization history (Objectives only).
    It saves 'F' every 'save_every' generations. The design variables 'X' are included 
    only at the final generation to save memory.
    """
    def __init__(self, filename, save_every=30, n_max_gen=None):
        super().__init__()
        self.filename = filename
        self.save_every = save_every
        self.n_max_gen = n_max_gen  
        
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize/clear the file
        with open(self.filename, "wb") as f:
            pass

    def notify(self, algorithm):
        gen = algorithm.n_gen
        
        is_multiple = (gen % self.save_every == 0)
        is_last = (self.n_max_gen is not None and gen == self.n_max_gen)
        
        if is_multiple or is_last:
            pop = algorithm.pop
            
            # Base dictionary: always saved when conditions are met
            data_to_save = {
                "gen": gen,
                "F": pop.get("F")
            }
            
            # Add 'X' (the population structural variables) only at the end
            if is_last:
                data_to_save["X"] = pop.get("X")
            
            # Incremental save (append binary)
            with open(self.filename, "ab") as f:
                pickle.dump(data_to_save, f)




class SingleObjectiveProblemVectorized(Problem):
    """
    A vectorized single-objective problem wrapper for pymoo optimization.
    
    This class is designed for performance: it evaluates the entire population 
    in a single batch rather than looping through individuals one by one.
    
    Requirements:
    - `obj_func`: Must accept a population matrix `x` with shape (pop_size, n_var) 
      and return objective values with shape (pop_size, 1).
    - `constr_funcs`: Any constraint functions must also operate in batch mode, 
      returning an array of shape (pop_size,) for the given constraint.
    """

    def __init__(
        self,
        obj_func,
        obj_kwargs=None,
        constr_funcs=[],
        constr_kwargs_list=None,
        n_vars=2,
        xl=np.array([0, -5]),
        xu=np.array([5, 3]),
        **kwargs,
    ):
        """
        Initialize the vectorized single-objective problem.

        Args:
            obj_func (callable): The main objective function to minimize.
            obj_kwargs (dict, optional): Keyword arguments to pass to the objective function.
            constr_funcs (list of callables, optional): List of inequality constraint functions.
            constr_kwargs_list (list of dicts, optional): Keyword arguments corresponding to each constraint function.
            n_vars (int): Number of design variables.
            xl (np.array): Lower bounds for the variables.
            xu (np.array): Upper bounds for the variables.
        """
        # The number of inequality constraints equals the number of provided constraint functions
        n_constraints = len(constr_funcs)

        super().__init__(
            n_var=n_vars,
            n_obj=1,
            n_ieq_constr=n_constraints,
            xl=xl,
            xu=xu,
            **kwargs,
        )

        # Store functions and their specific parameters
        self.obj_func = obj_func
        self.obj_kwargs = obj_kwargs if obj_kwargs is not None else {}
        self.constr_funcs = constr_funcs
        
        # If constraint kwargs are not provided, initialize an empty dict for each constraint
        self.constr_kwargs_list = (
            constr_kwargs_list
            if constr_kwargs_list is not None
            else [{} for _ in constr_funcs]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the population's objective and constraints.
        
        Args:
            x (np.ndarray): Population matrix with shape (pop_size, n_var).
            out (dict): Dictionary where results are stored for pymoo.
                - out["F"]: Objective values, expected shape (pop_size, 1).
                - out["G"]: Constraint values, expected shape (pop_size, n_constraints).
        """

        # --- Evaluate Objective ---
        f_vals = self.obj_func(x, **self.obj_kwargs)
        
        # --- Shape Validation (F) ---
        # Ensure the objective function returned a properly shaped column vector
        assert f_vals.shape[0] == x.shape[0] and f_vals.ndim == 2 and f_vals.shape[1] == 1, \
            f"obj_func must return a 2D array with shape (pop_size, 1). Got {f_vals.shape} instead."

        out["F"] = f_vals

        # --- Evaluate Constraints ---
        if self.constr_funcs:
            g_vals = []
            for func, kwargs_dict in zip(self.constr_funcs, self.constr_kwargs_list):
                # Evaluate the specific constraint across the entire population
                g = func(x, **kwargs_dict)  # Expected return shape: (pop_size,)

                # --- Shape Validation (G) ---
                assert g.shape[0] == x.shape[0], \
                    f"Constraint function must return an array of length pop_size={x.shape[0]}. Got {g.shape[0]} instead."

                # Convert (pop_size,) to (pop_size, 1) and append to our list
                g_vals.append(g.reshape(-1, 1))
            
            # Horizontally stack all constraint columns into a matrix of shape (pop_size, n_constraints)
            out["G"] = np.hstack(g_vals)