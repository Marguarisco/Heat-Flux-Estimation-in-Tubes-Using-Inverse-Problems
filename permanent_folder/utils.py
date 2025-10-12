import numpy as np
from numba import njit
from direct_problem import ADIMethod
import os

@njit
def tikhonov_regularization(heat_flux: np.ndarray, use_differences: bool = True) -> float:
    if not use_differences:
        return np.sum(heat_flux ** 2)
    else:
        diffs = np.diff(heat_flux)
        return np.sum(diffs ** 2)
@njit
def calculate_jacobian(heat_flux: np.ndarray, T_estimated: np.ndarray, historical_temperatures: np.ndarray, delta: float) -> np.ndarray:
    
    q_params = len(heat_flux)
    m_params = len(T_estimated)
    J = np.zeros((m_params, q_params), dtype=np.float64)

    for j in range(q_params):
        dheat_flux = heat_flux[j] * delta
        
        J[:,j] = (historical_temperatures[j] - T_estimated) / dheat_flux #dT/dq

    return J

def load_or_generate_random_values(mesh_size: int, filename: str = None) -> np.ndarray:

    if os.path.exists(filename):

        values = np.load(filename)[-1]
        print(f"Loaded array from {filename} with shape {values.shape}")
    else:
        values = np.random.normal(0, 1, mesh_size)
        np.save(filename, values)
        print(f"Generated and saved new array to {filename} with shape {values.shape}")
    return values

def initializer():
    """
    Initializer function to precompute ADIMethod for caching purposes.
    """
    q_dummy = np.ones(20, dtype=np.float64) * 100.0
    ADIMethod(q_dummy)

def run_experiment(filename, radial_size, angular_size, max_simulation_time, dt):
    theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False) 

    heat_flux = ((-2000.0) * (theta / np.pi) ** 2) + 2000.0

    estimated_temperature = ADIMethod(heat_flux, radial_size, angular_size, max_simulation_time, dt)

    np.savez_compressed(
        filename,
        heat_flux = heat_flux,
        estimated_temperature = estimated_temperature
    )