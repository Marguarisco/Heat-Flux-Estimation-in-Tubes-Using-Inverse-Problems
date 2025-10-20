import numpy as np
from numba import njit
from scipy.integrate import simpson as simps
from typing import Tuple
from direct_problem import ADIMethod
import os

@njit
def tikhonov_regularization(parameters: np.ndarray, use_differences: bool = True) -> float:

    if not use_differences:
        return np.sum(parameters ** 2)
    else:
        diffs = np.diff(parameters)
        return np.sum(diffs ** 2)
    
@njit
def calculate_jacobian(parameters: np.ndarray, T_estimated: np.ndarray, historical_temperatures: np.ndarray, delta: float) -> np.ndarray:
    parameters_len = historical_temperatures.shape[0]
    iteration  = T_estimated.shape[0]
    pertu_len = T_estimated.shape[1]

    
    J = np.zeros((iteration, pertu_len, parameters_len), dtype=np.float64)

    for i in range(iteration):
        for j in range(parameters_len):
            dp = parameters[j] * delta

            J[i, :, j] = (historical_temperatures[j, i] - T_estimated[i]) / dp #dT/dp
    
    return J

def calculate_step(step: int, J: np.ndarray, direcao_descida: np.ndarray, T_measured: np.ndarray, T_estimated: np.ndarray, experiment_time: int, alpha_regul: int, parameters: np.ndarray) -> float:

    ddt_dx = np.tensordot(J, direcao_descida, axes=([2], [0])) 
    diff_temp = T_measured - T_estimated  
    
    integral_term = np.sum((diff_temp - step * ddt_dx) * (-ddt_dx), axis=1) 

    integral_result = simps(integral_term, x=np.arange(experiment_time))

    diff_p = np.diff(parameters) 
    diff_d = np.diff(direcao_descida, axis=0) 

    regularization_term = experiment_time * alpha_regul * np.sum((diff_p + (step * diff_d)) * diff_d)

    return integral_result + regularization_term

def load_or_generate_random_values(shape: Tuple[int, int], filename: str = None) -> np.ndarray:

    if filename is None:
        filename = f'random_values_{shape[0]}x{shape[1]}.npy'

    if os.path.exists(filename):
        values = np.load(filename)
        print(f"Loaded array from {filename} with shape {values.shape}")
    else:
        values = np.random.normal(0, 1, shape)
        np.save(filename, values)
        print(f"Generated and saved new array to {filename} with shape {values.shape}")
    return values

def initializer():
    q_dummy = np.ones(20, dtype=np.float64) * 100.0
    ADIMethod(q_dummy)

def run_experiment(filename, radial_size, angular_size, total_simulation_time, dt):
    theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False) 

    heat_flux = ((-2000.0) * (theta / np.pi) ** 2) + 4000.0

    periodic_heat_flux = np.zeros((total_simulation_time, angular_size), dtype=np.float64)

    for i in range(total_simulation_time):
        periodic_heat_flux[i] = heat_flux * (1 + np.sin(np.pi * i / total_simulation_time))

    estimated_temperature = ADIMethod(periodic_heat_flux, radial_size, angular_size, total_simulation_time, dt)

    np.savez_compressed(
        filename,
        heat_flux = heat_flux,
        estimated_temperature = estimated_temperature
    )