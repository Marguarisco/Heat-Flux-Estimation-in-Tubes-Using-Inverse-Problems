import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def copy_arrays(destination: np.ndarray, source: np.ndarray) -> None:
    """
    Copies the values from the source array to the destination array.

    Parameters:
    destination (np.ndarray): The array where values will be copied to.
    source (np.ndarray): The array from which values will be copied.
    """
    destination[:] = source[:]

@numba.jit(nopython=True, fastmath=True, cache=True)
def solve_tridiagonal_system(
    lower_diagonal: np.ndarray, 
    main_diagonal: np.ndarray, 
    upper_diagonal: np.ndarray, 
    rhs: np.ndarray, 
    size: int
) -> np.ndarray:
    """
    Solves a tridiagonal linear system using the Thomas algorithm.

    Parameters:
    lower_diagonal (np.ndarray): The lower diagonal of the tridiagonal matrix.
    main_diagonal (np.ndarray): The main diagonal of the tridiagonal matrix.
    upper_diagonal (np.ndarray): The upper diagonal of the tridiagonal matrix.
    rhs (np.ndarray): The right-hand side vector.
    size (int): The size of the system.

    Returns:
    np.ndarray: The solution vector.
    """
    c_prime = np.zeros(size - 1, dtype=np.float64)
    d_prime = np.zeros(size, dtype=np.float64)
    solution = np.zeros(size, dtype=np.float64)

    # Initialize the first elements
    inv_denom = 1.0 / main_diagonal[0]
    c_prime[0] = upper_diagonal[0] * inv_denom
    d_prime[0] = rhs[0] * inv_denom

    # Forward elimination
    for i in range(1, size - 1):
        inv_denom = 1.0 / (main_diagonal[i] - lower_diagonal[i - 1] * c_prime[i - 1])
        c_prime[i] = upper_diagonal[i] * inv_denom
        d_prime[i] = (rhs[i] - lower_diagonal[i - 1] * d_prime[i - 1]) * inv_denom

    # Last element of d_prime
    d_prime[size - 1] = (rhs[size - 1] - lower_diagonal[size - 2] * d_prime[size - 2]) / (
        main_diagonal[size - 1] - lower_diagonal[size - 2] * c_prime[size - 2])

    # Back substitution
    solution[-1] = d_prime[-1]
    for i in range(size - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]

    return solution

@numba.jit(nopython=True, fastmath=True, cache=True)
def solve_implicit_radial(
    current_temp: np.ndarray, 
    gamma_tt: np.ndarray, 
    main_diag_r: np.ndarray, 
    upper_diag_r: np.ndarray, 
    lower_diag_r: np.ndarray, 
    gamma_0: np.ndarray, 
    gamma_j: float, 
    external_temp: float, 
    angular_size: int, 
    radial_size: int, 
    new_temp: np.ndarray, 
    rhs_r: np.ndarray
) -> np.ndarray:
    """
    Solves the implicit radial step for each theta using the tridiagonal solver.

    Parameters:
    current_temp (np.ndarray): Current temperature matrix.
    gamma_tt (np.ndarray): Coefficient array for theta-direction terms.
    main_diag_r (np.ndarray): Main diagonal for radial tridiagonal systems.
    upper_diag_r (np.ndarray): Upper diagonal for radial tridiagonal systems.
    lower_diag_r (np.ndarray): Lower diagonal for radial tridiagonal systems.
    gamma_0 (np.ndarray): Boundary condition parameter at r=0.
    gamma_j (float): Boundary condition parameter at r=r_ext.
    external_temp (float): External temperature boundary condition.
    angular_size (int): Number of theta divisions.
    radial_size (int): Number of radial divisions.
    new_temp (np.ndarray): Array to store the new temperatures.
    rhs_r (np.ndarray): Right-hand side vector for radial systems.

    Returns:
    np.ndarray: Updated temperature matrix after radial implicit solve.
    """

    for j in range(angular_size):
        prev_j = (j - 1) % angular_size
        next_j = (j + 1) % angular_size

        # Boundary condition at r = 0
        rhs_r[0] = (-gamma_0[j] + gamma_tt[0] * current_temp[prev_j, 0]
                   + (1 - 2 * gamma_tt[0]) * current_temp[j, 0]
                   + gamma_tt[0] * current_temp[next_j, 0])

        # Internal points
        rhs_r[1:radial_size - 1] = (gamma_tt[1:radial_size - 1] * current_temp[prev_j, 1:radial_size - 1]
                              + (1 - 2 * gamma_tt[1:radial_size - 1]) * current_temp[j, 1:radial_size - 1]
                              + gamma_tt[1:radial_size - 1] * current_temp[next_j, 1:radial_size - 1])

        # Boundary condition at r = r_ext
        rhs_r[-1] = (-gamma_j * external_temp
                    + gamma_tt[-1] * current_temp[prev_j, -1]
                    + (1 - 2 * gamma_tt[-1]) * current_temp[j, -1]
                    + gamma_tt[-1] * current_temp[next_j, -1])

        # Solve the tridiagonal system
        new_temp[j, :] = solve_tridiagonal_system(lower_diag_r, main_diag_r, upper_diag_r, rhs_r, radial_size)

    return new_temp

@numba.jit(nopython=True, fastmath=True, cache=True)
def solve_implicit_theta(
    current_temp: np.ndarray, 
    gamma_r: np.ndarray, 
    gamma_rr: float, 
    gamma_tt: np.ndarray, 
    gamma_0: np.ndarray, 
    gamma_j: float, 
    external_temp: float, 
    angular_size: int, 
    radial_size: int, 
    new_temp: np.ndarray, 
    rhs_theta: np.ndarray, 
    main_diag_theta: np.ndarray, 
    aux_diag_theta: np.ndarray
) -> np.ndarray:
    """
    Solves the implicit theta step for each radial position using the tridiagonal solver.

    Parameters:
    current_temp (np.ndarray): Current temperature matrix.
    gamma_r (np.ndarray): Coefficient array for radial-direction terms.
    gamma_rr (float): Coefficient for radial diffusion.
    gamma_tt (np.ndarray): Coefficient array for theta-direction terms.
    gamma_0 (np.ndarray): Boundary condition parameter at r=0.
    gamma_j (float): Boundary condition parameter at r=r_ext.
    external_temp (float): External temperature boundary condition.
    angular_size (int): Number of theta divisions.
    radial_size (int): Number of radial divisions.
    new_temp (np.ndarray): Array to store the new temperatures.
    rhs_theta (np.ndarray): Right-hand side vector for theta systems.
    main_diag_theta (np.ndarray): Main diagonal for theta tridiagonal systems.
    aux_diag_theta (np.ndarray): Auxiliary diagonal for theta tridiagonal systems.

    Returns:
    np.ndarray: Updated temperature matrix after theta implicit solve.
    """
    
    for i in range(radial_size):
        radius_index = i % radial_size

        # Configure the diagonals for the tridiagonal system
        main_diag_theta.fill(1 + (2 * gamma_tt[radius_index]))
        aux_diag_theta.fill(-gamma_tt[radius_index])

        if i != 0 and i != radial_size - 1:
            rhs_theta[:] = (-gamma_r[radius_index] + gamma_rr) * current_temp[:, i - 1] \
                           + (1 - (2 * gamma_rr)) * current_temp[:, i] \
                           + (gamma_r[radius_index] + gamma_rr) * current_temp[:, i + 1]
        elif i == 0:  # Boundary condition at r=0
            rhs_theta[:] = -gamma_0[:] \
                           + (1 - (2 * gamma_rr)) * current_temp[:, i] \
                           + (2 * gamma_rr) * current_temp[:, i + 1]
        elif i == radial_size - 1:  # Boundary condition at r=r_ext
            rhs_theta[:] = -gamma_j * external_temp \
                           + (1 + gamma_j - (2 * gamma_rr)) * current_temp[:, i] \
                           + (2 * gamma_rr) * current_temp[:, i - 1]

        # Adjust boundary conditions for theta
        rhs_theta[0] += gamma_tt[radius_index] * current_temp[0, i]
        rhs_theta[-1] += gamma_tt[radius_index] * current_temp[-1, i]

        # Solve the tridiagonal system
        new_temp[:, i] = solve_tridiagonal_system(
            aux_diag_theta, main_diag_theta, aux_diag_theta, rhs_theta, angular_size
        )

    return new_temp

@numba.jit(nopython=True, fastmath=True)
def adi_method(simulation_time, args):

    temperature_args, spacial_args, material_args, flux_args = args

    T_inner, T_outer, T_tube = temperature_args
    dr, radial_size, radial_space, dtheta, angular_size, angular_space = spacial_args
    thermal_conductivity, specific_heat, density = material_args
    heat_flux, h_conv = flux_args

    thermal_diffusivity = thermal_conductivity / (specific_heat * density)  

    dt = 0.1

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # TEMPERATURE MATRICES
    current_temp = np.ones((angular_size, radial_size)) * T_tube
    new_temp = np.zeros_like(current_temp)

    # AUXILIARY VECTORS
    rhs_r = np.zeros(radial_size, dtype=np.float64)
    rhs_theta = np.zeros(angular_size, dtype=np.float64)

    # Coefficients for the ADI method
    gamma_r = thermal_diffusivity * dt / (4 * dr * radial_space)
    gamma_rr = thermal_diffusivity * dt / (2 * (dr ** 2))
    gamma_tt = thermal_diffusivity * dt / ((dtheta ** 2) * (2 * radial_space ** 2))

    # Boundary condition parameters
    beta_0 = (2 * dr * heat_flux) / thermal_conductivity
    gamma_0 = (gamma_r[0] - gamma_rr) * beta_0

    beta_j = (2 * dr * h_conv) / thermal_conductivity
    gamma_j = (-gamma_rr - gamma_r[-1]) * beta_j 

    # Diagonals for the radial direction
    lower_diag_r = -gamma_rr * np.ones(radial_size - 1)
    main_diag_r = np.ones(radial_size) * (1 + 2 * gamma_rr)
    main_diag_r[-1] = 1 + 2 * gamma_rr - gamma_j
    upper_diag_r = lower_diag_r - gamma_r[:radial_size - 1]
    upper_diag_r[0] = -2 * gamma_rr  # Adjust for boundary condition at r=0
    lower_diag_r = lower_diag_r + gamma_r[1:radial_size]
    lower_diag_r[-1] = -2 * gamma_rr  # Adjust for boundary condition at r=r_ext

    # Diagonals for the theta direction
    main_diag_theta = np.ones(angular_size)
    aux_diag_theta = np.zeros(angular_size - 1)

    # History of external temperatures
    time_step = 1
    T_ext_history = np.zeros((simulation_time, angular_size, radial_size), dtype=np.float64)
    T_ext_history[0, :, :] = current_temp

    while time_step < simulation_time:
        

        for _ in range(1):

            # Solve the implicit radial step
            new_temp = solve_implicit_radial(
                current_temp, gamma_tt, main_diag_r, upper_diag_r, lower_diag_r, 
                gamma_0, gamma_j, T_outer, angular_size, radial_size, new_temp, rhs_r
            )
            copy_arrays(current_temp, new_temp)
            
            # Solve the implicit theta step
            new_temp = solve_implicit_theta(
                current_temp, gamma_r, gamma_rr, gamma_tt, gamma_0, gamma_j, 
                T_outer, angular_size, radial_size, new_temp, rhs_theta, 
                main_diag_theta, aux_diag_theta
            )
            copy_arrays(current_temp, new_temp)

        # Record the external temperature at the boundary
        T_ext_history[time_step, :, :] = current_temp
        time_step += 1

    return T_ext_history[:time_step, :], dt

if __name__ == '__main__':
    import pandas as pd

    '''Nr = 9
    Ntheta = 80
    N_max = 19000

    # Define the theta distribution
    Theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False) 

    heat_flux = ((-2000.0) * (Theta / np.pi) ** 2) + 2000.0
    periodic_heat_flux = np.zeros((N_max, Ntheta), dtype=np.float64)

    for i in range(N_max):
        periodic_heat_flux[i] = heat_flux * (1 + np.sin(np.pi * i / N_max))

    # Execute the ADI method
    T_ext_history = ADIMethod(periodic_heat_flux, Nr, Ntheta, N_max)

    # Create a DataFrame to store the results
    df = pd.DataFrame(T_ext_history, columns=[f'Theta {i}' for i in range(Ntheta)])
    csv_filename = f'Transient Periodic File/Temperature_Boundary_External_{Nr}_{Ntheta}_{N_max}.csv'

    # Save the results to the CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")'''

