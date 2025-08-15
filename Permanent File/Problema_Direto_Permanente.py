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
    psi_tt: np.ndarray, 
    main_diag_r: np.ndarray, 
    upper_diag_r: np.ndarray, 
    lower_diag_r: np.ndarray, 
    gamma_0: np.ndarray, 
    gamma_j: float, 
    external_temp: float, 
    num_theta: int, 
    num_r: int, 
    new_temp: np.ndarray, 
    rhs_r: np.ndarray
) -> np.ndarray:
    """
    Solves the implicit radial step for each theta using the tridiagonal solver.

    Parameters:
    current_temp (np.ndarray): Current temperature matrix.
    psi_tt (np.ndarray): Coefficient array for theta-direction terms.
    main_diag_r (np.ndarray): Main diagonal for radial tridiagonal systems.
    upper_diag_r (np.ndarray): Upper diagonal for radial tridiagonal systems.
    lower_diag_r (np.ndarray): Lower diagonal for radial tridiagonal systems.
    gamma_0 (np.ndarray): Boundary condition parameter at r=0.
    gamma_j (float): Boundary condition parameter at r=r_ext.
    external_temp (float): External temperature boundary condition.
    num_theta (int): Number of theta divisions.
    num_r (int): Number of radial divisions.
    new_temp (np.ndarray): Array to store the new temperatures.
    rhs_r (np.ndarray): Right-hand side vector for radial systems.

    Returns:
    np.ndarray: Updated temperature matrix after radial implicit solve.
    """

    for j in range(num_theta):
        prev_j = (j - 1) % num_theta
        next_j = (j + 1) % num_theta

        # Boundary condition at r = 0
        rhs_r[0] = (
            -gamma_0[j]
            + psi_tt[0] * current_temp[prev_j, 0]
            + (1 - 2 * psi_tt[0]) * current_temp[j, 0]
            + psi_tt[0] * current_temp[next_j, 0]
        )

        # Internal points
        rhs_r[1:num_r - 1] = (
            psi_tt[1:num_r - 1] * current_temp[prev_j, 1:num_r - 1]
            + (1 - 2 * psi_tt[1:num_r - 1]) * current_temp[j, 1:num_r - 1]
            + psi_tt[1:num_r - 1] * current_temp[next_j, 1:num_r - 1]
        )

        # Boundary condition at r = r_ext
        rhs_r[-1] = (
            -gamma_j * external_temp
            + psi_tt[-1] * current_temp[prev_j, -1]
            + (1 - 2 * psi_tt[-1]) * current_temp[j, -1]
            + psi_tt[-1] * current_temp[next_j, -1]
        )

        # Solve the tridiagonal system using Thomas algorithm
        new_temp[j, :] = solve_tridiagonal_system(lower_diag_r, main_diag_r, upper_diag_r, rhs_r, num_r)

    return new_temp

@numba.jit(nopython=True, fastmath=True, cache=True)
def solve_implicit_theta(
    current_temp: np.ndarray, 
    psi_r: np.ndarray, 
    psi_rr: float, 
    psi_tt: np.ndarray, 
    gamma_0: np.ndarray, 
    gamma_j: float, 
    external_temp: float, 
    num_theta: int, 
    num_r: int, 
    new_temp: np.ndarray, 
    rhs_theta: np.ndarray, 
    main_diag_theta: np.ndarray, 
    aux_diag_theta: np.ndarray
) -> np.ndarray:
    """
    Solves the implicit theta step for each radial position using the tridiagonal solver.

    Parameters:
    current_temp (np.ndarray): Current temperature matrix.
    psi_r (np.ndarray): Coefficient array for radial-direction terms.
    psi_rr (float): Coefficient for radial diffusion.
    psi_tt (np.ndarray): Coefficient array for theta-direction terms.
    gamma_0 (np.ndarray): Boundary condition parameter at r=0.
    gamma_j (float): Boundary condition parameter at r=r_ext.
    external_temp (float): External temperature boundary condition.
    num_theta (int): Number of theta divisions.
    num_r (int): Number of radial divisions.
    new_temp (np.ndarray): Array to store the new temperatures.
    rhs_theta (np.ndarray): Right-hand side vector for theta systems.
    main_diag_theta (np.ndarray): Main diagonal for theta tridiagonal systems.
    aux_diag_theta (np.ndarray): Auxiliary diagonal for theta tridiagonal systems.

    Returns:
    np.ndarray: Updated temperature matrix after theta implicit solve.
    """
    for i in range(num_r):
        radius_index = i % num_r

        # Configure the diagonals for the tridiagonal system
        main_diag_theta.fill(1 + (2 * psi_tt[radius_index]))
        aux_diag_theta.fill(-psi_tt[radius_index])

        if i != 0 and i != num_r - 1:
            rhs_theta[:] = (
                (-psi_r[radius_index] + psi_rr) * current_temp[:, i - 1]
                + (1 - (2 * psi_rr)) * current_temp[:, i]
                + (psi_r[radius_index] + psi_rr) * current_temp[:, i + 1]
            )
        elif i == 0:  # Boundary condition at r=0
            rhs_theta[:] = (
                -gamma_0[:]
                + (1 - (2 * psi_rr)) * current_temp[:, i]
                + (2 * psi_rr) * current_temp[:, i + 1]
            )
        elif i == num_r - 1:  # Boundary condition at r=r_ext
            rhs_theta[:] = (
                -gamma_j * external_temp
                + (1 + gamma_j - (2 * psi_rr)) * current_temp[:, i]
                + (2 * psi_rr) * current_temp[:, i - 1]
            )

        # Adjust boundary conditions for theta
        rhs_theta[0] += psi_tt[radius_index] * current_temp[0, i]
        rhs_theta[-1] += psi_tt[radius_index] * current_temp[-1, i]

        # Solve the tridiagonal system using Thomas algorithm
        new_temp[:, i] = solve_tridiagonal_system(
            aux_diag_theta, main_diag_theta, aux_diag_theta, rhs_theta, num_theta
        )

    return new_temp

@numba.jit(nopython=True, fastmath=True, cache=True)
def ADIMethod(
    heat_flux: np.ndarray,
    num_r: int = 9, 
    num_theta: int = 20, 
    max_time_steps: int = 1e10
) -> np.ndarray:
    """
    Executes the Alternating Direction Implicit (ADI) method to solve the diffusion equation.

    Parameters:
    heat_flux (np.ndarray): Heat flux distribution as a function of theta. 
    num_r (int): Number of radial divisions. Default is 9.
    num_theta (int): Number of angular divisions. Default is 20.
    max_time_steps (int): Maximum number of time steps.

    Returns:
    np.ndarray: Final temperature distribution matrix.
    """
    # PHYSICAL PARAMETERS
    r_inner, r_outer = 0.1, 0.15  # Inner and outer rad (meters)
    T_inner, T_outer, T_tube = 300.0, 300.0, 300.0  # Temperatures in Kelvin
    h_conv = 25.0  # Convective heat transfer coefficient (W/m²K)
    thermal_conductivity = 201.0  # Thermal conductivity (W/mK)
    specific_heat = 900.0  # Specific heat capacity (J/kgK)
    density = 2700.0  # Density (kg/m³)
    thermal_diffusivity = thermal_conductivity / (specific_heat * density)  # Thermal diffusivity (m²/s)

    # SPACING
    dr = (r_outer - r_inner) / (num_r - 1)  # Radial step size
    dtheta = 2 * np.pi / num_theta  # Angular step size (radians)
    dt = 0.1  # Time step size (seconds)

    # MESH GRID
    radii = np.linspace(r_inner, r_outer, num_r)

    # TEMPERATURE MATRICES
    current_temp = np.ones((num_theta, num_r)) * T_tube
    new_temp = np.zeros_like(current_temp)
    previous_temp = np.zeros_like(current_temp)

    # AUXILIARY VECTORS
    rhs_r = np.zeros(num_r, dtype=np.float64)
    rhs_theta = np.zeros(num_theta, dtype=np.float64)

    # Coefficients for the ADI method
    psi_r = thermal_diffusivity * dt / (4 * dr * radii)
    psi_rr = thermal_diffusivity * dt / (2 * (dr ** 2))
    psi_tt = thermal_diffusivity * dt / (dtheta ** 2) / (2 * radii ** 2)

    # Boundary condition parameters
    beta_0 = (2 * dr * heat_flux) / thermal_conductivity
    gamma_0 = (psi_r[0] - psi_rr) * beta_0

    beta_j = (2 * dr * h_conv) / thermal_conductivity
    gamma_j = (-psi_rr - psi_r[-1]) * beta_j 

    time_step = 0
    tol_steady_state = 1e-4
    diff = 1.0

    # Diagonals for the radial direction
    aux_diag_r = (-psi_rr * np.ones(num_r - 1))

    main_diag_r = np.ones(num_r) * (1 + (2 * psi_rr))
    main_diag_r[-1] = (1 + (2 * psi_rr) - gamma_j)

    upper_diag_r = aux_diag_r - psi_r[:num_r - 1]
    upper_diag_r[0] = (-2 * psi_rr)  # Adjust for boundary condition at r=0

    lower_diag_r = aux_diag_r + psi_r[1:num_r]
    lower_diag_r[-1] = (-2 * psi_rr)  # Adjust for boundary condition at r=r_ext


    # Diagonals for the angular direction
    main_diag_theta = np.ones(num_theta)
    aux_diag_theta = np.zeros(num_theta - 1)

    # Iterative process until steady state or maximum time steps
    while time_step <= max_time_steps and diff > tol_steady_state:
        time_step += 1

        # Solve the implicit radial step
        new_temp = solve_implicit_radial(
            current_temp, psi_tt, main_diag_r, upper_diag_r, lower_diag_r, 
            gamma_0, gamma_j, T_outer, num_theta, num_r, new_temp, rhs_r
        )
        copy_arrays(current_temp, new_temp)

        # Solve the implicit theta step
        new_temp = solve_implicit_theta(
            current_temp, psi_r, psi_rr, psi_tt, gamma_0, gamma_j, 
            T_outer, num_theta, num_r, new_temp, rhs_theta, 
            main_diag_theta, aux_diag_theta
        )
        copy_arrays(current_temp, new_temp)
        
        # Calculate the maximum difference for steady state
        diff = np.max(np.abs(current_temp - previous_temp))
        copy_arrays(previous_temp, current_temp)

    
    return current_temp

if __name__ == '__main__':
    import pandas as pd

    Nr = 25
    Ntheta = 3
    max_time_steps = 6000  # Converted to integer

    # Define the theta distribution
    Theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False) 

    # Define the heat source as a quadratic function of Theta
    #heat_flux = ((-2000.0) * (Theta / np.pi) ** 2) + 2000.0
    heat_flux = np.ones(Ntheta) * 1000

    # Execute the ADI method
    final_temperature = ADIMethod(heat_flux, Nr, Ntheta, max_time_steps)

    # Create a DataFrame to store the results
    df = pd.DataFrame(final_temperature)
    csv_filename = f'Permanent File/T_simulated_{Nr}_{Ntheta}.csv'

    # Save the results to the CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

