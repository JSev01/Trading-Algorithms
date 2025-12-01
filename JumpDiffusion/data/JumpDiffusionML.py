import pandas as pd
import numpy as np
from scipy.stats import norm
import math
from scipy.optimize import minimize
from scipy.stats import norm as normal_dist
import matplotlib.pyplot as plt
import seaborn as sns


# Calculate the PDF of log returns under Merton Jump Diffusion Model
def merton_pdf(x, params, delta=1/252, tol=1e-3):
    gamma, sigma, lambda_, alpha, beta = params
    # Gamma = drift term. Sigma = volatility. Lambda = intensity. Alpha = mean. beta = stdev. delta = time increment
    density = 0.0
    prev_density = -np.inf  # Previous density
    n = 0  # Start with zero jumps
    
    while True:
        jump_prob = np.exp(-lambda_ * delta) * (lambda_ * delta)**n / math.factorial(n)  # PMF of Nt
        mean = (gamma - 0.5 * sigma**2) * delta + n * alpha
        var = sigma**2 * delta + n * beta**2
        curr_density = jump_prob * norm.pdf(x, mean, np.sqrt(var))
        density += curr_density
        # change in density below threshold
        if abs(curr_density - prev_density) < tol:
            break
        prev_density = curr_density
        n += 1
    return density


# Calculate the negative log-likelihood for the Merton Jump Diffusion Model
def negative_log_likelihood(params, log_returns, delta=1/252):
    gamma, sigma, lambda_, alpha, beta = params
    # Gamma = drift term. Sigma = volatility. Lambda = intensity. Alpha = mean. beta = stdev. delta = time increment
    if sigma <= 0 or lambda_ < 0 or beta <= 0:
        return np.inf
    
    log_likelihood = 0.0
    for r in log_returns:
        density = merton_pdf(r, params, delta)
        if density > 0:
            log_likelihood += np.log(density)
        else:
            return np.inf
    return -log_likelihood


# Estimate Merton model parameters using MLE
def estimate_parameters(log_returns, delta=1/252):
    # Initial parameters, just a guess for now
    initial_params = [0.01, 0.1, 1.0, 0.0, 0.1]
    bounds = [
        (None, None),   # gamma
        (1e-6, None),   # sigma>0
        (0.0, None),    # lambda >= 0
        (None, None),   # alpha
        (1e-6, None)    # beta > 0
    ]

    # Perform optimization
    result = minimize(
        negative_log_likelihood,  # Function to minimize
        initial_params,  # Starting point for optmiziation
        args=(log_returns, delta),  # Additional arguments
        bounds=bounds,  # Make sure we stay in the specified bounds
        method='L-BFGS-B',  # Optimization algorithm (suitable for bounded problems)
    )
    return result.x, result  # Two values: estimated parameters, results object (with more information)


# Calculate Standard Errors
def compute_standard_errors(params, log_returns, delta=1/252):
    def hessian(params):
        n = len(params)
        H = np.zeros((n, n))
        epsilon = 1e-5

        for i in range(n):
            for j in range(n):
                # First partial derivative
                params_ij = params.copy()
                params_ij[i] += epsilon
                params_ij[j] += epsilon

                params_i = params.copy()
                params_i[i] += epsilon

                params_j = params.copy()
                params_j[j] += epsilon

                # Second partial derivative of likelihood
                H[i, j] = (negative_log_likelihood(params_ij, log_returns, delta) - 
                           negative_log_likelihood(params_i, log_returns, delta) -
                           negative_log_likelihood(params_j, log_returns, delta) +
                           negative_log_likelihood(params, log_returns, delta)) / (epsilon**2)
        return H
    H = hessian(params)
    try:
        cov_matrix = np.linalg.inv(H)  # (Hessian)^-1
        std_errors = np.sqrt(np.diag(cov_matrix))
        return std_errors
    except np.linalg.LinAlgError:
        return np.array([np.nan] * len(params))


# Hypthesis testing
def test_hypothesis(params, std_errors):
    lambda_ = params[2]
    alpha = params[3]
    se_lambda = std_errors[2]
    se_alpha = std_errors[3]

    # Test statistics
    z_lambda = lambda_ / se_lambda
    z_alpha = alpha / se_alpha

    # P values for a two sided test
    p_lambda = 2 * (1 - normal_dist.cdf(abs(z_lambda)))
    p_alpha = 2 * (1 - normal_dist.cdf(abs(z_alpha)))
    
    return {
        'lambda': {
            'parameter': lambda_,
            'std_error': se_lambda,
            'z_statistic': z_lambda,
            'p_value': p_lambda
        },
        'alpha': {
            'parameter': alpha,
            'std_error': se_alpha,
            'z_statistic': z_alpha,
            'p_value': p_alpha
        }
    }


# Checking efficiency of the likelihood calculation
def merton_pdf_vectorized(x_values, params, delta_t=1/252, tol=1e-3):
    gamma, sigma, lambda_, alpha, beta = params
    densities = np.zeros_like(x_values, dtype=float)
    n = 0  # Start with zero jumps
    prev_density = np.zeros_like(x_values, dtype=float)  # Initialize previous density to zero
    
    while True:
        jump_prob = np.exp(-lambda_ * delta_t) * (lambda_ * delta_t)**n / math.factorial(n)
        mean_n = (gamma - 0.5 * sigma**2) * delta_t + n * alpha
        std_n = np.sqrt(sigma**2 * delta_t + n * beta**2)

        curr_density = jump_prob * norm.pdf(x_values, mean_n, std_n)

        densities += curr_density
        if np.max(np.abs(curr_density - prev_density)) < tol:
            break
        prev_density = curr_density
        n += 1
    
    return densities


def plot_model(log_returns, params, delta_t=1/252):
    plt.figure(figsize=(12, 8))
    sns.histplot(log_returns, bins=50, stat='density', alpha=0.6, label='Observed Returns')

    x_range = np.linspace(min(log_returns), max(log_returns), 1000)
    pdf_values = merton_pdf_vectorized(x_range, params, delta_t)
    
    plt.plot(x_range, pdf_values, 'r-', linewidth=2, label='Fitted Merton Model')
    
    # Normal distribution for comparison
    mean_returns = np.mean(log_returns)
    std_returns = np.std(log_returns)
    normal_pdf = norm.pdf(x_range, mean_returns, std_returns)

    plt.plot(x_range, normal_pdf, 'g--', linewidth=2, label='Normal Distribution') 
    plt.title('Merton Jump-Diffusion Model Fit', fontsize=16)
    plt.xlabel('Log Returns', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('merton_model_fit.png', dpi=300)
    plt.show()


def main():
    # Loading data
    data = pd.read_csv('data/price_history.csv', header=None)
    dates = pd.to_datetime(data.iloc[:, 0].values)
    prices = data.iloc[:, 1].values
    log_returns = np.log(prices[1:] / prices[:-1])

    # Estimating parameters
    params, opt_result = estimate_parameters(log_returns)
    gamma, sigma, lambda_, alpha, beta = params

    # Computing standard errors
    std_errors = compute_standard_errors(params, log_returns)

    # Performing hypothesis tests
    test_results = test_hypothesis(params, std_errors)

    print("\nMerton Jump-Diffusion Model Parameter Estimates:")
    print(f"gamma: {gamma:.6f} ± {std_errors[0]:.6f}")
    print(f"sigma: {sigma:.6f} ± {std_errors[1]:.6f}")
    print(f"lambda: {lambda_:.6f} ± {std_errors[2]:.6f}")
    print(f"alpha: {alpha:.6f} ± {std_errors[3]:.6f}")
    print(f"beta: {beta:.6f} ± {std_errors[4]:.6f}")
    
    print("\nHypothesis Tests:")
    print(f"H0: lambda = 0, p-value: {test_results['lambda']['p_value']:.6f}")
    print(f"H0: alpha = 0, p-value: {test_results['alpha']['p_value']:.6f}")
    
    print("\nOptimization Details:")
    print(f"Success: {opt_result.success}")
    print(f"Number of function evaluations: {opt_result.nfev}")
    plot_model(log_returns, params)


if __name__ == "__main__":
    main()