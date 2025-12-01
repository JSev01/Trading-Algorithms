import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from JumpDiffusionML import plot_model

# Read data
data = pd.read_csv('data/price_history.csv', header=None)
dates = pd.to_datetime(data.iloc[:, 0].values)
prices = data.iloc[:, 1].values

# Estimated parameters (actual estimated parameters from the model, just copied and pasted to save time)
gamma = 0.238989
sigma = 0.276186
lambda_ = 33.775186
alpha = -0.003033
beta = 0.050050

# Simulation parameters
n_paths = 500       # Number of simulation paths
delta_t = 1/252
T = 2               # 2 year forecast
n_steps = int(T / delta_t)

# Function to simulate price paths
def simulate_merton_paths(S0, gamma, sigma, lambda_, alpha, beta, n_paths, n_steps, delta_t):
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0  # Set initial price
    
    for i in range(n_paths):
        for j in range(1, n_steps + 1):
            # Normal random numbers for diffusion
            Z = np.random.normal(0, 1)
            # Poisson random number for jump count
            N = np.random.poisson(lambda_ * delta_t)
            # Generate jump sizes if any jumps occur
            if N > 0:
                Y = np.random.normal(alpha, beta, N).sum()
            else:
                Y = 0
            # Log returns
            log_return = (gamma - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z + Y
            paths[i, j] = paths[i, j-1] * np.exp(log_return)
    return paths

# Last price from historical data as starting point
S0 = prices[-1]

# Future dates
last_date = dates[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps + 2)]

# Simulate price paths
price_paths = simulate_merton_paths(S0, gamma, sigma, lambda_, alpha, beta, n_paths, n_steps, delta_t)

# Statistics from simulation
mean_path = np.mean(price_paths, axis=0)
median_path = np.median(price_paths, axis=0)

plt.figure(figsize=(14, 8))
plt.plot(dates, prices, color='black', label='Historical Prices')
plt.plot(future_dates, mean_path, color='blue', label='Mean Forecast')
plt.plot(future_dates, median_path, color='green', label='Median Forecast')

for i in range(min(10, n_paths)):
    plt.plot(future_dates, price_paths[i], alpha=0.1, color='gray')

plt.title('Merton Jump Diffusion Model: 2-Year Price Forecast', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save new forecast to CSV
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Mean_Forecast': mean_path,
    'Median_Forecast': median_path
})

forecast_df.to_csv('price_forecast_2years.csv', index=False)
print(f"Starting price: {S0:.2f}")
print(f"Forecasted price (mean) after 2 years: {mean_path[-1]:.2f}")
print(f"Forecasted price (median) after 2 years: {median_path[-1]:.2f}")

plt.savefig('price_forecast_plot.png', dpi=300)
plt.show()