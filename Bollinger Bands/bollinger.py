import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download S&P 500 data (using SPY ETF as proxy)
print("Downloading S&P 500 data...")
df = yf.download("SPY", interval="1h", period="2y")
df = df[['Close']]
df.reset_index(inplace=True)
df.rename(columns={'Datetime':'Time'}, inplace=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print(f"Downloaded {len(df)} hours of data")

# === BOLLINGER BANDS CALCULATION ===
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14

# Middle band (SMA)
df["sma"] = df["Close"].rolling(window=BB_PERIOD).mean()

# Standard deviation
df["std"] = df["Close"].rolling(window=BB_PERIOD).std()

# Upper and lower bands
df["upper_band"] = df["sma"] + (BB_STD * df["std"])
df["lower_band"] = df["sma"] - (BB_STD * df["std"])

# === RSI CALCULATION ===
# Calculate price changes
delta = df["Close"].diff()

# Separate gains and losses
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# Calculate average gain and loss
avg_gain = gain.rolling(window=RSI_PERIOD).mean()
avg_loss = loss.rolling(window=RSI_PERIOD).mean()

# Calculate RS and RSI
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# === GENERATE SIGNALS ===
df["position"] = 0  # 0 = no position, 1 = long

# Entry condition: Price touches lower band AND RSI < 30 (oversold)
entry_condition = (df["Close"] <= df["lower_band"]) & (df["rsi"] < 30)

# Exit condition: Price crosses back above middle band (mean reversion complete)
exit_condition = df["Close"] >= df["sma"]

# Apply signals
for i in range(len(df)):
    if i == 0:
        continue
    
    # If we're not in a position, check for entry
    if df.loc[i-1, "position"] == 0:
        if entry_condition.iloc[i]:
            df.loc[i, "position"] = 1
    # If we're in a position, check for exit
    else:
        if exit_condition.iloc[i]:
            df.loc[i, "position"] = 0
        else:
            df.loc[i, "position"] = df.loc[i-1, "position"]  # Hold position

df["position_change"] = df["position"].diff()

# === RISK MANAGEMENT ===
stop_loss = 0.02      # 2%
take_profit = 0.04    # 4% (2:1 reward-risk)

# === TRADE SIMULATION ===
trades = []
in_trade = False

for i, row in df.iterrows():
    if not in_trade:
        # Check for entry signal
        if row["position_change"] == 1:
            entry_price = row["Close"]
            entry_time = row["Time"]
            in_trade = True
    else:
        current_price = row["Close"]
        
        # Stop-loss
        if current_price <= entry_price * (1 - stop_loss):
            trades.append({
                "EntryTime": entry_time,
                "EntryPrice": entry_price,
                "ExitTime": row["Time"],
                "ExitPrice": current_price,
                "ExitType": "StopLoss"
            })
            in_trade = False
        # Take-profit
        elif current_price >= entry_price * (1 + take_profit):
            trades.append({
                "EntryTime": entry_time,
                "EntryPrice": entry_price,
                "ExitTime": row["Time"],
                "ExitPrice": current_price,
                "ExitType": "TakeProfit"
            })
            in_trade = False
        # Mean reversion exit
        elif row["position_change"] == -1:
            trades.append({
                "EntryTime": entry_time,
                "EntryPrice": entry_price,
                "ExitTime": row["Time"],
                "ExitPrice": current_price,
                "ExitType": "MeanReversion"
            })
            in_trade = False

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate P&L
trades_df["PnL_pct"] = (trades_df["ExitPrice"] - trades_df["EntryPrice"]) / trades_df["EntryPrice"] * 100

# === PERFORMANCE METRICS ===
print(f"\n=== TRADE SUMMARY ===")
print(f"Total trades: {len(trades_df)}")
print(f"\nExit type breakdown:")
print(trades_df["ExitType"].value_counts())

print(f"\nWin rate: {(trades_df['PnL_pct'] > 0).sum() / len(trades_df) * 100:.1f}%")
print(f"Average P&L per trade: {trades_df['PnL_pct'].mean():.2f}%")
print(f"Best trade: {trades_df['PnL_pct'].max():.2f}%")
print(f"Worst trade: {trades_df['PnL_pct'].min():.2f}%")

# Calculate strategy returns for Sharpe ratio
df["position_shifted"] = df["position"].shift(1).fillna(0)
df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["ret"].fillna(0, inplace=True)
df["strat_ret"] = df["position_shifted"] * df["ret"]
df["equity"] = (1 + df["strat_ret"]).cumprod()

# Performance metrics
total_return = df["equity"].iloc[-1] - 1
annual_bars = 24 * 252
annualized_return = (1 + total_return) ** (annual_bars / len(df)) - 1
vol = df["strat_ret"].std() * np.sqrt(annual_bars)
sharpe = annualized_return / vol if vol != 0 else np.nan

print(f"\n=== PERFORMANCE METRICS ===")
print(f"Total return: {total_return:.2%}")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized volatility: {vol:.2%}")
print(f"Sharpe ratio (no rf): {sharpe:.2f}")

# === CREATE VISUALIZATION ===
# Calculate position changes for the full dataframe
df["Time"] = pd.to_datetime(df["Time"])

# Plot last 30 days
last_month = df["Time"].max() - pd.Timedelta(days=30)
plot_df = df[df["Time"] >= last_month].copy()

# Find buy/sell signals in this window
buys = plot_df[plot_df["position_change"] == 1]
sells = plot_df[plot_df["position_change"] == -1]

plt.figure(figsize=(15, 8))

# Plot price and Bollinger Bands
plt.plot(plot_df["Time"], plot_df["Close"], color='black', label="S&P 500 (SPY) Close", linewidth=1.5, zorder=2)
plt.plot(plot_df["Time"], plot_df["sma"], color='blue', label=f"Middle Band (SMA {BB_PERIOD})", linewidth=1.2, linestyle='--')
plt.plot(plot_df["Time"], plot_df["upper_band"], color='red', label=f"Upper Band (+{BB_STD}σ)", linewidth=1, linestyle='--', alpha=0.7)
plt.plot(plot_df["Time"], plot_df["lower_band"], color='green', label=f"Lower Band (-{BB_STD}σ)", linewidth=1, linestyle='--', alpha=0.7)

# Fill between bands
plt.fill_between(plot_df["Time"], plot_df["lower_band"], plot_df["upper_band"], alpha=0.1, color='gray')

# Plot BUY signals (green triangles)
if len(buys) > 0:
    plt.scatter(buys["Time"], buys["Close"], color='green', label="Buy Signal (Oversold)", marker='^', s=150, zorder=5, edgecolors='black')

# Plot SELL signals (red triangles)
if len(sells) > 0:
    plt.scatter(sells["Time"], sells["Close"], color='red', label="Sell Signal (Mean Reversion)", marker='v', s=150, zorder=5, edgecolors='black')

plt.title("S&P 500 Bollinger Bands Mean Reversion Strategy (Last 30 Days)", fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sp500_bollinger_bands.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved as 'sp500_bollinger_bands.png'")
plt.show()

print(f"\n=== FIRST 5 TRADES ===")
print(trades_df[["EntryTime", "EntryPrice", "ExitPrice", "ExitType", "PnL_pct"]].head())