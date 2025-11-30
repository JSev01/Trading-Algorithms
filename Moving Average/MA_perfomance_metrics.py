import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#EURUSD data
df = yf.download("EURUSD=X", interval="1h", period="1y")
df = df[['Close']]
df.reset_index(inplace=True)
df.rename(columns={'Datetime':'Time'}, inplace=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

#create windows
fastWindow = 20
slowWindow = 100

df["sma_fast"] = df["Close"].rolling(window=fastWindow).mean()
df["sma_slow"] = df["Close"].rolling(window=slowWindow).mean()

#position (1 = long, 0 = flat)
df["position"] = 0
df.loc[df["sma_fast"] > df["sma_slow"], "position"] = 1

#shift position so todays position is applied on next bars return
df["position_shifted"] = df["position"].shift(1).fillna(0)

#log returns on EURUSD
df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["ret"].fillna(0, inplace=True)

#strategy returns
df["strat_ret"] = df["position_shifted"] * df["ret"]

#equity curve
df["equity"] = (1 + df["strat_ret"]).cumprod()

#basic performance metrics
total_return = df["equity"].iloc[-1] - 1
annual_bars = 24 * 252  # 1H candles and ~252 trading days (one normal year)
annualized_return = (1 + total_return) ** (annual_bars / len(df)) - 1

vol = df["strat_ret"].std() * np.sqrt(annual_bars)
sharpe = annualized_return / vol if vol != 0 else np.nan

print(f"Total return: {total_return:.2%}")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized volatility: {vol:.2%}")
print(f"Sharpe ratio (no rf): {sharpe:.2f}")


#calculate position change
df["position_change"] = df["position"].diff()

#create plotting window for last 30 days
df["Time"] = pd.to_datetime(df["Time"])
last_month = df["Time"].max() - pd.Timedelta(days=60)
plot_df = df[df["Time"] >= last_month].copy()

#buy and sell signals
buys = plot_df[plot_df["position_change"] == 1.0]
sells = plot_df[plot_df["position_change"] == -1.0]

plt.figure(figsize=(15,7))

#closing price
plt.plot(plot_df["Time"], plot_df["Close"], color='black', label="Close Price")
#fast SMA blue, slow SMA red
plt.plot(plot_df["Time"], plot_df["sma_fast"], color='blue', label=f"SMA {fastWindow}")
plt.plot(plot_df["Time"], plot_df["sma_slow"], color='red', label=f"SMA {slowWindow}")

#buy signals (green arrows)
plt.scatter(buys["Time"], buys["Close"], color='green', label="Buy Signal", marker='^', s=100, zorder=5)
#sell signals (red arrows)
plt.scatter(sells["Time"], sells["Close"], color='red', label="Sell Signal", marker='v', s=100, zorder=5)

plt.title("EUR/USD: Moving Average Crossover Signals (Last Month)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()