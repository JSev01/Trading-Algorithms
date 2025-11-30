import pandas as pd
import numpy as np
import yfinance as yf

#data
df = yf.download("EURUSD=X", interval="1h", period="2y")
df = df[['Close']]
df.reset_index(inplace=True)
df.rename(columns={'Datetime':'Time'}, inplace=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

#fast and slow SMA
fastWindow = 20
slowWindow = 100
df["sma_fast"] = df["Close"].rolling(window=fastWindow).mean()
df["sma_slow"] = df["Close"].rolling(window=slowWindow).mean()

#long and short positions
df["position"] = 0
df.loc[df["sma_fast"] > df["sma_slow"], "position"] = 1   #go long
df.loc[df["sma_fast"] < df["sma_slow"], "position"] = -1  #go short

df["position_change"] = df["position"].diff()

#risk management 1:2 ratio
stop_loss = 0.01 
take_profit = 0.02 

#===TRADE SIMULATION===
trades = []
in_trade = False
current_direction = 0  #track if we're going long (1) or short (-1)

for i, row in df.iterrows():
    if not in_trade:
        #entry: position changes from 0 to 1 (long) or 0 to -1 (short)
        #or from 1 to -1 (flip from long to short) or -1 to 1 (flip from short to long)
        if row["position_change"] == 1:  #go long
            entry_price = row["Close"]
            entry_time = row["Time"]
            current_direction = 1
            in_trade = True
        elif row["position_change"] == -1:  #go short
            entry_price = row["Close"]
            entry_time = row["Time"]
            current_direction = -1
            in_trade = True
        elif row["position_change"] == 2:  #flip from short (-1) to long (1)
            entry_price = row["Close"]
            entry_time = row["Time"]
            current_direction = 1
            in_trade = True
        elif row["position_change"] == -2:  #flip from long (1) to short (-1)
            entry_price = row["Close"]
            entry_time = row["Time"]
            current_direction = -1
            in_trade = True
    else:
        current_price = row["Close"]
        
        #=== LONG POSITION EXITS ===
        if current_direction == 1:
            #sstop loss: price falls 1% below entry
            if current_price <= entry_price * (1 - stop_loss):
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Long",
                    "ExitType": "StopLoss"
                })
                in_trade = False
            #take-profit: price rises 2% above entry
            elif current_price >= entry_price * (1 + take_profit):
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Long",
                    "ExitType": "TakeProfit"
                })
                in_trade = False
            #crossover exit OR flip to short
            elif row["position_change"] == -2:  #flip to short
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Long",
                    "ExitType": "Crossover"
                })
                #immediately enter short position
                entry_price = row["Close"]
                entry_time = row["Time"]
                current_direction = -1
                in_trade = True
        
        #=== SHORT POSITION EXITS ===
        elif current_direction == -1:
            #stop-loss for SHORT: price RISES 1% above entry
            if current_price >= entry_price * (1 + stop_loss):
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Short",
                    "ExitType": "StopLoss"
                })
                in_trade = False
            #take-profit for SHORT: price FALLS 2% below entry
            elif current_price <= entry_price * (1 - take_profit):
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Short",
                    "ExitType": "TakeProfit"
                })
                in_trade = False
            #crossover exit OR flip to long
            elif row["position_change"] == 2:  #flip to long
                trades.append({
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitTime": row["Time"],
                    "ExitPrice": current_price,
                    "Direction": "Short",
                    "ExitType": "Crossover"
                })
                #immediately enter long position
                entry_price = row["Close"]
                entry_time = row["Time"]
                current_direction = 1
                in_trade = True

#convert date to DataFrame
trades_df = pd.DataFrame(trades)

#=== P&L ===
trades_df["PnL_pct"] = 0.0

#long trades: profit when exit > entry
long_mask = trades_df["Direction"] == "Long"
trades_df.loc[long_mask, "PnL_pct"] = (
    (trades_df.loc[long_mask, "ExitPrice"] - trades_df.loc[long_mask, "EntryPrice"]) 
    / trades_df.loc[long_mask, "EntryPrice"] * 100
)

#short trades: profit when exit < entry
short_mask = trades_df["Direction"] == "Short"
trades_df.loc[short_mask, "PnL_pct"] = (
    (trades_df.loc[short_mask, "EntryPrice"] - trades_df.loc[short_mask, "ExitPrice"]) 
    / trades_df.loc[short_mask, "EntryPrice"] * 100
)

#=== OUT-OF-SAMPLE TEST WITH €10,000 === (actually not OOS)

#define out-of-sample period (last 6 months)
OOS_START = df["Time"].max() - pd.Timedelta(days=182)

print(f"\n=== OUT-OF-SAMPLE PERIOD ===")
print(f"OOS Start: {OOS_START}")
print(f"OOS End: {df['Time'].max()}")

#filter trades that occurred in the OOS period
oos_trades = trades_df[trades_df["EntryTime"] >= OOS_START].copy()

print(f"\nNumber of trades in OOS: {len(oos_trades)}")

if len(oos_trades) == 0:
    print("No trades in OOS period - try extending the date range!")
else:
    #simulate portfolio starting with €10,000
    initial_capital = 10_000.0
    portfolio_values = [initial_capital]
    
    for idx, trade in oos_trades.iterrows():
        #P&L percentage is already calculated correctly for long/short
        pct_return = trade["PnL_pct"] / 100  #convert from percentage to decimal
        
        #apply to current portfolio value
        new_value = portfolio_values[-1] * (1 + pct_return)
        portfolio_values.append(new_value)
    
    #add portfolio tracking to trades dataframe
    oos_trades["PortfolioValue"] = portfolio_values[1:]
    
    #calculate final results
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    #calculate max drawdown
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\n=== OUT-OF-SAMPLE PERFORMANCE (LONG/SHORT) ===")
    print(f"Starting Capital: €{initial_capital:,.2f}")
    print(f"Final Portfolio Value: €{final_value:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Profit/Loss: €{final_value - initial_capital:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    #trade statistics in OOS
    print(f"\n=== OOS TRADE STATISTICS ===")
    print(f"Total OOS Trades: {len(oos_trades)}")
    print(f"Long Trades: {(oos_trades['Direction'] == 'Long').sum()}")
    print(f"Short Trades: {(oos_trades['Direction'] == 'Short').sum()}")
    print(f"Win Rate: {(oos_trades['PnL_pct'] > 0).sum() / len(oos_trades) * 100:.1f}%")
    print(f"Average P&L per trade: {oos_trades['PnL_pct'].mean():.2f}%")
    print(f"Best Trade: {oos_trades['PnL_pct'].max():.2f}%")
    print(f"Worst Trade: {oos_trades['PnL_pct'].min():.2f}%")
    
    print(f"\nExit Type Breakdown (OOS):")
    print(oos_trades["ExitType"].value_counts())
    
    #show first and last few trades with portfolio progression
    print(f"\n=== FIRST 3 OOS TRADES ===")
    print(oos_trades[["EntryTime", "Direction", "EntryPrice", "ExitPrice", "ExitType", "PnL_pct", "PortfolioValue"]].head(3))
    
    print(f"\n=== LAST 3 OOS TRADES ===")
    print(oos_trades[["EntryTime", "Direction", "EntryPrice", "ExitPrice", "ExitType", "PnL_pct", "PortfolioValue"]].tail(3))


#=== RESULTS ===
print(f"\n=== LONG/SHORT TRADE SUMMARY ===")
print(f"Total trades: {len(trades_df)}")
print(f"\nDirection breakdown:")
print(trades_df["Direction"].value_counts())
print(f"\nExit type breakdown:")
print(trades_df["ExitType"].value_counts())

print(f"\nWin rate: {(trades_df['PnL_pct'] > 0).sum() / len(trades_df) * 100:.1f}%")
print(f"Average P&L per trade: {trades_df['PnL_pct'].mean():.2f}%")
print(f"Best trade: {trades_df['PnL_pct'].max():.2f}%")
print(f"Worst trade: {trades_df['PnL_pct'].min():.2f}%")

print(f"\n=== FIRST 5 TRADES ===")
print(trades_df[["EntryTime", "Direction", "EntryPrice", "ExitPrice", "ExitType", "PnL_pct"]].head())