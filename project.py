#!/usr/bin/env python
# coding: utf-8

# In[68]:


import yfinance as yf
import pandas as pd
import pandas_ta as ta

# Download ETF data from Yahoo Finance
tickers = ["AMJ"]
start_date = "2018-08-19"
end_date = "2023-09-30"
etf_data = yf.download(tickers, start=start_date, end=end_date)
etf_data


# In[69]:


# Define the period for RSI
rsi_period = 14

# Calculate RSI
etf_data['RSI'] = ta.rsi(etf_data['Adj Close'], length=rsi_period)

# Calculate overbought/oversold conditions
etf_data['Overbought'] = (etf_data['RSI'] > 70).astype(int)
etf_data['Oversold'] = (etf_data['RSI'] < 30).astype(int)

# Calculate divergence between price and RSI
etf_data['Price_RSI_Divergence'] = etf_data['Close'].diff() - etf_data['RSI'].diff()

# Calculate rate of change of RSI
etf_data['ROC_RSI'] = etf_data['RSI'].pct_change() * 100

# Calculate RSI trend confirmation
etf_data['RSI_Trend_Confirmation'] = (etf_data['RSI'] > etf_data['RSI'].shift(1)).astype(int)

# Assuming 'Close' is the column containing closing prices
etf_data['EMA'] = ta.ema(etf_data['Close'], length=14)  # Adjust the period as needed

# Feature 1: EMA over a specific period
# Already calculated and stored in 'EMA' column

# Feature 2: Difference between current price and EMA
etf_data['Price_EMA_Difference'] = etf_data['Close'] - etf_data['EMA']

# Feature 3: Slope of EMA
etf_data['Slope_EMA'] = ta.slope(etf_data['EMA'])

# Feature 4: EMA convergence or divergence
etf_data['EMA_Convergence'] = (etf_data['Close'] > etf_data['EMA']).astype(int)
etf_data['EMA_Divergence'] = (etf_data['Close'] < etf_data['EMA']).astype(int)

# Feature 5: Rate of change of EMA
etf_data['ROC_EMA'] = etf_data['EMA'].pct_change() * 100

# Assuming 'Close' is the column containing closing prices
etf_data['SMA'] = ta.sma(etf_data['Close'], length=14)  # Adjust the period as needed

# Feature 1: SMA over a specific period
# Already calculated and stored in 'SMA' column

# Feature 2: Difference between current price and SMA
etf_data['Price_SMA_Difference'] = etf_data['Close'] - etf_data['SMA']

# Feature 3: Slope of SMA
etf_data['Slope_SMA'] = ta.slope(etf_data['SMA'])

# Feature 4: SMA convergence or divergence
etf_data['SMA_Convergence'] = (etf_data['Close'] > etf_data['SMA']).astype(int)
etf_data['SMA_Divergence'] = (etf_data['Close'] < etf_data['SMA']).astype(int)

# Feature 5: Rate of change of SMA
etf_data['ROC_SMA'] = etf_data['SMA'].pct_change() * 100

# Display the DataFrame with calculated features
etf_data


# In[70]:


dmi = ta.adx(etf_data.High, etf_data.Low, etf_data.Close)
etf_data['ADX']=dmi['ADX_14']
etf_data['DMI+']=dmi['DMP_14']
etf_data['DMI-']=dmi['DMN_14']
# Calculate ADX trend strength
etf_data['ADX_Trend_Strength'] = etf_data['ADX'].rolling(window=3).mean()  # Adjust the rolling window parameter

# Calculate DI convergence or divergence

etf_data['DI_Convergence_Divergence'] = etf_data['DMI+'] - etf_data['DMI-']  # Adjust the length parameter


# In[71]:


etf_data.isna().sum()


# In[72]:


etf_data=etf_data.dropna()


# In[73]:


etf_data.info()


# In[74]:


import yfinance as yf
import numpy as np
tickers =['^GSPC','^DJI','^IXIC','^NYA','^RUT']
start_date = "2018-10-01"
end_date = "2023-09-30"
indexdf = yf.download(tickers, start=start_date, end=end_date)


# In[75]:


indexdf


# In[61]:


indexdf.info()


# In[76]:


# Get the unique symbols in the second level of the MultiIndex
symbols = indexdf.columns.get_level_values(1).unique()

# Create a dictionary to store DataFrames for each symbol
dfs = {}

# Split the DataFrame based on symbols
for symbol in symbols:
    dfs[symbol] = indexdf.xs(key=symbol, axis=1, level=1)

# Access individual DataFrames using their corresponding symbols
df_dji = dfs["^DJI"]
df_gspc = dfs["^GSPC"]
df_ixic = dfs["^IXIC"]
df_nya = dfs["^NYA"]
df_rut = dfs["^RUT"]


# In[77]:


df_dji = df_dji.rename(columns={'Adj Close': 'DJI_Adj Close', 'Close': 'DJI_Close', 'High': 'DJI_High', 'Low': 'DJI_Low', 'Open': 'DJI_Open','Volume': 'DJI_Volume'})
df_gspc = df_gspc.rename(columns={'Adj Close': 'GSPC_Adj Close', 'Close': 'GSPC_Close', 'High': 'GSPC_High', 'Low': 'GSPC_Low', 'Open': 'GSPC_Open','Volume': 'GSPC_Volume'})
df_ixic = df_ixic.rename(columns={'Adj Close': 'IXIC_Adj Close', 'Close': 'IXIC_Close', 'High': 'IXIC_High', 'Low': 'IXIC_Low', 'Open': 'IXIC_Open','Volume': 'IXIC_Volume'})
df_nya = df_nya.rename(columns={'Adj Close': 'NYA_Adj Close', 'Close': 'NYA_Close', 'High': 'NYA_High', 'Low': 'NYA_Low', 'Open': 'NYA_Open','Volume': 'NYA_Volume'})
df_rut = df_rut.rename(columns={'Adj Close': 'RUT_Adj Close', 'Close': 'RUT_Close', 'High': 'RUT_High', 'Low': 'RUT_Low', 'Open': 'RUT_Open','Volume': 'RUT_Volume'})


# In[78]:


from functools import reduce
# List of DataFrames
dfs = [etf_data, df_dji, df_gspc, df_ixic, df_nya, df_rut]

# Merge DataFrames using reduce and lambda function
merged_df = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs))


# In[79]:


merged_df

