#!/usr/bin/env python
# coding: utf-8

# In[45]:


import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from functools import reduce


# In[46]:


tickers = ["AMJ","XLY","XLB","^GSPC","^DJI","^IXIC","^NYA","^RUT"]
start_date = "2018-08-19"
end_date = "2023-09-30"
etf_data = {}
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    
    etf_data[ticker] = data

    
# ETF data frames
amj_df = etf_data["AMJ"]
xly_df = etf_data["XLY"]
xlb_df = etf_data["XLB"]

# Index data frames
gspc_df = etf_data["^GSPC"]
dji_df = etf_data["^DJI"]
ixic_df = etf_data["^IXIC"]
nya_df = etf_data["^NYA"]
rut_df = etf_data["^RUT"]

dji_df = dji_df.rename(columns={'Adj Close': 'DJI_Adj Close', 'Close': 'DJI_Close', 'High': 'DJI_High', 'Low': 'DJI_Low', 'Open': 'DJI_Open','Volume': 'DJI_Volume'})
gspc_df = gspc_df.rename(columns={'Adj Close': 'GSPC_Adj Close', 'Close': 'GSPC_Close', 'High': 'GSPC_High', 'Low': 'GSPC_Low', 'Open': 'GSPC_Open','Volume': 'GSPC_Volume'})
ixic_df = ixic_df.rename(columns={'Adj Close': 'IXIC_Adj Close', 'Close': 'IXIC_Close', 'High': 'IXIC_High', 'Low': 'IXIC_Low', 'Open': 'IXIC_Open','Volume': 'IXIC_Volume'})
nya_df = nya_df.rename(columns={'Adj Close': 'NYA_Adj Close', 'Close': 'NYA_Close', 'High': 'NYA_High', 'Low': 'NYA_Low', 'Open': 'NYA_Open','Volume': 'NYA_Volume'})
rut_df = rut_df.rename(columns={'Adj Close': 'RUT_Adj Close', 'Close': 'RUT_Close', 'High': 'RUT_High', 'Low': 'RUT_Low', 'Open': 'RUT_Open','Volume': 'RUT_Volume'})


# In[47]:


etf_data


# In[48]:


def analytical_inferences(etf_data):
    '''
    Function calculates all inferences possible like RSI, EMA, SMA. 
    Input: 1 data frame containing ETF data 
    Output: The data frame with all calculated values for the particular ETF.
    '''
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
    
    dmi = ta.adx(etf_data.High, etf_data.Low, etf_data.Close)
    etf_data['ADX']=dmi['ADX_14']
    etf_data['DMI+']=dmi['DMP_14']
    etf_data['DMI-']=dmi['DMN_14']
    # Calculate ADX trend strength
    etf_data['ADX_Trend_Strength'] = etf_data['ADX'].rolling(window=3).mean()  # Adjust the rolling window parameter
    
    # Calculate DI convergence or divergence
    etf_data['DI_Convergence_Divergence'] = etf_data['DMI+'] - etf_data['DMI-']  # Adjust the length parameter
    return etf_data


# In[49]:


amj_df = analytical_inferences(amj_df)
xly_df = analytical_inferences(xly_df)
xlb_df = analytical_inferences(xlb_df)


# In[50]:


amj_combined_df = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), [amj_df, dji_df, gspc_df, ixic_df, nya_df, rut_df]))
xly_combined_df = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), [xly_df, dji_df, gspc_df, ixic_df, nya_df, rut_df]))
xlb_combined_df = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), [xlb_df, dji_df, gspc_df, ixic_df, nya_df, rut_df]))


# In[51]:


amj_combined_df


# In[ ]:





# In[ ]:





# In[ ]:




