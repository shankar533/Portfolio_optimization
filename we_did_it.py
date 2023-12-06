#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,TimeDistributed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import Input, Dense, LSTM, Reshape
from keras.models import Model


# In[2]:


tickers = ["AMJ","XLY","XLB","^GSPC","^DJI","^IXIC","^NYA","^RUT"]
start_date = "2018-08-19"
end_date = "2023-11-01"
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

# drop Volume
amj_df = amj_df.drop('Volume', axis = 1)

dji_df = dji_df.rename(columns={'Adj Close': 'DJI_Adj Close', 'Close': 'DJI_Close', 'High': 'DJI_High', 'Low': 'DJI_Low', 'Open': 'DJI_Open','Volume': 'DJI_Volume'})
gspc_df = gspc_df.rename(columns={'Adj Close': 'GSPC_Adj Close', 'Close': 'GSPC_Close', 'High': 'GSPC_High', 'Low': 'GSPC_Low', 'Open': 'GSPC_Open','Volume': 'GSPC_Volume'})
ixic_df = ixic_df.rename(columns={'Adj Close': 'IXIC_Adj Close', 'Close': 'IXIC_Close', 'High': 'IXIC_High', 'Low': 'IXIC_Low', 'Open': 'IXIC_Open','Volume': 'IXIC_Volume'})
nya_df = nya_df.rename(columns={'Adj Close': 'NYA_Adj Close', 'Close': 'NYA_Close', 'High': 'NYA_High', 'Low': 'NYA_Low', 'Open': 'NYA_Open','Volume': 'NYA_Volume'})
rut_df = rut_df.rename(columns={'Adj Close': 'RUT_Adj Close', 'Close': 'RUT_Close', 'High': 'RUT_High', 'Low': 'RUT_Low', 'Open': 'RUT_Open','Volume': 'RUT_Volume'})


# In[23]:


def analytical_inferences(etf_data):
    '''
    Function calculates all technical indicators possible like RSI, EMA, SMA. 
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


# In[24]:


def stats_for_model(etf_data):
    '''
    Function calculates all technical indicators possible like RSI, EMA, SMA. 
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


# In[25]:


# Function call for calculating technical indicators
amj_df = analytical_inferences(amj_df)
xly_df = analytical_inferences(xly_df)
xlb_df = analytical_inferences(xlb_df)


# In[26]:


amj_df_stats = stats_for_model(amj_df)


# In[27]:


def merge_stock_index_data(stock_data, dji_df, gspc_df, ixic_df, nya_df, rut_df):
    new_df = pd.DataFrame()
    new_df = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), [stock_data, dji_df, gspc_df, ixic_df, nya_df, rut_df]))
    new_df = new_df.dropna()
    return new_df


# In[28]:


# Joining the stock data with index data
amj_combined_df = merge_stock_index_data(amj_df, dji_df, gspc_df, ixic_df, nya_df, rut_df)
xly_combined_df = merge_stock_index_data(xly_df, dji_df, gspc_df, ixic_df, nya_df, rut_df)
xlb_combined_df = merge_stock_index_data(xlb_df, dji_df, gspc_df, ixic_df, nya_df, rut_df)


# In[29]:


def data_preparation_for_dbn(stock_data):
    '''
    The function prepares data for the models defined below. 
    This function manipulates the data so that the model inputs previous day's data to predict return for current day.
    Input: Data frame with all features required for regression.
    Output: Manipulated Data frame
    '''
    new_df = pd.DataFrame()
    new_df = stock_data.shift(1)
    new_df = new_df.dropna()
    new_df = pd.merge(stock_data['Adj Close'], new_df, on = 'Date', how='inner')
    new_df = new_df.rename(columns={'Adj Close_x': 'Curr Adj Close', 'Adj Close_y': 'Prev Adj Close'})
    
    return new_df


# In[30]:


# Function Call for data preparation
amj_manipulated_data = data_preparation_for_dbn(amj_df_stats)


# In[31]:


def data_transformation(data):
    X = data.drop('Curr Adj Close', axis =1)
    y = data['Curr Adj Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
    
    return X, y, X_train, X_test, y_train, y_test


# In[32]:


def data_transformation_model2(data):
    X = data.drop(data.iloc[:, 5:], axis = 1)
    y = data.iloc[:, :4].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
    
    return X, y, X_train, X_test, y_train, y_test,X_scaled


# In[42]:


model2_df = amj_manipulated_data.iloc[:, :5].copy()


# In[43]:


model2_df=model2_df.rename(columns={'Open': 'Prev Open', 'High': 'Prev High', 'Low': 'Prev Low', 'Close': 'Prev Close'})


# In[44]:


model2_df


# In[45]:


model2_df = pd.merge(amj_combined_df.iloc[:, :5], model2_df, on = 'Date', how='inner')
#model2_df = model2_df.drop(['Adj Close'],axis=1)
model2_df = model2_df.drop(['Adj Close'],axis=1)
model2_df


# In[46]:


model2_df = model2_df.rename(columns={'Open': 'Curr Open', 'High': 'Curr High', 'Low': 'Curr Low', 'Close': 'Curr Close'})


# In[47]:


model2_df


# In[49]:


####################################### DBN Begins here to predict Adj Close ###################################################
X, y, *_ = data_transformation(amj_manipulated_data)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X)
dbn_input = Input(shape=(scaled_features.shape[1],))
x = Dense(units=100, activation='relu')(dbn_input)
x = Dense(units=80, activation='relu')(x)
x = Dense(units=60, activation='relu')(x)
dbn_output = Dense(units=20, activation='linear')(x)
dbn_model = Model(inputs=dbn_input, outputs=dbn_output)
dbn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the DBN model
dbn_model.fit(scaled_features, y, epochs=50, batch_size=32)

dbn_output = dbn_model.predict(scaled_features)
n_samples, n_features = dbn_output.shape
n_timesteps = 1
dbn_output_reshaped = dbn_output.reshape((n_samples, n_timesteps, n_features))
lstm_input = Input(shape=(n_timesteps, n_features))

X_train, X_test, y_train, y_test = train_test_split(dbn_output_reshaped,y, test_size=0.2, random_state=42)
######################################## LSTM begins here to predict Adj Close ###################################################
x = LSTM(50, activation='relu')(lstm_input)
lstm_output = Dense(1)(x)  # Output layer with one neuron for regression
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_model.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) loss for regression
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)


# In[55]:


X,y,*_,X_scaled = data_transformation_model2(model2_df)


# In[56]:


####################################### DBN Begins here to predict OHLC #############################
scaler_2 = MinMaxScaler()
X_scaled_2 = scaler_2.fit_transform(X)
# Define and train the DBN model
dbn_model_2 = Sequential()
dbn_model_2.add(Dense(units=100, activation='relu', input_dim=X_scaled_2.shape[1]))
dbn_model_2.add(Dense(units=80, activation='relu'))
dbn_model_2.add(Dense(units=60, activation='relu'))
dbn_model_2.add(Dense(units=4, activation='linear'))  # Output layer with 20 neurons
dbn_model_2.compile(optimizer='adam', loss='mean_squared_error')

# Train the DBN model
dbn_model_2.fit(X_scaled_2, y, epochs=50, batch_size=32)

# Get the output of the DBN as the input for LSTM
dbn_output_2 = dbn_model_2.predict(X_scaled_2)

# Reshape the features for LSTM input (assuming a time series structure)
n_samples, n_features = dbn_output_2.shape
n_timesteps=1
dbn_output_reshaped_2 = dbn_output_2.reshape((n_samples, n_timesteps, n_features))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dbn_output_reshaped_2, y, test_size=0.2, random_state=42)

 ######################################## LSTM begins here to predict Adj Close ###################################################
lstm_model_2 = Sequential()
lstm_model_2.add(LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)))
lstm_model_2.add(Dense(4))  # Output layer with one neuron for regression
lstm_model_2.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) loss for regression

# Train the LSTM model
lstm_model_2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the LSTM model
mse = lstm_model_2.evaluate(X_test, y_test, verbose=0)
print(f'Mean Squared Error on test set: {mse}')

# Make predictions using the LSTM model
predictions = lstm_model_2.predict(X_test)

# Convert predictions back to the original scale
#predicted_prices = scaler.inverse_transform(np.concatenate([X_test.reshape(-1, n_features)[:, :-1], predictions], axis=1))[:, -1]
predicted_prices=predictions.flatten()


# In[57]:


# Function to generate dates:
def generate_dates(year):
    # Generate a list of dates for the given year
    dates = pd.date_range(f'11-01-{year}', f'10-31-{year+1}', freq='D')
    
    # Convert the list of dates into a DataFrame
    dates_df = pd.DataFrame(dates, columns=['Date'])
    
    return dates_df


# In[58]:


def predict_adj_close(temp_input_prep):
    
    scaled_new_features = scaler.transform(temp_input_prep.drop('Curr Adj Close', axis =1))

    # Get the output of the DBN model
    dbn_new_output = dbn_model.predict(scaled_new_features)
    
    # Reshape the features for LSTM input
    n_samples, n_features = dbn_new_output.shape
    n_timesteps = 1
    dbn_new_output_reshaped = dbn_new_output.reshape((n_samples, n_timesteps, n_features))
    
    # Make predictions using the LSTM model
    new_predictions = lstm_model.predict(dbn_new_output_reshaped)
    return new_predictions.flatten()


# In[59]:


def predict_OHLC(temp_input_prep):
    
    scaled_new_features = scaler_2.transform(temp_input_prep)

    # Get the output of the DBN model
    dbn_new_output = dbn_model_2.predict(scaled_new_features)
    
    # Reshape the features for LSTM input
    n_samples, n_features = dbn_new_output.shape
    n_timesteps = 1
    dbn_new_output_reshaped = dbn_new_output.reshape((n_samples, n_timesteps, n_features))
    
    # Make predictions using the LSTM model
    new_predictions = lstm_model_2.predict(dbn_new_output_reshaped)
    return new_predictions.flatten()


# In[60]:


major_df = pd.DataFrame()
temp_input_prep = pd.DataFrame()
major_df['Date'] = generate_dates(2023)
major_df['Prev O'] = 0
major_df['Prev H'] = 0
major_df['Prev L'] = 0
major_df['Prev C'] = 0
major_df['Curr O'] = 0
major_df['Curr H'] = 0
major_df['Curr L'] = 0
major_df['Curr C'] = 0
major_df['Sim Adj Close'] = 0


major_df['Prev O'][0] = amj_manipulated_data['Open'][-1]
major_df['Prev H'][0] = amj_manipulated_data['High'][-1]
major_df['Prev L'][0] = amj_manipulated_data['Low'][-1]
major_df['Prev C'][0] = amj_manipulated_data['Close'][-1]
#
for i in range(0,366):
    sim_adj = predict_adj_close(amj_manipulated_data[-1:])
    major_df['Sim Adj Close'][i] = sim_adj
    a = [[0]*5]
    a[0][0] = major_df['Prev O'][i]
    a[0][1] = major_df['Prev H'][i]
    a[0][2] = major_df['Prev L'][i]
    a[0][3] = major_df['Prev C'][i]
    a[0][4] = major_df['Sim Adj Close'][i]
    b = predict_OHLC(a)
    major_df['Curr O'][i] = b[0]
    major_df['Curr H'][i] = b[1]
    major_df['Curr L'][i] = b[2]
    major_df['Curr C'][i] = b[3]
    
    
    new_row = {'Open': major_df['Curr O'][i], 'High': major_df['Curr H'][i],'Low': major_df['Curr L'][i],'Close': major_df['Curr C'][i], 'Adj Close': major_df['Sim Adj Close'][i]}
    new_index = major_df['Date'][i]
    
    amj_df.loc[new_index] = new_row
    
    manipulated_amj_df = stats_for_model(amj_df)
    amj_manipulated_data = data_preparation_for_dbn(manipulated_amj_df)
    
    major_df['Prev O'][i+1] = major_df['Curr O'][i]
    major_df['Prev H'][i+1] = major_df['Curr H'][i]
    major_df['Prev L'][i+1] = major_df['Curr L'][i]
    major_df['Prev C'][i+1] = major_df['Curr C'][i]
    
  
    #stats_for_model(etf_data)
    #data_preparation_for_dbn(stock_data)
    #data_transformation(data)
    #temp_input_prep = 
    #sim_adj_close = predict_adj_close(temp_input_prep)
    



# In[62]:


major_df.head(51)


# In[63]:


amj_df


# In[ ]:




