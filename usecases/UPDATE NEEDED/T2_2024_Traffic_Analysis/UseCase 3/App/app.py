
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\Local MOAI PROJECT\merged_data.csv"
df = pd.read_csv(file_path)

# Renaming vehicle_class_1 to vehicle_class_13 according to your provided columns
df = df.rename(columns={
    'vehicle_class_1': 'Light_Vehicles',
    'vehicle_class_2': 'Light_Vehicles_with_Trailer',
    'vehicle_class_3': 'Two_Axle_Trucks_Buses',
    'vehicle_class_4': 'Three_Axle_Trucks_Buses',
    'vehicle_class_5': 'Four_Axle_Trucks',
    'vehicle_class_6': 'Three_Axle_Articulated_Vehicles',
    'vehicle_class_7': 'Four_Axle_Articulated_Vehicles',
    'vehicle_class_8': 'Five_Axle_Articulated_Vehicles',
    'vehicle_class_9': 'Six_Axle_Articulated_Vehicles',
    'vehicle_class_10': 'B_Double_Trucks',
    'vehicle_class_11': 'Double_Road_Trains',
    'vehicle_class_12': 'Triple_Road_Trains',
    'vehicle_class_13': 'Unknown_Vehicles'
})

# List of vehicle types
vehicle_types = [
    'Light_Vehicles',
    'Light_Vehicles_with_Trailer',
    'Two_Axle_Trucks_Buses',
    'Three_Axle_Trucks_Buses',
    'Four_Axle_Trucks',
    'Three_Axle_Articulated_Vehicles',
    'Four_Axle_Articulated_Vehicles',
    'Five_Axle_Articulated_Vehicles',
    'Six_Axle_Articulated_Vehicles',
    'B_Double_Trucks',
    'Double_Road_Trains',
    'Triple_Road_Trains',
    'Unknown_Vehicles'
]

# Streamlit UI elements
st.title("Traffic Volume Prediction using LSTM")

# User inputs
date = st.selectbox('Select Date', df['date'].unique())
time = st.selectbox('Select Time', df['time'].unique())
vehicle_type = st.selectbox('Select Vehicle Type', vehicle_types)

# Filter data based on date and time
df_filtered = df[(df['date'] == date) & (df['time'] == time)]

# Check if filtered data is available
if df_filtered.empty:
    st.warning("No data available for the selected filters. Please select different options.")
else:
    # Normalize the selected vehicle type data
    scaler = MinMaxScaler()
    df_filtered_scaled = scaler.fit_transform(df_filtered[[vehicle_type]])

    # Prepare the data for LSTM with more time steps to handle volatility
    def create_dataset(data, time_step=3):  # Using time_step=3 to capture more temporal dependencies
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Setting time_step = 3 for better prediction of volatile data
    time_step = 3
    X, Y = create_dataset(df_filtered_scaled, time_step)

    # Ensure there's enough data for training
    if len(X) == 0 or len(Y) == 0:
        st.warning("Not enough data for training. Please select different options.")
    else:
        # Split the data into training and testing datasets (70% training, 30% testing)
        train_size = int(len(X) * 0.7)
        test_size = len(X) - train_size
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Create and fit the LSTM model with Dropout and EarlyStopping
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(time_step, 1)))  # Increased LSTM units
        model.add(Dropout(0.4))  # Increased dropout to handle volatility
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(50))
        model.add(Dense(1))

        # Compile the model with Adam optimizer and reduced learning rate
        optimizer = Adam(learning_rate=0.001)  # Reduced learning rate for smoother convergence
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_test, Y_test), callbacks=[early_stopping])

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Calculate RMSE and MAE
        train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
        train_mae = mean_absolute_error(Y_train, train_predict)
        test_mae = mean_absolute_error(Y_test, test_predict)

        # Streamlit visualization
        st.write(f"Train RMSE: {train_rmse}, Train MAE: {train_mae}")
        st.write(f"Test RMSE: {test_rmse}, Test MAE: {test_mae}")

        # Plotting the actual and predicted values
        fig, ax = plt.subplots()
        ax.plot(df_filtered_scaled, label='Actual Traffic Volume')

        # Plot training predictions
        train_predict_plot = np.empty_like(df_filtered_scaled)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

        # Plot testing predictions - Fixing the mismatch in shape
        test_predict_plot = np.empty_like(df_filtered_scaled)
        test_predict_plot[:, :] = np.nan

        start_index = len(train_predict) + (time_step * 2)
        end_index = min(start_index + len(test_predict), len(test_predict_plot))  # Ensure we don't exceed array bounds
        test_predict_plot[start_index:end_index, :] = test_predict[:(end_index - start_index)]  # Correct the index

        ax.plot(train_predict_plot, label='Training Prediction')
        ax.plot(test_predict_plot, label='Testing Prediction')

        ax.set_title(f'Actual vs Predicted Traffic Volume for {vehicle_type}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Traffic Volume')
        plt.legend()
        st.pyplot(fig)

        # Button to predict
        if st.button('Predict'):
            st.success("Prediction completed and plotted!")
