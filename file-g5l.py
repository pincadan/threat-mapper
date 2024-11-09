import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Load and preprocess network traffic data
data = pd.read_csv("network_traffic.csv")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define LSTM model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(1, data_scaled.shape[1]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM model on network traffic data
model.fit(data_scaled, data['label'], epochs=10, batch_size=32)

# Analyze network traffic and identify anomalies
def analyze_traffic(traffic):
    traffic_scaled = scaler.transform(traffic)
    anomaly_prob = model.predict(traffic_scaled)
    if anomaly_prob > 0.5:
        return True, anomaly_prob
    else:
        return False, anomaly_prob

# Example usage
new_traffic = [[100, 200, 150, 250, 120, 80, 90, 110]]
is_anomaly, prob = analyze_traffic(new_traffic)
if is_anomaly:
    print(f"Anomaly detected with probability: {prob}")
    # Suggest security measures
else:
    print("No anomalies detected")