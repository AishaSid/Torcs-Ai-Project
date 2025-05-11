import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ast
import os
from sklearn.preprocessing import OneHotEncoder

# Define chunk size (adjust if memory issues persist)
CHUNK_SIZE = 10000

# Step 1: Initialize lists to store preprocessed data
X_chunks = []
y_reg_chunks = []  # Steer, Accel, Brake
y_cls_chunks = []  # Gear_output (one-hot)

# List of CSV files - all available datasets
files = [
    "datasets/Peugeot_Oval.csv",
    "datasets/Lancer_Road.csv",
    "datasets/Lancer_Oval.csv",
    "datasets/Lancer_Dirt.csv",
    "datasets/Corolla_Road.csv",
    "datasets/Corolla_Oval.csv",
    "datasets/Peugeot_Dirt.csv",
    "datasets/Peugeot_Road.csv",
    "datasets/Corolla_Dirt.csv"
]

# Expected columns (from sample data)
expected_columns = [
    "Angle", "TrackPos", "DistFromStart", "DistRaced", "Z",
    "SpeedX", "SpeedY", "SpeedZ", "Gear", "RPM", "Fuel", "Damage",
    "RacePos", "CurLapTime", "LastLapTime", "Focus", "Track",
    "Opponents", "WheelSpinVel", "Steer", "Accel", "Brake",
    "Clutch", "ControlFocus", "Gear_output", "Meta"
]

def calculate_additional_features(chunk):
    """Calculate additional features that might be useful for the model."""
    # Calculate speed magnitude
    chunk['speed_magnitude'] = np.sqrt(chunk['SpeedX']**2 + chunk['SpeedY']**2 + chunk['SpeedZ']**2)
    
    # Calculate acceleration (change in speed over time)
    chunk['acceleration'] = chunk['speed_magnitude'].diff()
    
    # Calculate turn sharpness (rate of change of angle)
    chunk['turn_sharpness'] = chunk['Angle'].diff()
    
    # Calculate track position stability
    chunk['track_pos_stability'] = chunk['TrackPos'].rolling(window=5).std()
    
    # Calculate speed stability
    chunk['speed_stability'] = chunk['speed_magnitude'].rolling(window=5).std()
    
    # Fill NaN values created by diff() and rolling()
    chunk = chunk.fillna(0)
    
    return chunk

# Step 2: Process each CSV file in chunks
for file in files:
    if not os.path.exists(file):
        print(f"Error: {file} not found")
        continue
    
    print(f"Processing {file}...")
    # Read CSV in chunks
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
        # Verify columns
        if "Gear.1" in chunk.columns:
            chunk.rename(columns={"Gear.1": "Gear_output"}, inplace=True)
                             
        if set(chunk.columns) != set(expected_columns):
            print(f"Warning: {file} has unexpected columns: {set(chunk.columns) - set(expected_columns)}")
            continue
        
        # Select relevant columns early to reduce memory
        chunk = chunk[["Track", "TrackPos", "Angle", "SpeedX", "SpeedY", "SpeedZ", 
                      "Gear", "RPM", "Opponents", "Steer", "Accel", "Brake", "Gear_output"]]
        
        # Step 3: Parse list columns
        try:
            chunk["Track"] = chunk["Track"].apply(ast.literal_eval)
            chunk["Opponents"] = chunk["Opponents"].apply(ast.literal_eval)
        except ValueError as e:
            print(f"Error parsing list columns in {file}: {e}")
            continue
        
        # Step 4: Calculate additional features
        chunk = calculate_additional_features(chunk)
        
        # Step 5: Clean data
        # Filter valid output ranges
        chunk = chunk[
            (chunk["Steer"].between(-1, 1)) &
            (chunk["Accel"].between(0, 1)) &
            (chunk["Brake"].between(0, 1)) &
            (chunk["Gear_output"].isin([-1, 0, 1, 2, 3, 4, 5, 6]))
        ]
        
        # Step 6: Derive MinOpponent
        chunk["MinOpponent"] = chunk["Opponents"].apply(lambda x: min(x))
        
        # Step 7: Create input array
        X_chunk = np.hstack([
            np.array(chunk["Track"].tolist()),  # 19 columns
            chunk[["TrackPos", "Angle", "SpeedX", "SpeedY", "SpeedZ", 
                  "Gear", "RPM", "MinOpponent", "speed_magnitude", 
                  "acceleration", "turn_sharpness", "track_pos_stability", 
                  "speed_stability"]].values
        ])
        
        # Step 8: Normalize inputs
        # Track: Divide by 200m
        X_chunk[:, :19] = np.clip(X_chunk[:, :19] / 200, 0, 1)
        # TrackPos: Map [-1, 1] to [0, 1]
        X_chunk[:, 19] = (X_chunk[:, 19] + 1) / 2
        # Angle: Map [-π, π] to [0, 1]
        X_chunk[:, 20] = (X_chunk[:, 20] + np.pi) / (2 * np.pi)
        # SpeedX, SpeedY, SpeedZ: Divide by 100 m/s
        X_chunk[:, 21:24] = np.clip(X_chunk[:, 21:24] / 100, 0, 1)
        # Gear: Map [-1, 6] to [0, 1]
        X_chunk[:, 24] = (X_chunk[:, 24] + 1) / 7
        # RPM: Divide by 10000
        X_chunk[:, 25] = np.clip(X_chunk[:, 25] / 10000, 0, 1)
        # MinOpponent: Divide by 200m
        X_chunk[:, 26] = np.clip(X_chunk[:, 26] / 200, 0, 1)
        # Additional features normalization
        X_chunk[:, 27:] = np.clip(X_chunk[:, 27:], -1, 1)  # Clip to [-1, 1] range
        
        # Step 9: Outputs
        # Regression outputs: Steer, Accel, Brake
        y_reg_chunk = chunk[["Steer", "Accel", "Brake"]].values
        # Classification output: Gear_output (one-hot encoded)
        gear_encoder = OneHotEncoder(categories=[[-1, 0, 1, 2, 3, 4, 5, 6]], sparse_output=False)
        y_cls_chunk = gear_encoder.fit_transform(chunk[["Gear_output"]])
        
        # Append to chunks
        X_chunks.append(X_chunk)
        y_reg_chunks.append(y_reg_chunk)
        y_cls_chunks.append(y_cls_chunk)

# Step 10: Combine chunks
if not X_chunks:
    raise ValueError("No valid data processed from CSV files")
X = np.vstack(X_chunks)
y = {
    'regression': np.vstack(y_reg_chunks),  # Steer, Accel, Brake
    'classification': np.vstack(y_cls_chunks)  # Gear_output (one-hot)
}
print(f"Combined data shape: X={X.shape}, y_regression={y['regression'].shape}, y_classification={y['classification'].shape}")

# Step 11: Split data
X_train, X_val, y_reg_train, y_reg_val = train_test_split(
    X, y['regression'], test_size=0.2, random_state=42
)
_, _, y_cls_train, y_cls_val = train_test_split(
    X, y['classification'], test_size=0.2, random_state=42
)

# Step 12: Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_reg_train.npy", y_reg_train)  # Steer, Accel, Brake
np.save("y_cls_train.npy", y_cls_train)  # Gear_output (one-hot)
np.save("X_val.npy", X_val)
np.save("y_reg_val.npy", y_reg_val)
np.save("y_cls_val.npy", y_cls_val)

# Print shapes for verification
print(f"X_train shape: {X_train.shape}")  # (~92807, 32)
print(f"y_reg_train shape: {y_reg_train.shape}")  # (~92807, 3)
print(f"y_cls_train shape: {y_cls_train.shape}")  # (~92807, 8)
print(f"X_val shape: {X_val.shape}")
print(f"y_reg_val shape: {y_reg_val.shape}")
print(f"y_cls_val shape: {y_cls_val.shape}")