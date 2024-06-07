import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch




# Select features and target variable
features = [ 'charge_throughput', 'cell_capacity',    'absolute_time', 'total_time',
            'duration_below_2A', 'duration_between_2A_3A', 'duration_above_3A']
target = 'cumulative_capacity_drop'

# Separate even and odd cell numbers
even_cells = combined_features_df[combined_features_df['cell_number'] % 2 == 0]
odd_cells = combined_features_df[combined_features_df['cell_number'] % 2 != 0]

# Function to create sequences for each cell
def create_sequences(data):
    sequences = []
    for cell_number, group in data.groupby('cell_number'):
        group = group.sort_values(by='load_number', ascending=True)  # Sort by load number
        num_rows = len(group)
        if num_rows >= 11:  
            for i in range(num_rows - 10):
                sequence = group.iloc[i:i + 10][features].values.flatten()
                target_value = group.iloc[i + 10][target]  # Target is the capacity for the next 10th load number
                sequences.append((sequence, target_value))
    return sequences

# Create sequences for even cells
even_sequences = create_sequences(even_cells)

# Split even sequences into train and validation sets
train_sequences, val_sequences = train_test_split(even_sequences, test_size=0.5, random_state=42)

# Create sequences for odd cells (for testing)
test_sequences = create_sequences(odd_cells)

# Separate features and targets for train, validation, and test sets
train_features, train_targets = zip(*[(seq[0], seq[1]) for seq in train_sequences])
val_features, val_targets = zip(*[(seq[0], seq[1]) for seq in val_sequences])
test_features, test_targets = zip(*[(seq[0], seq[1]) for seq in test_sequences])

# Normalize the features
scaler = MinMaxScaler()
scaler.fit(train_features)

# Normalize train, validation, and test features
train_features_scaled = scaler.transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)


# Convert to PyTorch tensors
inputs_train = torch.tensor(train_features_scaled, dtype=torch.float32).unsqueeze(1)
targets_train = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)

inputs_val = torch.tensor(val_features, dtype=torch.float32).unsqueeze(1)
targets_val = torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)

inputs_test = torch.tensor(test_features, dtype=torch.float32).unsqueeze(1)
targets_test = torch.tensor(test_targets, dtype=torch.float32).unsqueeze(1)