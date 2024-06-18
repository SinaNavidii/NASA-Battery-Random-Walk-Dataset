import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

combined_features_df = pd.read_pickle('../NASA dataset/archive/dataset/dataset/combined_features_df_capacity_drop_updated.pkl')
combined_features_df = combined_features_df[combined_features_df['Capacity_drop(SOH)'] > -0.2]
# Group by 'cell_number' and calculate cumulative sum of 'Capacity_drop(SOH)'
combined_features_df['cumulative_capacity_drop'] = combined_features_df.groupby('cell_number')['Capacity_drop(SOH)'].cumsum()
#fix absolute time 
combined_features_df['absolute_time'] = combined_features_df.groupby('cell_number')['total_time'].cumsum()



# # Reset the index after dropping rows
combined_features_df.reset_index(drop=True, inplace=True)

# Select features and target variable
features = ['charge_throughput', 'cell_capacity', 'absolute_time', 'total_time',
            'duration_below_2A', 'duration_between_2A_3A', 'duration_above_3A']
target = 'cumulative_capacity_drop'

# Load additional data from pickle files
file_paths = ['Aug_data/Sim1_30RPT_50RWcycles_(0.5-1)_25C.pkl', 'Aug_data/Sim2_10RPT_50RWcycles_(0.5-5)_40C.pkl', 'Aug_data/Sim3_20RPT_25RWcycles_(0.5-1.5).pkl','Aug_data/Sim4_20RPT_50RWcycles_(0.5-4)_ChrgTimeVariable.pkl','Aug_data/Sim6_20RPT_50RWcycles_(0.5-5)_40C.pkl']
additional_dfs = [pd.read_pickle(file_path) for file_path in file_paths]

# Combine all additional dataframes into one
additional_data_df = pd.concat(additional_dfs, ignore_index=True)

# Function to create sequences for each dataset
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

# Create sequences for the original dataset
train_cells = combined_features_df[combined_features_df['cell_number'] % 4 == 0]   # Increase the {4} to see the effect of augmentation! -> 2,4,8,12
test_cells = combined_features_df[combined_features_df['cell_number'] % 4 != 0]
train_sequences = create_sequences(train_cells)
test_sequences = create_sequences(test_cells)

# Create sequences for the additional data
additional_sequences = create_sequences(additional_data_df)

# Combine additional sequences with original training sequences
train_sequences.extend(additional_sequences)

# Separate features and targets for train and test sets
train_features, train_targets = zip(*[(seq[0], seq[1]) for seq in train_sequences])
test_features, test_targets = zip(*[(seq[0], seq[1]) for seq in test_sequences])

# Normalize the features
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Convert to numpy arrays
train_features_scaled = np.array(train_features_scaled)
train_targets = np.array(train_targets).reshape(-1, 1)
test_features_scaled = np.array(test_features_scaled)
test_targets = np.array(test_targets).reshape(-1, 1)

# Elastic Net
en_model = ElasticNet(alpha=0.01, l1_ratio=0.001)
en_model.fit(train_features_scaled, train_targets)

# Predictions for Elastic Net
en_pred = en_model.predict(test_features_scaled)

# Calculate MAE and RMSE for Elastic Net
en_mae = mean_absolute_error(test_targets, en_pred)
en_rmse = np.sqrt(mean_squared_error(test_targets, en_pred))

# Plot predicted vs true cumulative capacity drop for Elastic Net
plt.figure(figsize=(6, 5.5))
plt.scatter(test_targets, en_pred, s=85, c='crimson', alpha=0.6, edgecolors='w')
plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2)
plt.xlabel('True Cumulative Capacity Drop', fontsize=20)
plt.ylabel('Predicted Cumulative Capacity Drop', fontsize=20)

plt.xticks([0.4,0.6,0.8,1,1.2,1.4])
plt.yticks([0.4,0.6,0.8,1,1.2,1.4])


# Add MAE and RMSE to the plot in a box
textstr = f'MAE: {en_mae:.4f}\nRMSE: {en_rmse:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.gca().text(0.6, 0.05, textstr, transform=plt.gca().transAxes, fontsize=20, verticalalignment='bottom', bbox=props)

plt.tight_layout()


plt.savefig("Exported Figs/Augmented_Enet_%4Split_v0.svg", format='svg', dpi=600)

plt.show()

