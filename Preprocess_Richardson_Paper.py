import os
import scipy.io
import pandas as pd
import numpy as np

def process_directory(directory):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over the files in the directory
    for file_name in os.listdir(directory):
        if file_name.startswith('RW') and file_name.endswith('.mat'):
            # Load the MATLAB data file
            mat_data = scipy.io.loadmat(os.path.join(directory, file_name))
            # Extract the data and create a DataFrame
            data = pd.DataFrame(mat_data['data'][0][0][0][0])
            # Add a column for RW number
            data['Cell_number'] = int(file_name.split('.')[0][2:])
            # Append the DataFrame to the list
            dfs.append(data)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Initialize a column for load numbers
    combined_df['load_number'] = 0

    # Initialize a column for ref numbers
    combined_df['ref_number'] = 0

    # Iterate over each RW_number separately
    for rw_number in combined_df['Cell_number'].unique():
    
        load_number = 0
        ref_number = 0
        # Filter the DataFrame for the current rw_number
        rw_data = combined_df[combined_df['Cell_number'] == rw_number]
        
        # Iterate over cycles in the current RW_number
        for cycle_index in range(len(rw_data)):
            # Check if the comment contains "random walk"
            if 'random walk' in str(rw_data['comment'].iloc[cycle_index]).lower():
                # Increment load number only if it's the first cycle of a new batch of consecutive "random walk" cycles
                if cycle_index == 0 or 'random walk' not in str(rw_data['comment'].iloc[cycle_index - 1]).lower():
                    load_number += 1
                # Assign the load number to the corresponding row
                combined_df.loc[(combined_df['Cell_number'] == rw_number) & (combined_df.index == rw_data.index[cycle_index]), 'load_number'] = load_number
            
            if 'reference' in str(rw_data['comment'].iloc[cycle_index]).lower():
                # Increment ref number only if it's the first cycle of a new batch of consecutive "reference" cycles
                if cycle_index == 0 or 'reference' not in str(rw_data['comment'].iloc[cycle_index - 1]).lower():
                    ref_number += 1
                # Assign the ref number to the corresponding row
                combined_df.loc[(combined_df['Cell_number'] == rw_number) & (combined_df.index == rw_data.index[cycle_index]), 'ref_number'] = ref_number

    # Initialize an empty list to store feature dictionaries
    feature_dicts = []

    # Iterate over each cell number separately
    for cell_number in combined_df['Cell_number'].unique():
        # Filter the DataFrame to include only the rows with the current cell number
        cell_data = combined_df[combined_df['Cell_number'] == cell_number]

        ### Calculate the initial capacity for the specified cell ###
        # ref_data = cell_data[(cell_data['ref_number'] == 1)]
        
        # for cycle_index, row in ref_data.iterrows():
        #     current = ref_data['current'][cycle_index]
        #     time = ref_data['time'][cycle_index]
            
        #     dt = np.diff(time)
        #     current = current[:, :-1]
        #     capacity = np.cumsum(current * dt) / 3600
        # Q_cell_init = capacity.max()

        # ASSUME THAT THE INITIAL CAP IS 2.1 Ah FOR ALL THE CELLS
        Q_cell_init = 2.1 # Ah 
        

        # Iterate over each load number separately for this cell
        for load_number in cell_data['load_number'].unique():
            # Filter the DataFrame to include only the rows with the current load number
            load_data = cell_data[cell_data['load_number'] == load_number]

            # First feature: Total time elapsed during load number
            first_time = load_data['time'].apply(lambda x: x[0].min()).min()
            last_time = load_data['time'].apply(lambda x: x[0].max()).max()
            total_time_load = last_time - first_time

            # Second feature: Charge throughput during the load pattern
            charge_throughputs = []
            for cycle_index, row in load_data.iterrows():
                current = load_data['current'][cycle_index]
                time = load_data['time'][cycle_index]
                
                dt = np.diff(time)
                current = current[:, :-1]
                capacity = np.sum(np.abs(current) * dt) / 3600
                charge_throughputs.append(capacity)
            Q_thru = np.abs(np.mean(charge_throughputs))

            # Third feature: Absolute time value since the beginning of the dataset
            last_time_load = load_data.iloc[-1]['time'][0].max()
            start_time_combined_df = combined_df.iloc[0]['time'][0].min()
            absolute_time = last_time_load - start_time_combined_df

            # Fourth feature: Cell capacity during the load pattern (based on the ref cycle with the same index)
            capacities = []
            ref_data = cell_data[(cell_data['ref_number'] == load_number + 1 )] 
            for cycle_index, row in ref_data.iterrows():
                current = ref_data['current'][cycle_index]
                time = ref_data['time'][cycle_index]
                
                dt = np.diff(time)
                current = current[:, :-1]
                capacity = np.cumsum(current * dt) / 3600
                if len(capacity) > 0:  # Check if capacity array is not empty
                    capacities.append(capacity.max()) 
            Q_cell = np.max(capacities) if capacities else 0


            # Calculate previous ref_number capacity to find the capacity drop 
            capacities_ = []
            ref_data_ = cell_data[(cell_data['ref_number'] == load_number )] 
            for cycle_index, row in ref_data_.iterrows():
                current = ref_data_['current'][cycle_index]
                time = ref_data_['time'][cycle_index]
                
                dt = np.diff(time)
                current = current[:, :-1]
                capacity_ = np.cumsum(current * dt) / 3600
                if len(capacity_) > 0:  # Check if capacity array is not empty
                    capacities_.append(capacity_.max()) 
            Q_cell_previous = np.max(capacities_) if capacities_ else 0

        
            # Fifth, sixth, and seventh features: Duration below 0°C, between 0 and 40°C, and above 40°C
            duration_below_5 = 0
            duration_between_5_40 = 0
            duration_above_40 = 0
            for _, row in load_data.iterrows():
                times = row['time'][0]
                temperatures = row['temperature'][0]
                for i in range(len(times)):
                    temperature_reading = temperatures[i]
                    time_difference = times[i] - times[0]
                    if temperature_reading < 5:
                        duration_below_5 += time_difference
                    elif 0 <= temperature_reading <= 40:
                        duration_between_5_40 += time_difference
                    else:
                        duration_above_40 += time_difference

            # eigth, nineth, and tenthth features: Duration below 2A, between 2A and 3A, and above 3A
            duration_below_2A = 0
            duration_between_2A_3A = 0
            duration_above_3A = 0
            for _, row in load_data.iterrows():
                times = row['time'][0]
                currents = row['current'][0]
                for i in range(len(times)):
                    current_reading = currents[i]
                    time_difference = times[i] - times[0]
                    if np.abs(current_reading) < 2:
                        duration_below_2A += time_difference
                    elif 2 <= np.abs(current_reading) <= 3:
                        duration_between_2A_3A += time_difference
                    else:
                        duration_above_3A += time_difference


            
            # Store the calculated features in a dictionary
            feature_dict = {
                'cell_number': cell_number,
                'load_number': load_number,
                'total_time': total_time_load,
                'charge_throughput': Q_thru,
                'absolute_time': absolute_time,
                'cell_capacity': Q_cell,
                'duration_below_5': duration_below_5,
                'duration_between_5_40': duration_between_5_40,
                'duration_above_40': duration_above_40,
                'duration_below_2A': duration_below_2A,
                'duration_between_2A_3A': duration_between_2A_3A,
                'duration_above_3A': duration_above_3A,
                'Capacity_drop(SOH)' : Q_cell_previous - Q_cell
            }

            # Append the feature dictionary to the list
            feature_dicts.append(feature_dict)

    # Convert the list of feature dictionaries to a DataFrame
    features_df = pd.DataFrame(feature_dicts)

    # Return the resulting DataFrame
    return features_df


# Define the directories to process
directories = [
    'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/',
    'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/',
    'RW_Skewed_High_40C_DataSet_2Post/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/',
    'RW_Skewed_High_Room_Temp_DataSet_2Post/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/',
    'RW_Skewed_Low_40C_DataSet_2Post/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/',
    'RW_Skewed_Low_Room_Temp_DataSet_2Post/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/',
    'Battery_Uniform_Distribution_Charge_Disc/Battery_Uniform_Distribution_Charge_Disc/data/Matlab/',

]

# Initialize an empty list to store feature DataFrames from each directory
all_features_dfs = []

# Process each directory and append the resulting feature DataFrame to the list
for directory in directories:
    features_df = process_directory(directory)
    all_features_dfs.append(features_df)

# Concatenate all feature DataFrames into a single DataFrame
combined_features_df = pd.concat(all_features_dfs, ignore_index=True)

# Drop rows where cell_capacity is equal to zero
combined_features_df = combined_features_df[combined_features_df['cell_capacity'] != 0]

combined_features_df = combined_features_df[combined_features_df['Capacity_drop(SOH)'] > -0.2]

# Group by 'cell_number' and calculate cumulative sum of 'Capacity_drop(SOH)'
combined_features_df['cumulative_capacity_drop'] = combined_features_df.groupby('cell_number')['Capacity_drop(SOH)'].cumsum()
#fix absolute time 
combined_features_df['absolute_time'] = combined_features_df.groupby('cell_number')['total_time'].cumsum()


# Reset the index after dropping rows
combined_features_df.reset_index(drop=True, inplace=True)


# Print the resulting combined feature DataFrame
combined_features_df
