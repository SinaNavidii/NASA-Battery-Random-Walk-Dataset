import numpy as np
import pandas as pd

def extract_features_and_save(RPT_sol_loaded, num_iterations, num_repeats, i):
    RW_cycles = []
    RW_capacities = []
    RW_throughputs = []  # Charge throughput for each 50 RW cycles
    RW_times = []  # Time of each load pattern
    absolute_times = []  # Absolute time from the start
    duration_below_2A = []  # Duration below 2A
    duration_between_2A_3A = []  # Duration between 2A and 3A
    duration_above_3A = []  # Duration above 3A

    RPT_cycles = []
    RPT_capacities = []

    total_absolute_time = 0

    for i in range(num_repeats):
        charge_throughput = 0
        load_pattern_time = 0
        time_below_2A = 0
        time_between_2A_3A = 0
        time_above_3A = 0
        
        for j in range(num_iterations):
            cycle_number = i * (num_iterations + 1) + j + 1
            RW_cycles.append(cycle_number)
            
            # Calculate capacities
            start_capacity = (
                RPT_sol_loaded.cycles[cycle_number - 1]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[0]
            )
            end_capacity = (
                RPT_sol_loaded.cycles[cycle_number - 1]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[-1]
            )
            RW_capacities.append(end_capacity - start_capacity)
            
            # Extract current and time data
            current_data = RPT_sol_loaded.cycles[cycle_number - 1]
            current_entries_for_Q_thru = current_data.steps[2]["Current [A]"].entries
            current_entries_for_current_durations = current_data.steps[0]["Current [A]"].entries
            time_entries = current_data.steps[2]["Time [s]"].entries
            
            # Calculate charge throughput and time
            for k in range(len(current_entries_for_Q_thru) - 1):
                current = current_entries_for_Q_thru[k]
                delta_time = time_entries[k + 1] - time_entries[k]
                charge_throughput += np.abs(current) * delta_time / 3600  # Convert to A.h
                
            for k in range(len(current_entries_for_current_durations) - 1):
                current = current_entries_for_current_durations[k]
                if current < 2:
                    time_below_2A += delta_time
                elif 2 <= current < 3:
                    time_between_2A_3A += delta_time
                else:
                    time_above_3A += delta_time
            
            load_pattern_time += time_entries[-1]
        
        # Add throughput and times for the current load pattern
        RW_throughputs.append(charge_throughput)
        RW_times.append(load_pattern_time)
        duration_below_2A.append(time_below_2A)
        duration_between_2A_3A.append(time_between_2A_3A)
        duration_above_3A.append(time_above_3A)
        total_absolute_time += load_pattern_time
        absolute_times.append(total_absolute_time)
        
        # RPT cycle
        RPT_cycle_number = (i + 1) * (num_iterations + 1)
        RPT_cycles.append(RPT_cycle_number / 20)
        start_capacity = RPT_sol_loaded.cycles[RPT_cycle_number - 1][
            "Discharge capacity [A.h]"
        ].entries[0]
        end_capacity = RPT_sol_loaded.cycles[RPT_cycle_number - 1][
            "Discharge capacity [A.h]"
        ].entries[-1]
        RPT_capacities.append(end_capacity - start_capacity)

    # Calculate capacity drop for each RPT cycle
    capacity_drop = [2.1 - 1.948487]  # Assuming the first RPT cycle starts with full capacity
    for i in range(1, len(RPT_capacities)):
        drop = RPT_capacities[i - 1] - RPT_capacities[i]
        capacity_drop.append(drop)

    # Calculate cumulative capacity drop for each load pattern
    cumulative_capacity_drop_all = np.cumsum(capacity_drop)

    # Create a DataFrame for features and target
    data = {
        'charge_throughput': RW_throughputs,
        'cell_capacity': RPT_capacities,  # Use RPT cell capacities
        'absolute_time': absolute_times,
        'total_time': RW_times,
        'duration_below_2A': duration_below_2A,
        'duration_between_2A_3A': duration_between_2A_3A,
        'duration_above_3A': duration_above_3A,
        'capacity_drop': capacity_drop,
        'cumulative_capacity_drop': cumulative_capacity_drop_all
    }

    df = pd.DataFrame(data)

    # Add 'load_number' and 'cell_number' columns
    df['load_number'] = df.index + 1  # Calculate load number based on index
    df['cell_number'] = f'sim_{i}'  # Fill cell_number column with 'sim_i'

    # Save DataFrame as a pickle file
    pickle_path = f'Aug_data/Sim_{i}.pkl'
    df.to_pickle(pickle_path)