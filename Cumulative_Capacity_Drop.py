import pandas as pd
combined_features_df = pd.read_pickle('combined_features_df_capacity_drop_updated.pkl')

combined_features_df = combined_features_df[combined_features_df['Capacity_drop(SOH)'] > -0.2]
# Group by 'cell_number' and calculate cumulative sum of 'Capacity_drop(SOH)'
combined_features_df['cumulative_capacity_drop'] = combined_features_df.groupby('cell_number')['Capacity_drop(SOH)'].cumsum()

# Reset the index after dropping rows
combined_features_df.reset_index(drop=True, inplace=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Determine the group number for each cell
group_mapping = {
    1: [1, 2, 7, 8],
    2: [3, 4, 5, 6],
    3: [25, 26, 27, 28],
    4: [17, 18, 19, 20],
    5: [21, 22, 23, 24],
    6: [13, 14, 15, 16],
    7: [10, 11, 12, 9]
}

# Map cell numbers to group numbers
combined_features_df['group_number'] = np.nan
for group_number, cell_numbers in group_mapping.items():
    combined_features_df.loc[combined_features_df['cell_number'].isin(cell_numbers), 'group_number'] = group_number

# Plot cumulative capacity drop color-coded by group number
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_features_df, x='load_number', y='cumulative_capacity_drop', hue='group_number', palette='tab10')

# Add horizontal dashed lines
plt.axhline(0.2 * 2.1, color='gray', linestyle='--', linewidth=1)
plt.axhline(0.4 * 2.1, color='gray', linestyle='--', linewidth=1)

# Add labels and title
plt.xlabel('', fontsize=14)
plt.ylabel('', fontsize=14)
# plt.title('Cumulative Capacity Drop by Load Number (Grouped by Four Cells)', fontsize=16)

# Set font size of ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=20, bbox_to_anchor=(1, 0.85), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
