import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os
from tqdm import tqdm
import csv



# Assuming `aggregated_changes_per_slice` is your dictionary with the data
csv_file_path = 'aggregated_changes.csv'



# Configuration
scenarios = ['ssp126', 'ssp370', 'ssp585']
time_slices = ['2011-2040', '2041-2070', '2071-2100']
models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
base_dir = '/p/projects/gvca/data/chelsa_cmip6/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies'
colors = ['blue', 'green', 'red']  # Colors for the time slices

koppen_mapping_short = {
    1: "Af", 2: "Am", 3: "As", 4: "Aw", 5: "BWk", 6: "BWh", 7: "BSk", 8: "BSh",
    9: "Cfa", 10: "Cfb", 11: "Cfc", 12: "Csa", 13: "Csb", 14: "Csc", 15: "Cwa",
    16: "Cwb", 17: "Cwc", 18: "Dfa", 19: "Dfb", 20: "Dfc", 21: "Dfd", 22: "Dsa",
    23: "Dsb", 24: "Dsc", 25: "Dsd", 26: "Dwa", 27: "Dwb", 28: "Dwc", 29: "Dwd",
    30: "ET", 31: "EF"
}

koppen_colors = {
    1: '#960000',
    2: '#ff0000',
    3: '#ffcccc',
    4: '#ffcc00',
    5: '#ffff64',
    6: '#cc8d14',
    7: '#ccaa54',
    8: '#00ff00',
    9: '#96ff00',
    10: '#c8ff00',
    11: '#b46400',
    12: '#966400',
    13: '#5a3c00',
    14: '#003200',
    15: '#005000',
    16: '#007800',
    17: '#ff6eff',
    18: '#ffb4ff',
    19: '#e6c8ff',
    20: '#c8c8c8',
    21: '#c8b4ff',
    22: '#9a7fb3',
    23: '#8759b3',
    24: '#6f24b3',
    25: '#320032',
    26: '#640064',
    27: '#c800c8',
    28: '#c81485',
    29: '#64ffff',
    30: '#6496ff',
    31: '#6000ff',
}


#bbox = [45, 34, 90, 56]  # Define your bounding box
bbox = [4.5, 43.5, 16, 50]  #Alps
# Function to convert lat/long to pixel coordinates
def world_to_pixel(geo_matrix, x, y):
    ulX, ulY, xDist, yDist = geo_matrix[0], geo_matrix[3], geo_matrix[1], geo_matrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / -yDist)
    return (pixel, line)

# Function to process a single model file
def process_model(model_path, historical_array, minx, miny, maxx, maxy):
    future_dataset = gdal.Open(model_path)
    future_array = future_dataset.ReadAsArray(minx, miny, maxx - minx, maxy - miny)
    changes_dict_model = {}

    # Calculate total iterations for the progress bar
    total_iterations = future_array.shape[0] * future_array.shape[1]

    with tqdm(total=total_iterations, desc="Processing Model") as pbar:
        for i in range(future_array.shape[0]):
            for j in range(future_array.shape[1]):
                historical_class = historical_array[i, j]
                future_class = future_array[i, j]
                if historical_class == 30:
                    if future_class != historical_class:  # Check if there's a change
                        change_pair = (historical_class, future_class)
                        changes_dict_model[change_pair] = changes_dict_model.get(change_pair, 0) + 1
                # Update progress bar after each inner loop iteration
                pbar.update(1)
    return changes_dict_model



# Load the historical climate classification TIFF
historical_dataset_path = os.path.join(base_dir, '1981-2010', 'bio', 'CHELSA_kg2_1981-2010_V.2.1.tif')
historical_dataset = gdal.Open(historical_dataset_path)
geo_transform = historical_dataset.GetGeoTransform()
minx, maxy = world_to_pixel(geo_transform, bbox[0], bbox[1])
maxx, miny = world_to_pixel(geo_transform, bbox[2], bbox[3])
historical_array = historical_dataset.ReadAsArray(minx, miny, maxx - minx, maxy - miny)
historical_ET_count = np.sum(historical_array == 30)



# Placeholder for the ensemble mean changes per class for each time slice and scenario
ensemble_mean_changes = {scenario: {time_slice: {} for time_slice in time_slices} for scenario in scenarios}


# Process the models and calculate the ensemble mean changes
for scenario in scenarios:
    for time_slice in time_slices:
        model_changes = []
        for model in models:
            dir_path = os.path.join(base_dir, time_slice, model, scenario, 'bio')
            if os.path.exists(dir_path):
                model_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if "_kg2_" in f and f.endswith('.tif')]
                for model_path in tqdm(model_paths, desc=f'Processing {model} {time_slice} {scenario}'):
                    changes = process_model(model_path, historical_array, minx, miny, maxx, maxy)
                    model_changes.append(changes)
            else:
                print(f"Directory does not exist: {dir_path}")

        # Aggregate changes across all models and calculate the ensemble mean
        aggregated_changes = {}
        for changes in model_changes:
            for change, count in changes.items():
                if change[0] == 30:  # Ensure we're only looking at changes from ET class
                    aggregated_changes[change] = aggregated_changes.get(change, 0) + count

        # Calculate the mean changes for this scenario and time slice
        ensemble_mean_changes[scenario][time_slice] = {change: count / len(models) for change, count in aggregated_changes.items()}

# Configuration for plot size and font
plt.rcParams['font.size'] = 30  # Adjust font size as needed

# Setup the figure layout
num_scenarios = len(scenarios)
num_cols = len(time_slices)

# Create a figure to accommodate the adjusted layout
fig, axs = plt.subplots(num_scenarios, num_cols, figsize=(num_cols * 6, num_scenarios * 6))  # Adjust the figsize to your needs

# Function to format the labels with percentages and count
def format_labels(sizes, labels):
    total = sum(sizes)
    labels_with_pct = [f'{label}\n({size/total:.1%})' for size, label in zip(sizes, labels)]
    return labels_with_pct

# Function to filter out sizes and labels where the percentage is less than 0.01%
def filter_small_percentages(sizes, labels, threshold=0.15):
    filtered_sizes = []
    filtered_labels = []
    total = sum(sizes)
    for size, label in zip(sizes, labels):
        if (size / total) * 100 > threshold:
            filtered_sizes.append(size)
            filtered_labels.append(label)
    return filtered_sizes, filtered_labels

# Loop through scenarios and time slices to plot the ensemble mean changes
for i, scenario in enumerate(scenarios):
    for j, time_slice in enumerate(time_slices):
        # Get the mean changes for the current scenario and time slice
        mean_changes = ensemble_mean_changes[scenario][time_slice]

        labels = [koppen_mapping_short.get(change[1], 'Other') for change in mean_changes]
        sizes = [mean_changes[change] for change in mean_changes]
        colors = [koppen_colors.get(change[1], 'grey') for change in mean_changes]

        # Filter sizes and labels to exclude values below 0.01%
        filtered_sizes, filtered_labels = filter_small_percentages(sizes, labels)

        colors = [koppen_colors.get(change[1], 'grey') for change, count in zip(mean_changes, filtered_sizes) if count > 0]

        # Calculate the number of ET grids that remain unchanged
        total_changed_ET = sum(filtered_sizes)
        unchanged_ET = historical_ET_count - total_changed_ET
        if (unchanged_ET / historical_ET_count) * 100 > 0.01:
            filtered_labels.append('ET')
            filtered_sizes.append(unchanged_ET)
            colors.append(koppen_colors[30])  # Color for unchanged ET

        # Format the labels with percentages and count
        formatted_labels = format_labels(filtered_sizes, filtered_labels)

        # Plot pie chart for the current scenario and time slice
        ax = axs[i, j]  # Current subplot
        wedges, texts = ax.pie(filtered_sizes, colors=colors, startangle=90, labels=formatted_labels, labeldistance=.60)

        # Adjust the position of the labels to be outside the wedges
        for text in texts:
            text.set_horizontalalignment('center')

        # Set the title for the current pie chart
        ax.set_title(f'{scenario} {time_slice}', fontsize=35, fontweight='bold', bbox=dict(facecolor='none', edgecolor='none', pad=5))
        
plt.tight_layout()
# Place the legend on the figure
plt.savefig('aggregated_migrations_by_scenario_adjusted_ALPS.png',bbox_inches='tight', dpi=300)






























