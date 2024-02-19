import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os
from tqdm import tqdm
import csv
from matplotlib.patches import Patch


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


##bbox = [45, 34, 46, 36]  # Define your bounding box
bbox = [4.5, 43.5, 16, 50]  #Alps
##bbox  = [65, 25,105,56] #Asia 
# Function to convert lat/long to pixel coordinates
def world_to_pixel(geo_matrix, x, y):
    ulX, ulY, xDist, yDist = geo_matrix[0], geo_matrix[3], geo_matrix[1], geo_matrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / -yDist)
    return (pixel, line)
# Function to check if the label should be inside or outside and draw a line
def autopct_generator(limit):
    def inner_autopct(pct):
        return ('%.1f%%' % pct) if pct > limit else ''
    return inner_autopct

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

# Function to calculate label positions with a line pointing to small slices
def label_position(wedge, text, ax):
    ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(
        text.get_text(), xy=(x, y), xytext=(1.5*x, 1.5*y),
        arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle),
        ha=horizontalalignment
    )
    text.set_visible(False)

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

# Create a set to collect unique categories that are present in the pie charts
unique_categories = set()

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
        mean_changes = ensemble_mean_changes[scenario][time_slice]
        for change, count in mean_changes.items():
            print("count", str(count), change)
            if count > 0:
                unique_categories.add(change[1])  # Add the category index to the set
                
        # Calculate the mean changes for this scenario and time slice
        ensemble_mean_changes[scenario][time_slice] = {change: count / len(models) for change, count in aggregated_changes.items()}


# Now create legend patches for only the unique categories that were used
def create_legend_patches(unique_categories, koppen_colors, koppen_mapping_short):
    patches = []
    # Create a reverse mapping from the koppen_mapping_short
    reverse_koppen_mapping = {v: k for k, v in koppen_mapping_short.items()}
    for category in unique_categories:
        label = koppen_mapping_short[reverse_koppen_mapping[category]]
        color = koppen_colors[reverse_koppen_mapping[category]]
        patches.append(Patch(color=color, label=label))
    return patches



# legend_patches = create_legend_patches(unique_categories, koppen_colors, koppen_mapping_short)



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
    labels_with_pct = [f'{label}' for size, label in zip(sizes, labels)]
    return labels_with_pct

# Function to check if a label should go inside or outside the pie
def label_function(sizes, threshold=0.05):
    def func(pct):
        return '' if pct < threshold else f'{pct:.1f}%'
    return func



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
        # Define the autopct function with a threshold
        autopct = label_function(filtered_sizes)

        colors = [koppen_colors.get(change[1], 'grey') for change, count in zip(mean_changes, filtered_sizes) if count > 0]

        # Calculate the number of ET grids that remain unchanged
        total_changed_ET = sum(filtered_sizes)
        unchanged_ET = historical_ET_count - total_changed_ET
        if historical_ET_count > 0:
            if (unchanged_ET / historical_ET_count) * 100 > 0.01:
                filtered_labels.append('ET')
                filtered_sizes.append(unchanged_ET)
                colors.append(koppen_colors[30])  # Color for unchanged ET

        # Format the labels with percentages and count
        formatted_labels = format_labels(filtered_sizes, filtered_labels)

        # Plot pie chart for the current scenario and time slice
#        ax = axs[i, j]  # Current subplot
#        wedges, texts = ax.pie(filtered_sizes, colors=colors, startangle=90, labels=formatted_labels, labeldistance=.55)

#        # Adjust the position of the labels to be outside the wedges
#        for text in texts:
#            text.set_horizontalalignment('center')

        # Determine which slices to explode out based on a size threshold
        explode_threshold = 0.15  # Example threshold percentage (1.5%)
        explode = [0.1 if (size / sum(sizes)) < explode_threshold else 0 for size in filtered_sizes]
        # Plot pie chart for the current scenario and time slice
        ax = axs[i, j]  # Current subplot
        # This threshold represents the percentage limit under which labels will be outside
        threshold = 5
        # Calculate the percentages for the filtered sizes
        filtered_sizes_percentages = [size / sum(filtered_sizes) * 100 for size in filtered_sizes]

        # Create pie chart
        wedges, _ = ax.pie(
            filtered_sizes, 
            colors=colors, 
#            labels=formatted_labels, 
            startangle=90 
#            autopct=autopct_generator(threshold),
#            pctdistance=0.6  # Adjust this as needed
        )

        print(formatted_labels,"formatted_lables\n")
        unique_categories.update(formatted_labels)
        # Set the title for the current pie chart
        ax.set_title(f'{scenario} {time_slice}', fontsize=35, fontweight='bold')
#        # Set the title for the current pie chart
#        ax.set_title(f'{scenario} {time_slice}', fontsize=35, fontweight='bold', bbox=dict(facecolor='none', edgecolor='none', pad=5))


unique_labels = set()
print(unique_categories)
## Loop through each list and add labels to the set
#for label_list in unique_categories:
#    unique_labels.update(label_list)
##print(unique_labels)

## Convert the set to a list to get your final list of unique labels
#unique_labels = list(unique_labels)

##unique_labels.sort()

##print(unique_labels)

legend_patches = create_legend_patches(unique_categories, koppen_colors, koppen_mapping_short)
#print(unique_categories)
# Add a legend for the whole figure
plt.tight_layout()
plt.subplots_adjust(bottom=0.05)
fig.legend(handles=legend_patches, loc='upper center', ncol=len(legend_patches), fontsize=35, bbox_to_anchor=(0.5, -0.025))
# Place the legend on the figure
plt.savefig('aggregated_migrations_by_scenario_adjusted_ALPS.png',bbox_inches='tight', dpi=300)
#plt.savefig('aggregated_migrations_by_scenario_adjusted_ASIA.png',bbox_inches='tight', dpi=300)





























