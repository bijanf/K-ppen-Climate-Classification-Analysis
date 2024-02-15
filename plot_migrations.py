import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import matplotlib.patches as mpatches
import pickle
import os 
#      -----------------------------------
#              N A M E  L I S T
# Number of top migrations to display
name='historical_middle_CA.png'
#name='ssp585_2071-2100_spain.png'
top_n = 20  # Set this to your desired number of top migrations
x_max_lim = 400000
#scenario='ssp585'
#scenario='ssp370'
scenario='ssp126'
time_slice='2071-2100'
#time_slice='2011-2040'
#time_slice='2041-2070'
bbox = [45, 34, 90, 56] #Central Asia
#          E N D  OF N A M E L I S T
#      -------------------------------------

koppen_mapping_short = {
    1: "Af ",
    2: "Am ",
    3: "As ",
    4: "Aw ",
    5: "BWk",
    6: "BWh",
    7: "BSk",
    8: "BSh",
    9: "Cfa",
    10: "Cfb",
    11: "Cfc",
    12: "Csa",
    13: "Csb",
    14: "Csc",
    15: "Cwa",
    16: "Cwb",
    17: "Cwc",
    18: "Dfa",
    19: "Dfb",
    20: "Dfc",
    21: "Dfd",
    22: "Dsa",
    23: "Dsb",
    24: "Dsc",
    25: "Dsd",
    26: "Dwa",
    27: "Dwb",
    28: "Dwc",
    29: "Dwd",
    30: "ET ",
    31: "EF "
}


# Function to transform latitude/longitude to pixel coordinates
def world_to_pixel(geo_matrix, x, y):
    ulX = geo_matrix[0]
    ulY = geo_matrix[3]
    xDist = geo_matrix[1]
    yDist = geo_matrix[5]
    rtnX = geo_matrix[2]
    rtnY = geo_matrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)



# Load the historical and future climate classification TIFF files
historical_dataset = gdal.Open('/p/projects/gvca/data/chelsa_cmip6/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies/1981-2010/bio/CHELSA_kg2_1981-2010_V.2.1.tif')

dir_models = '/p/projects/gvca/data/chelsa_cmip6/envicloud/chelsa/chelsa_V2/GLOBAL/climatologies/'+time_slice+'/'
models = [dir_models+'GFDL-ESM4/'+scenario+'/bio/CHELSA_kg2_'+time_slice+'_gfdl-esm4_'+scenario+'_V.2.1.tif',
dir_models+'IPSL-CM6A-LR/'+scenario+'/bio/CHELSA_kg2_'+time_slice+'_ipsl-cm6a-lr_'+scenario+'_V.2.1.tif',
dir_models+'MPI-ESM1-2-HR/'+scenario+'/bio/CHELSA_kg2_'+time_slice+'_mpi-esm1-2-hr_'+scenario+'_V.2.1.tif',
dir_models+'MRI-ESM2-0/'+scenario+'/bio/CHELSA_kg2_'+time_slice+'_mri-esm2-0_'+scenario+'_V.2.1.tif',
dir_models+'UKESM1-0-LL/'+scenario+'/bio/CHELSA_kg2_'+time_slice+'_ukesm1-0-ll_'+scenario+'_V.2.1.tif']





geo_transform = historical_dataset.GetGeoTransform()
minx, maxy = world_to_pixel(geo_transform, bbox[0], bbox[1])
maxx, miny = world_to_pixel(geo_transform, bbox[2], bbox[3])
# Read the raster data as numpy arrays for the specified region
historical_array  = historical_dataset.ReadAsArray(minx, miny, maxx - minx, maxy - miny)






# Adjusted Function to process each model and calculate changes focusing on ET migrations
def process_model(model_path):
    future_dataset = gdal.Open(model_path)
    future_array = future_dataset.ReadAsArray(minx, miny, maxx - minx, maxy - miny)

    changes_dict_model = {}
    for i in range(historical_array.shape[0]):
        for j in range(historical_array.shape[1]):
            historical_class = historical_array[i, j]
            future_class = future_array[i, j]
            if historical_class == 30 and historical_class != future_class:  # Check if historical class is ET
                change_pair = (historical_class, future_class)
                if change_pair not in changes_dict_model:
                    changes_dict_model[change_pair] = 1
                else:
                    changes_dict_model[change_pair] += 1
    return changes_dict_model



# Process each model and store the results
all_changes = [process_model(model) for model in models]

# Aggregate changes to calculate mean and standard deviation
aggregated_changes = {}
for changes in all_changes:
    for change, count in changes.items():
        if change not in aggregated_changes:
            aggregated_changes[change] = []
        aggregated_changes[change].append(count)

means = {change: np.mean(counts) for change, counts in aggregated_changes.items()}
std_devs = {change: np.std(counts) for change, counts in aggregated_changes.items()}


# Directly proceed to plotting without filtering for top N
sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)  # Sort all ET migrations
migrations = [f"{koppen_mapping_short.get(src)} -> {koppen_mapping_short.get(dst)}" for (src, dst), _ in sorted_means if src == 30]
mean_counts = [mean for _, mean in sorted_means if _[0] == 30]
error = [std_devs[change] for change, _ in sorted_means if change[0] == 30]

# Adjusted plot code to reflect ET specific migrations
plt.figure(figsize=(10, 8))
plt.barh(migrations, mean_counts, xerr=error, color='skyblue')
plt.xlabel('Mean Count of Migrations (km$^2$)', fontsize= 15)
plt.ylabel('Migration Type', fontsize= 15)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 15)
plt.gca().invert_yaxis()
plt.xlim(0, x_max_lim)
plt.ylim(0,12)
plt.tight_layout()

# Save the plot
plt.savefig('ET_classification_aggregated_migrations_'+time_slice+'_'+scenario+'.png')
