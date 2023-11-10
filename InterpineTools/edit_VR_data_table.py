# Framework to add a new data column to existing data file

from math import nan
import os
from posixpath import basename
import numpy as np
import pandas as pd
import sys

# if __name__ == '__main__':

input_file = r'K:\SAG_Redwood\08_Analysis\Final_Diameters_ellipses_v15.csv'
output_file = r'K:\SAG_Redwood\08_Analysis\Final_Diameters_ellipses_dub.csv'
output_dir=''

# global constants
dbh_height = 1.4
index_col = 'TreeId'
name1 = 'ma_diameter'
name2 = 'Tree_Height'
add_name1 = 'dub'

def  get_diameter_under_bark(taper, tree_height, dbh_height):

    B7 =  .7016
    B8 =  .5646
    B9 = -.6188
    Z = (tree_height-dbh_height)/tree_height

    dub = (taper**2)*(B7 + B8*Z + B9*(Z**2))
    return np.sqrt(dub)

def compute_new_data(data):

    data=data.convert_dtypes()

    id_list = data[index_col].values
    new_col = np.zeros(id_list.size)
    vub = []
    for tree_id in np.unique(id_list):
        tree = data[data[index_col] == tree_id]
        # tree.drop(col, axis=1, inplace=True)
        value1 = np.asarray(tree[name1], dtype=float)
        value2 = np.asarray(tree[name2], dtype=float)
        # value2=value2[0] # get the first entry for repeating values

        dub = get_diameter_under_bark(value1, value2[0], dbh_height)
        new_col[data[data[index_col] == tree_id].index.values.astype(int)] = dub
    
        # vub.append(volume_from_taper(dub, value2[0], dbh_height))

    # for r,h in enumerate(report_data['Height']):
    #         report_data.at[r,'Height'] += field_data.at[r,'Prev TotalHt']
    #         report_data.at[r,'Review'] = 1

    return new_col, vub

def volume_from_taper(taper, tree_height, dbh_height) : 
    ### calculates volume from taper
    # arguments:
    # taper - 2D array, 1st column is bin heights, 2nd column is diameter at that height
    # tree_height - the height of the tree - used to calculate a cone volume from the top of taper to the top of tree
    # dbh_height - desired height of DBH measurement
    #
    # Returns:
    # volume_1 - sum of the fulstrum volume along the taper + cone volume at the top
    #  volume 2 - a cylinder volume under DBH_height + cone volume above dbh_height
    # dbh_bin  - the height of the DBH measurement
            
    # # Sum of the cylinders and cone at the top.
    bin_heights = taper[1:,0] - taper[:-1,0]
    radius_1 = taper[:-1,1] / 2
    radius_2 = taper[1:,1] / 2
    volume_1 = np.sum((1 / 3) * np.pi * bin_heights * (radius_1**2 + radius_2**2 + radius_1*radius_2))
    # add a cone to the top
    volume_1 += (tree_height - taper[-1,0])*(1/3)*np.pi*((taper[-1,1]/2)**2)
    # add a cylinder at the bottom down to 0.3m above the ground
    base_volume = (np.pi*(taper[0,1]/2)**2)*(taper[0,0]-0.3)
    volume_1 += base_volume

    # Volume 2 - a simple vertical cone from DBH to treetop + volume of a cylinder from DBH to ground.
    if dbh_height in taper[:,0]:
        dbh_bin = dbh_height
    else :
        closest_bin = np.argmin(np.abs(taper[:,0]-dbh_height))
        dbh_bin = taper[closest_bin,0]
    
    DBH = float(taper[taper[:,0]==dbh_bin,1])
    volume_2 = np.pi * ((DBH/2) ** 2) * (((tree_height - dbh_bin)/3) + dbh_bin)
    
    return volume_1, volume_2, dbh_bin

############# main starts here ##################

if os.path.isfile(input_file):
    data = pd.read_csv(input_file)
else: 
    print(f"Cannot find file {input_file}. Exiting...")
    sys.exit()

# if not(os.path.isdir(output_dir)):
#     os.mkdir(output_dir)

data.columns = data.columns.str.strip() # Remove whitespaces and tabs in colummn names
print(data.columns)

# Index.dropna()
data.dropna(axis=0, subset=[name1, name2], inplace=True)
data.reset_index(drop=True, inplace=True)

# extract info from input 
new_data = data[[index_col, name1, name2]].convert_dtypes()
new_col, vub = compute_new_data(new_data)
data.insert(loc=len(data.columns), column=add_name1, value=new_col)




data.to_csv(output_file, index=False, sep=',')

# data.to_csv(os.path.join(output_dir, outfile), index=False, sep=',') # save it into a new file; filename starting with the PlotID 
# print(f'{output_dir}\{outfile} saved')

#######################

