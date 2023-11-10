# splits the VR table of multiple plots into separate csv files for each plot

import os
from posixpath import basename
import numpy as np
import pandas as pd
import sys

# if __name__ == '__main__':
project = r'K:\DPI' 
filepath = project + r'\VR_processed\PNB168normalBackback.table.data.csv'
name_ext = r'QC_data'
output_dir = r'K:\DPI\FSCT_work'

def vr2fsct_tree_data(vr_data):

    # vr_data = pd.read_csv(filepath)
    # plotID = os.path.basename(filepath).split('_')[0]

    plotID = vr_data['PlotID']
    plotID=plotID.iloc[0].strip()
    print(f'Plot {plotID}')

    report_data = pd.DataFrame(
        columns=['PlotId','TreeId','x_tree_BH','y_tree_BH','z_tree_BH','Distance','Bearing','DBH','CCI_at_BH','Height','Review'])

#     # tree_data.insert(loc=len(tree_data.columns), column='TreeNum', value=0)            
    vr_data=vr_data.convert_dtypes()
    heights = np.array(vr_data['Position'])
    report_data['Height'] = heights[1::2]     # tree height is in the 3rd row of each tree- big assumption
                                              # TODO use Structural and F columns to filter the rows storing the height      
    report_data['z_tree_BH'] = heights[0::2] 
    vr_data.drop(vr_data.index[1::2], axis=0, inplace=True)     # delete the rows with Height 

    report_data['PlotId'] = plotID
    # report_data['TreeID'] = np.array(vr_data['TreeNum'])
    report_data['x_tree_BH'] = np.array(vr_data['X'])
    report_data['y_tree_BH'] = np.array(vr_data['Y'])
    report_data['Distance'] = np.array(vr_data['Distance'])
    report_data['Bearing'] = np.array(vr_data['Bearing'])
    report_data['DBH'] = np.array(vr_data['Diameter'])
    report_data['CCI_at_BH'] = 100
    report_data['Review'] = 1    

    report_data = report_data.sort_values(by='Bearing')
    report_data['TreeId'] = np.arange(len(report_data))+1

    # tree_data = tree_data.convert_dtypes()    

    return report_data



if os.path.isfile(filepath):
    vr_data = pd.read_csv(filepath)
else: 
    print(f"Cannot find file {filepath}. Exiting...")
    sys.exit()

if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)
# vr_data = vr_data.drop(['StemNum','Level','BrMo','Sw','F','Resin','Alive','Internode','sStructural'], axis=1)
vr_data.drop(vr_data.columns[6:13], axis=1, inplace=True)
vr_data.drop(vr_data.columns[[2,3]], axis=1, inplace=True)
vr_data.drop(vr_data.index[0::3], axis=0, inplace=True)

vr_data.columns = vr_data.columns.str.strip() # Remove whitespaces and tabs in colummn names
print(vr_data.columns)
plotID = np.asarray(vr_data['PlotID'])                      # get the plot ID column
treeID = np.array(vr_data['TreeNum'])           # get tree number column
start_indices = np.where(treeID==1)[0]    # find the first row for each plot - treeNum==1
start_indices = np.asarray(start_indices[0::2])
for plot_start in start_indices.flatten():          # loop through all plots in the table starting at treeNum==1
    plot_name = plotID[plot_start]                # get the plot ID
    plot_data = vr_data[plotID == plot_name]   # get all data for this plot
    plot_name = plot_name.strip()
    outfile = f'{plot_name}_{name_ext}.csv'

    report_data = vr2fsct_tree_data(plot_data)
    report_data.to_csv(os.path.join(output_dir, outfile), index=False, sep=',') # save it into a new file; filename starting with the PlotID 

    outpath = output_dir + f'\{plot_name}_FT_output'
    if os.path.isdir(outpath):
        print(f'Saving {outpath}\{outfile}')
        report_data.to_csv(os.path.join(outpath, outfile), index=False, sep=',') # save it into a new file; filename starting with the PlotID 
        print('Done.')
    else:
        print(f'Cannot find {outpath} for plot {plot_name}')
#######################

