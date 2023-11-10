# splits the VR table of multiple plots into separate csv files for each plot

from math import nan
import os
from posixpath import basename
import numpy as np
import pandas as pd
import sys

# if __name__ == '__main__':
project = r'K:\SAG_Redwood' 
filepath = project + r'\Redwood_FR92_5_PlotExport_Updated.csv'
name_ext = r'Field_data'
output_dir = r'K:\SAG_Redwood\FSCT_work'

def FSCT_data_format(field_data):

    # field_data = pd.read_csv(filepath)
    # plotID = os.path.basename(filepath).split('_')[0]

    plotID = field_data['Plot_Name']
    plotID=str(plotID.iloc[0].strip())
    print(f'Plot {plotID}')

    report_data = pd.DataFrame(
        columns=['PlotId','TreeId','x_tree_BH','y_tree_BH','z_tree_BH','Distance','Bearing','DBH','CCI_at_BH','Height','Review'])

    # field_data=field_data.convert_dtypes()
    # report_data['TreeID'] = np.array(field_data['TreeNum'])
    report_data['Distance'] = np.array(field_data['Distance'])
    report_data['Bearing'] = np.array(field_data['Bearing'])
    report_data['DBH'] = np.array(field_data['New_DBH'])*10
    report_data['CCI_at_BH'] = 100
    report_data['Review'] = 0   
    report_data['z_tree_BH'] = 1.4 
    report_data['PlotId'] = str(plotID)

    report_data['Height'] = field_data['New_Height'].fillna(0)  # replace NA values with zero
    field_data['Prev TotalHt'] = field_data['Prev TotalHt'].fillna(0)  # replace NA values with zero
    for r,h in enumerate(report_data['Height']):
        if h==0 :
            report_data.at[r,'Height'] += field_data.at[r,'Prev TotalHt']
            report_data.at[r,'Review'] = 1

    # bearing = np.round(np.degrees(np.arctan2(DBH_x, DBH_y))) % 360
    # tree_data['Bearing'] = np.array(bearing.flatten())
    xc = np.array(field_data['Distance'])*np.sin(np.radians(np.array(field_data['Bearing'])))
    yc = np.array(field_data['Distance'])*np.cos(np.radians(np.array(field_data['Bearing'])))
    report_data['x_tree_BH'] = np.round(xc,3)
    report_data['y_tree_BH'] = np.round(yc,3)
    
    report_data = report_data.sort_values(by='Bearing')
    report_data['TreeId'] = np.arange(len(report_data))+1
    return report_data

############# main starts here ##################

if os.path.isfile(filepath):
    field_data = pd.read_csv(filepath)
else: 
    print(f"Cannot find file {filepath}. Exiting...")
    sys.exit()

if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)

field_data.columns = field_data.columns.str.strip() # Remove whitespaces and tabs in colummn names
print(field_data.columns)

report_data = FSCT_data_format(field_data)
plot_name = report_data['PlotId'][0]
plot_name = 'Redwood'
outfile = f'{plot_name}_{name_ext}.csv'
report_data.to_csv(os.path.join(output_dir, outfile), index=False, sep=',') # save it into a new file; filename starting with the PlotID 
print(f'{output_dir}\{outfile} saved')
outpath = output_dir + f'\{plot_name}_FT_output'
if os.path.isdir(outpath):
    print(f'Saving {outpath}\{outfile}')
    report_data.to_csv(os.path.join(outpath, outfile), index=False, sep=',') # save it into a new file; filename starting with the PlotID 
    print('Done.')
else:
    print(f'Cannot find {outpath} for plot {plot_name}')
#######################

