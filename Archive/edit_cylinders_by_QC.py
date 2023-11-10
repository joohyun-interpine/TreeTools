# Edit cylinders according to QC data table

import os
import numpy as np
import pandas as pd
import glob

# if __name__ == '__main__':
    
input_path = r'K:\FPC_priority\FSCT_work'
name_ext = r'QC_data'
output_dir = r'K:\Taper_After_QC'
plot_radius = 11.28

if not(os.path.isdir(output_dir)):
    os.mkdir(output_dir)
   
os.chdir(output_dir)

filelist = glob.glob(input_path + '/**/*cleaned_cyls.csv', recursive = True)
for plot in filelist :

    plotID = os.path.basename(plot).split('_')[0]
    plot_folder = os.path.dirname(plot)
    # plotID=plotID.iloc[0].strip()
    print(plotID)
    data = pd.read_csv(plot)
    data=data.convert_dtypes()

    data.drop(columns='tree_volume', axis=1, inplace=True)  # don't need the volume
    data.drop(columns='parent_branch_id', axis=1, inplace=True)  # don't need the parent branch
    data.drop(columns='branch_id', axis=1, inplace=True) # don't need the branch id anymore

    # find trees that are outside the plot and delete them
    # x = np.float16(np.asarray(data['x']))
    # y = np.float16(np.asarray(data['y']))
    # distance = np.sqrt(x**2 + y**2)
    # outside_ids = np.unique(data[distance > plot_radius]['tree_id'])
    # if len(outside_ids)>0 :
    #     for id in outside_ids:
    #         # i = data[tree_ids==id].index
    #         #  data.drop(i, inplace=True)
    #         data = data[data['tree_id'] != id]
    # print(f'total umber of trees with cylinders: {len(num_trees)}')

    
    data.loc[data['segment_angle_to_horiz'].isnull(), 'segment_angle_to_horiz']=100 # replace NA values with 0
    cylinders=data.to_numpy()  # convert to numpy
    cylinders = cylinders[cylinders[:,data.columns.get_loc('segment_angle_to_horiz')] > 80, :] # keep segment_angle_to_horizon > 80

    # find the mean X and Y of all cylinders under 3m height
    mask = np.logical_and(cylinders[:,-1] < 3.01,    # height_above_dtm < 3
                            cylinders[:,7] > 0.1)   # CCI > 0.1
                            # cylinders[:,-2] > 80)    # segment_angle_to_horizon > 80
    cylinders = cylinders[mask, :] # get X and Y coordinates of filtered cylinders

    data = pd.DataFrame(cylinders, columns=data.columns) # save the the main stems only

    tree_id_col = data.columns.get_loc('tree_id')
    num_trees = np.unique(cylinders[:, tree_id_col]).shape[0]
    XY = np.zeros((num_trees,2))
    cyl_ids = np.zeros(num_trees) 
    i=0
    # make a list of tree ids and corresponding average XY coordinates
    for id in np.unique(cylinders[:,tree_id_col]): 
        this_tree = cylinders[:, tree_id_col]==id
        if (cylinders[this_tree,:].shape[0] > 2):
            cyl_ids[i]=id       
            XY[i,:] = np.mean(cylinders[this_tree, :2], axis=0) # get X and Y coordinates of filtered cylinders
            i +=1 
    if i==0 :
        print(f'\n skipping plot {plotID} \n')
        continue
    cyl_ids = cyl_ids[:i]
    XY = XY[:i,:]

    # Load QC data
    qc_data = pd.read_csv(plot_folder + f'\{plotID}_QC_data.csv')
    num_trees = np.unique(qc_data['TreeId'])
    print(f'\n Plot {plotID}, total number of QC trees: {len(num_trees)}')

    # search a 1m circle area around each QC tree for a corresponding FSCT tree
    for tree in np.unique(qc_data['TreeId']):
        tree_data = qc_data[qc_data['TreeId']==tree]

        x_qc = np.float16(tree_data['x_tree_BH'])
        y_qc = np.float16(tree_data['y_tree_BH'])
          
        # Find the distances to all cylinders
        dists = np.linalg.norm(np.vstack((x_qc, y_qc)).T - XY, axis=1)
        if np.min(dists) < .2 : # qc location is usually not further away than 1/2 metres
            # qc_id = qc_data.iloc[np.argmin(dists)]['TreeId']
            cid=int(cyl_ids[np.argmin(dists)])  # find the closest cylinder
            print(f'cylinders_id: {cid}, QC_id: {tree}, distance: {np.around(min(dists),2)}')

            # cylinders[cylinders[:,tree_id_col]==cid, tree_id_col] = tree TODO - carefull with new vs old ids
            XY[np.argmin(dists),:] = 100     # mark this cylinder_tree as matched
            qc_data = qc_data[qc_data['TreeId'] != tree]

            if np.all(XY == 100) : # no cylinders left for matching
                break          

    # # Fix diameters by adding the terrain band to the height_above_dtm - works for FPC - might be a bug in FSCT TODO
    # cylinders[:,-1] +=.1

    # TODO fix duplicate ids if you want to save the new cylinders
    # cyl_data = pd.DataFrame(cylinders, columns=data.columns)
    # cyl_data.sort_values('tree_id')
    # cyl_data.to_csv(f'{plotID}_cylinders_after_QC.csv', index=False)

    qc_data=qc_data[['PlotId','TreeId','x_tree_BH','y_tree_BH']]
    outdir = f'{input_path}\{plotID}_FT_output'
    try:
        qc_data.to_csv(f'{outdir}\{plotID}_QC_seeds.csv', index=False)
    except OSError as e:
        print(str(e))
    
    #  Keep the segment with the largest number of cylinders. Hoping this will be the main stem.
    # final_df = pd.DataFrame()
    # tree_counts = data.groupby(["tree_id"])["tree_id"].count()
    # trees = list(tree_counts.index) # get the number of trees in this plot
    # for i in trees:
    #     tree = data[data["tree_id"] == i]  # all cylinders for this tree
    #     main_branch = int(tree[["tree_id","branch_id"]].groupby("branch_id").count().idxmax()) # find the branch id with most members
    #     tree = tree[tree["branch_id"]==main_branch]
    #     tree = tree.sort_values(by=["tree_id","height_above_dtm"]).reset_index()
    #     tree = tree.drop_duplicates(subset=["tree_id","x","y","z"], keep='last').reset_index()
    #     final_df = pd.concat([final_df, tree], ignore_index=True)  
    # data = final_df # now data contains the main stems only
  
    
    


