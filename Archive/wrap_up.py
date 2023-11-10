import numpy as np
import os
import laspy
import pandas as pd
import shutil
import traceback
import sys
import subprocess
from copy import deepcopy
from scipy.spatial import cKDTree
from tools import get_output_path, get_diameter_under_bark

def add_dub_to_table(filepath) :

    workdir, plotID = get_output_path(filepath)
    workdir_ext = workdir + '/' + plotID + '_'                              # filenames starting with plotID
 
    # tree_data
    # dub = get_diameter_under_bark(taper, tree_height, dbh_height)
    # vub = 


def add_plotid_to_name(filepath):
    
    filepath = filepath.replace('\\', '/')
    filename = os.path.basename(filepath)
    if os.path.isdir(filename):
        # workdir = os.path.dirname(filepath)
        workdir = filepath
        plotID = filename.split('_')[0]
    else :
        filename = filename.split('_')[0]
        plotID = filename.split('.')[0]
        workdir = os.path.dirname(filepath) +'/' + plotID + '_FT_output/'

    # dirname = os.path.dirname(os.path.realpath(filename)).replace('\\', '/') + '/' + filename.split('/')[-1][:-4] + '_FT_output/'
    owd = os.getcwd()

    try: 
        # check if the filenames already have the plotID
        if os.path.isdir(workdir) :
            # if os.path.isfile(workdir + '/' + plotID + '_DTM.laz') :
            #     print('Files are already renamed.')
            #     return 0
            # else :
                
                os.chdir(workdir)
                print(os.getcwd())
                print("Renaming output files.")
                for f in os.listdir() :
                    f_name, f_ext = os.path.splitext(f)
                    if (f_name.split('_')[0] != plotID) and os.path.isfile(f):
                    # if (f_ext != ".csv") & (f_ext != ".pdf") :
                    # if (f_name == 'tree_data_report') | (f_name == "Plot_Report"):
                        # f_name = plotID + '_' + f_name
                    
                        new_name = f'{plotID}_{f_name}{f_ext}'
                        try: 
                            os.replace(f, new_name)
                        except PermissionError:
                            print(f'Error: Cannot rename {f}')
                            print("Operation not permitted.")

        else : print('Files are already renamed')
      
        # # move canopy image to plot output folder
        # canopy_file = f'{plotID}_canopy.png'
        # if not os.path.exists(canopy_file) : 
        #     print('Copying canopy image to plot folder')
        #     canopy_path = f'..\\..\\Canopy\\'
        #     if os.path.exists(canopy_path+canopy_file) :
        #         try:
        #             shutil.copyfile(canopy_path+canopy_file, canopy_file)
        #         except Exception as e:
        #             print(f'Cannot copy {canopy_file}: {str(e)}')
    
    except Exception as e:
        print(str(e))

    os.chdir(owd)



def split_by_seed():
    print('TODO')

def merge_las(parameters): 
    ## argument 1 is the current FT_output directory (for now)

    workdir, plotID = get_output_path(parameters['point_cloud_filename'])
    workdir_ext = workdir + '/' + plotID + '_'                              # filenames starting with plotID
    
    if parameters['plot_centre'] == [0,0] :
        radius = parameters['plot_radius']
    else :
        radius = (las2.header.maxs[0] - las2.header.mins[0])/2

    #  Add an empty tree_id attribute to C2_0_0.las 
    #   + remove the buffer 
    las2 = laspy.read(workdir_ext + 'C2_0_0.laz')
    # for dim in las2.point_format.dimensions: print(dim.name)

    mean_gps_time = np.mean(las2.gps_time)         # save this for the purposes of merging
    newlas = laspy.LasData(las2.header)            # Create a new las
    
    # assuming circular plots - remove buffer
    dists_fc = np.sqrt(np.square(las2.x) + np.square(las2.y))
    mask = dists_fc < radius
    # mask = (las2.x < radius) & (las2.x > radius*(-1)) & (las2.y < radius) & (las2.y > radius*(-1))
    newlas.points = las2.points[mask].copy()
    del las2
    newlas.add_extra_dim(laspy.ExtraBytesParams(name='tree_id', type='u1'))     # add an empty tree_id attribute to match the final output
    setattr(newlas, 'tree_id', [0]*len(newlas.points))                          # set the tree id to 0 - same as ground points
    newlas.write(workdir_ext + 'full.laz')
    print('full.las saved')
    del newlas


    # Add classification 8 for the DTM grid + gps time + remove buffer
    # 
    las = laspy.read(workdir_ext + 'DTM.laz')

    dists_fc = np.sqrt(np.square(las.x) + np.square(las.y))
    mask = dists_fc < radius
    las.points = las.points[mask].copy()
    las.classification  = [8]*las.header.point_count                # set classification number to 8
    las.gps_time = [mean_gps_time.astype(float)]*len(las)           # set gps time to the average gps of the segmented point cloud
    num_points = las.header.point_count
    for hdr in ['Ring', 'Range', 'label','height_above_dtm','tree_id']:  
        # print(hdr)
        if hdr == 'Ring' :    
            las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))             
            las.Ring = [0]*num_points
        elif hdr == 'Range' :
            las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))             
            las.Range =[0.]*num_points
        else:
            if hdr in ['height_above_dtm'] :
                las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))       # height is float
            else :
                las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))  # set tree_id, label are unsigned char
            
            setattr(las, hdr, [0]*las.header.point_count)                  # set tree_id, label and height_above_dtm to 0

    las.write(workdir_ext + 'DTM_class.laz')
    print('DTM_class.las saved')
    # del las

    # with laspy.open(workdir_ext + 'full.laz', mode='a') as outlas:
    #     outlas.append_points(las.points)


    # Add Ring and Range attributes to the tree_aware point cloud mark non classified points to 'unclassified'
    # 
    filename = workdir_ext + 'tree_aware_cropped_point_cloud.laz' 
    las = laspy.read(filename)
    las_class = las.classification
    indices = np.where(las_class == 0 ) # change classification of never classified (buffer) to 'unclassified'
    las_class[indices] = 1
    # newlas = laspy.LasData(las.header)   # we want to change the header because we are changing the attributes
    # newlas = laspy.create()
    # newlas.header.offsets = [0,0,0]      # Aglika
    # newlas.header.scales = [0.001, 0.001, 0.001]
    header = deepcopy(las.header)
    header.point_count = 0
    newlas = laspy.LasData(header)  
    newlas = laspy.create(file_version = las.header.version, point_format=7)
    newlas.classification = las_class
    newlas.gps_time = [mean_gps_time.astype(float)]*len(las)
    newlas.x = las.x
    newlas.y = las.y
    newlas.z = las.z

    # for d in newlas.point_format : print(d)

    # Add Ring and Range ...
    num_points = len(las.points)
    for hdr in ['Ring', 'Range', 'label','height_above_dtm','tree_id']:  
        # print(hdr)
        if hdr == 'Ring' :    
            newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))             
            newlas.Ring = [0]*num_points
        elif hdr == 'Range' :
            newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))             
            newlas.Range =[0.]*num_points
        else :
            if hdr in ['height_above_dtm'] :
                newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))
            else:
                newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))
            setattr(newlas, hdr, las[hdr])                  # copy values from original file - label, height_above_dtm and tree_id

    # newlas.user_data = las['tree_id']    
    # newlas = laspy.convert(newlas, point_format_id=7)
    newlas.write(workdir_ext + 'tree_ids.laz')              #
    print('tree_ids.las saved')
    del newlas

    try:
        cwd = os.getcwd()
        os.chdir(workdir)
    except IOError:
        print('IO system Error. Skipping the current file. ')
        return 
    try :
        file1 = workdir_ext + 'full.las '
        file2 = workdir_ext + 'tree_ids.las '
        file3 = workdir_ext + 'DTM_class.las '
        merge_command = 'C:/LAStools/bin/lasmerge.exe -i '+ file1 + file2 + file3 + '-o ' + workdir_ext + 'C2.las -olas'
        print(merge_command)
        if subprocess.call(merge_command)==0 :
            print('Merging done')
        else : print('Merging failed')

    except:
        print('Error')
        return 1
    
    os.remove(workdir_ext + 'full.laz')
    os.remove(workdir_ext + 'tree_ids.laz')
    os.remove(workdir_ext + 'DTM_class.laz')
    
    # command = 'C:/LAStools/bin/lasoptimize64.exe -i merged.laz -odix _opt -olaz'
    # if subprocess.call(command)==0 : print('lasoptimize done')
    # else : print('lasoptimize failed')
   
    os.chdir(cwd)
    return 0

def sort_tree_ids(filepath):

    try: 
        output_dir = filepath.replace('\\', '/')
        plotID=os.path.basename(output_dir)
        plotID = plotID.split('_')[0]
        plotID = plotID.split('.')[0]
        output_dir = os.path.dirname(output_dir) +'/' + plotID + '_FT_output/'
        np
        if os.path.isfile(output_dir + 'tree_data.csv') :
            tree_data = pd.read_csv(output_dir + 'tree_data.csv')
            oldTreeId = np.array(tree_data['TreeId'])
            Bearing = np.array(tree_data['Bearing'])
            TreeId, mask = sort_by_index(oldTreeId, Bearing)
            # TreeId = np.sort(TreeId, kind='stable')
            tree_data['TreeId'] = TreeId
            df = tree_data.sort_values(by = 'Bearing')
            if os.path.isfile(output_dir + 'tree_data_sorted.csv') :
                os.remove(output_dir + 'tree_data_sorted.csv')
            df.to_csv(output_dir + 'tree_data_sorted.csv', index=None, sep=',')     
            os.remove(output_dir + 'tree_data.csv')

            if os.path.isfile(output_dir + 'taper_data_sorted.csv') :
                os.remove(output_dir + 'taper_data_sorted.csv')
            tree_data = pd.read_csv(output_dir + 'taper_data.csv')
            tree_data['TreeId'] = TreeId    #  get the original TreeID from tree_data.csv
            df = tree_data.sort_values(by = 'TreeId')
            df.to_csv(output_dir + 'taper_data_sorted.csv', index=None, sep=',')     
            if os.path.isfile(output_dir + 'taper_data_sorted.csv') :
                os.remove(output_dir + 'taper_data.csv')
        else:
            print(f'Cannot find tree_data.csv for plot {plotID}')
            return
    except OSError as e:
        print(str(e))
        return(1)

def sort_by_index(data, index):
    # sort data according to the index AND replace data values with [1:number_of_original_unique_values]
    # data and index must be 1D arrays of equal length
    
    mask = np.argsort(index, kind='stable')    # sort the index preserving the order equal elements - 'stable'
    sorted = data[mask.tolist()]               # not a real data copy !!

    temp = deepcopy(data)                     # make a copy of original data to use as reference
    for i, ind in enumerate(mask) :           # i defaults to zero
        data[ind] = i+1                      # assign values in range(unique(data))        
    
    return data, mask


