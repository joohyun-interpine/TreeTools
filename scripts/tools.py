from typing import BinaryIO
from sklearn.neighbors import NearestNeighbors
import numpy as np
import laspy
# import dpctl
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, get_context
import pandas as pd
import os
import shutil
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from copy import deepcopy
import hdbscan
from multiprocessing import get_context
from scipy import spatial
from matplotlib import pyplot as plt
from numpy import pi, signbit


def vectors_inner_angle(v1, v2):
    ### returns values in the range [0, 2*pi] 
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)
    
    u1 = v1 / np.atleast_2d(np.linalg.norm(v1, axis=1)).T
    u2 = v2 / np.atleast_2d(np.linalg.norm(v2, axis=1)).T

    y = u1 - u2
    x = u1 + u2

    a0 = 2 * np.arctan(np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1))
    # if (not signbit(a0)) or signbit(pi - a0):
    #     return a0
    # elif signbit(a0): 
    #     return 0.0
    # else: 
    #     return pi
    return np.degrees(a0)

def get_output_path(filename):
    filepath = filename.replace('\\', '/')
    plotID=os.path.basename(filepath)
    plotID = plotID.split('_')[0]
    plotID = plotID.split('.')[0]
    workdir = os.path.dirname(filepath) +'/' + plotID + '_FT_output/'
    return workdir, plotID

def get_fsct_path(location_in_fsct=""):
    current_working_dir = os.getcwd()
    output_path = current_working_dir[: current_working_dir.index("TreeTools")+9]
    if len(location_in_fsct) > 0:
        output_path = os.path.join(output_path, location_in_fsct)
    return output_path.replace("\\", "/")

def make_folder_structure(path, filename):
    plotID = filename.split('_')[0]
    plotID = plotID.split('.')[0]
    output_dir = path + plotID +'_FT_output/'
    working_dir = path + plotID +'_FT_output/working_directory/'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    else:
        shutil.rmtree(working_dir, ignore_errors=True)
        os.makedirs(working_dir)

    return output_dir, working_dir


def subsample(args):
    X, min_spacing = args
    neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(X[:, :3])
    distances, indices = neighbours.kneighbors(X[:, :3])
    X_keep = X[distances[:, 1] >= min_spacing]
    i1 = [distances[:, 1] < min_spacing][0]
    i2 = [X[indices[:, 0], 2] < X[indices[:, 1], 2]][0]
    X_check = X[np.logical_and(i1, i2)]

    while np.shape(X_check)[0] > 1:
        neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(X_check[:, :3])
        distances, indices = neighbours.kneighbors(X_check[:, :3])
        X_keep = np.vstack((X_keep, X_check[distances[:, 1] >= min_spacing, :]))
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [X_check[indices[:, 0], 2] < X_check[indices[:, 1], 2]][0]
        X_check = X_check[np.logical_and(i1, i2)]
    return X_keep


def subsample_point_cloud(pointcloud, min_spacing, num_procs=1):
    """
    Args:
        pointcloud: The input point cloud.
        min_spacing: The minimum allowable distance between two points in the point cloud.
        num_procs: Number of threads to use when subsampling.

    Returns:
        pointcloud: The subsampled point cloud.
    """
    print("Subsampling...")
    print("Original number of points:", pointcloud.shape[0])

    if num_procs > 1:
        num_slices = num_procs
        Xmin = np.min(pointcloud[:, 0])
        Xmax = np.max(pointcloud[:, 0])
        Xrange = Xmax - Xmin
        slice_list = []
        kdtree = spatial.cKDTree(np.atleast_2d(pointcloud[:, 0]).T, leafsize=10000)
        for i in range(num_slices):
            min_bound = Xmin + i*(Xrange/num_slices)
            results = kdtree.query_ball_point(np.array([min_bound]), r=Xrange/num_slices)
            # mask = np.logical_and(pointcloud[:, 0] >= min_bound, pointcloud[:, 0] < max_bound)
            pc_slice = pointcloud[results]
            print("Slice size:", pc_slice.shape[0], '    Slice number:', i+1, '/', num_slices)
            slice_list.append([pc_slice, min_spacing])

        pointcloud = np.zeros((0, pointcloud.shape[1]))
        with get_context("spawn").Pool(processes=num_procs) as pool:
            for i in pool.imap_unordered(subsample, slice_list):
                pointcloud = np.vstack((pointcloud, i))

    else:
        pointcloud = subsample([pointcloud, min_spacing])

    print("Number of points after subsampling:", pointcloud.shape[0])
    return pointcloud


def load_file(filename, plot_centre=None, plot_radius=0, plot_radius_buffer=0, silent=False, headers_of_interest=None, return_num_points=False):
    if headers_of_interest is None:
        headers_of_interest = []
    if not silent:
        print('Loading file...', filename)
    file_extension = filename[-4:]
    coord_headers = ['x', 'y', 'z']
    output_headers = []

    if file_extension == '.las' or file_extension == '.laz':
        inFile = laspy.read(filename)
        header_names = list(inFile.point_format.dimension_names)
        # print(header_names)
        pointcloud = np.vstack((inFile.x, inFile.y, inFile.z))      
        #pointcloud = np.vstack((inFile.X, inFile.Y, inFile.Z))     # Aglika - TODO Change to integer computations for coordinates 
        if len(headers_of_interest) != 0:
            headers_of_interest = headers_of_interest[3:]
            for header in headers_of_interest:
                if header in header_names:
                    pointcloud = np.vstack((pointcloud, getattr(inFile, header)))
                    output_headers.append(header)

        pointcloud = pointcloud.transpose()

    elif file_extension == '.csv':
        pointcloud = np.array(pd.read_csv(filename, header=None, index_col=None, delim_whitespace=True))

    original_num_points = pointcloud.shape[0]

    if plot_centre is None:
        mins = np.min(pointcloud[:, :2], axis=0)
        maxes = np.max(pointcloud[:, :2], axis=0)
        plot_centre = (maxes+mins)/2

    if plot_radius > 0:
        distances = np.linalg.norm(pointcloud[:, :2] - plot_centre, axis=1)
        keep_points = distances < plot_radius + plot_radius_buffer
        pointcloud = pointcloud[keep_points]
    if return_num_points:
        return pointcloud, coord_headers + output_headers, original_num_points
    else:
        return pointcloud, coord_headers + output_headers

# Saving point cloud into file writing all non-zero attributes from headers_of_interest
def save_file(filename, pointcloud, headers_of_interest=None, silent=False, offsets=[0,0,0]):
    if headers_of_interest is None:
        headers_of_interest = []

    if pointcloud.shape[0] == 0:
        print(filename, 'is empty...')
        return 1
    else:
        if not silent:
            print('Saving file:', filename)
        if filename[-4:] == '.laz':
            las = laspy.create(file_version="1.4", point_format=7)
            las.header.global_encoding.value = 17       # see LAS specification
            # f = open(filename, "rb+")
            # f.seek(6)
            # f.write(bytes([17, 0, 0, 0]));

            las.header.point_count = pointcloud.shape[0]
            las.header.offsets = offsets
            las.header.scales = [0.001, 0.001, 0.001]   # asssign scales before assigning coordinates
                                                        # ow: coordinates will be scaled with 10^-1 ?!
            las.x = pointcloud[:, 0]
            las.y = pointcloud[:, 1]
            las.z = pointcloud[:, 2]

            skip=3 # the headers of interest must be called in the right order to match the right column in the np array
            headers_of_interest = headers_of_interest[skip:]
            for hdr in headers_of_interest :
                index = headers_of_interest.index(hdr) + skip
                # Check for non-zero and if not move to the next header item
                if not np.any(pointcloud[:, index]): 
                    index+=1
                    continue
                # save atributes according file_version and point_format
                if hdr in ['red', 'green', 'blue']:
                    setattr(las, hdr, pointcloud[:,index])
                elif hdr in ['intensity','return_number','gps_time','classification'] :
                    las[hdr] = pointcloud[:, index]   # this works for dimensions defined in the type of las file
                else :
                    
                    if hdr == 'Ring' :    
                        las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1', description="Lidar Ring in Velodyne"))             
                        las.Ring = pointcloud[:,index]                        
                    elif hdr == 'Range' :
                        las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4', description="Range from Lidar to Point"))             
                        las.Range = pointcloud[:,index]
                    else :
                        if hdr in ['label', 'CCI']:
                            dimtype='u1'
                            # las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))
                        elif hdr == 'tree_id' :
                            dimtype= 'i2'
                        else : # height_above_dtm, CCI, segment_angle, Vx, Vy, Vz, tree_id
                            dimtype = 'f4'
                            
                        las.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type=dimtype))
                        setattr(las, hdr, pointcloud[:,index])                                    
                
                index +=1
                if index == pointcloud.shape[1] : break 
           
            las.write(filename)
            
            if not silent:
                print("Saved.")

        elif filename[-4:] == '.csv':
            pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
            print("Saved to:", filename)
    
    return 0


def get_heights_above_DTM(points, DTM):
    # *** points is at least a 4-column array of type x,y,z,...,dtm_place_holder
    # print('Getting heights above DTM...')
    
    # grid = griddata((DTM[:, 0], DTM[:, 1]), DTM[:, 2], points[:, 0:2], method='linear')
    grid = griddata((DTM[:, 0], DTM[:, 1]), DTM[:, 2], points[:,0:2], method='nearest')
    return points[:,2] - grid

# def cluster_dbscan(points, eps, min_samples=3, n_jobs=1):
def cluster_dbscan(points, eps, min_samples=2, n_jobs=1):
    # with config_context(target_offload="gpu:0"):
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='kd_tree', n_jobs=n_jobs).fit(points[:, :3])
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(points[:,0], points[:,1], points[:,2], c=db.labels_)
    
    # return np.hstack((points, np.atleast_2d(db.labels_).T))
    return np.atleast_2d(db.labels_).T


def cluster_hdbscan(points, min_cluster_size, min_samples, eps):
    # with config_context(target_offload="gpu:0"):
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=float(eps)).fit_predict(points[:, :3])
    # cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=float(eps)).fit_predict(points[:, :3])
    return np.hstack((points, np.atleast_2d(cluster_labels).T))


def low_resolution_hack_mode(point_cloud, num_iterations, min_spacing, num_procs):
    print('Using low resolution point cloud hack mode...')
    print('Original point cloud shape:', point_cloud.shape)
    point_cloud_original = deepcopy(point_cloud)
    for i in range(num_iterations):
        duplicated = deepcopy(point_cloud_original)

        duplicated[:, :3] = duplicated[:, :3] + np.hstack(
                (np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1))))
        point_cloud = np.vstack((point_cloud, duplicated))
        point_cloud = subsample_point_cloud(point_cloud, min_spacing, num_procs)
    print('Hacked point cloud shape:', point_cloud.shape)
    return point_cloud


def get_taper(single_tree_cyls, bin_heights, tree_base_z, stem_points, vegetation_points, 
                MA_margin, cyl_dict, dbh_correction):

    # cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
    #                 tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)
    col_names = ['bin','diameter','bin_x','bin_y','stem_count','vegetation_count','pruned','CCI']
    # taper_record = np.zeros((1,len(col_names)))
    taper_record = []

    # idx = np.argmin(single_tree_cyls[:, cyl_dict['z']])
    # x_base = single_tree_cyls[idx, cyl_dict['x']]
    # y_base = single_tree_cyls[idx, cyl_dict['y']]
    # z_base = single_tree_cyls[idx, cyl_dict['z']]
    # tree_id = single_tree_cyls[idx, cyl_dict['tree_id']]

    top_flag = False
    for bin in bin_heights:
        bin_cylinders = single_tree_cyls[np.logical_and(single_tree_cyls[:, 2] >= tree_base_z + bin - MA_margin,
                                                  single_tree_cyls[:, 2] <= tree_base_z + bin + MA_margin)]        
        if bin_cylinders.shape[0] == 0: continue
        
        bin_cylinders = bin_cylinders[bin_cylinders[:, cyl_dict['radius']] > 0, :] 
        if bin_cylinders.shape[0] > 0:                  
            diameter = np.around(np.mean(bin_cylinders[:, cyl_dict['radius']]) * 2,3)     # Aglika - USING THE mean simulates moving average
                                                                               # max takes care of branches :/  TODO
            CCI = np.around(np.mean(bin_cylinders[:, cyl_dict['CCI']]), 0)
            bin_x = np.around(np.mean(bin_cylinders[:,0]), 3)  # bin centre coordinates 
            bin_y = np.around(np.mean(bin_cylinders[:,1]), 3)  # bin centre coordinates 
            
            stems = stem_points[np.logical_and(stem_points[:,2] >= tree_base_z + bin - .15,
                                                stem_points[:,2] <= tree_base_z + bin + .15)]   # bin stem points
            vegetation = vegetation_points[np.logical_and(vegetation_points[:,2] >= tree_base_z + bin - .15,
                                                vegetation_points[:,2] <= tree_base_z + bin + .15)]    # bin vegetation points
            s_count = stems.shape[0] 
            ns_count = vegetation.shape[0]
            pruned = (s_count > (ns_count/10)) * 1                                      

            # if top_flag : # already at the top of the tree
            #     if (s_count > 100) or (ns_count > 100):  # points from neighbours are coming in
            #         break   # the top is reached, discard this diameter and finish
            # else :
            #     if (s_count < 100) and (ns_count < 100):  # getting closer to the tree top
            #         top_flag = True

            # add DBH correction if bin < 2m
            if bin <= 2:
                diameter += dbh_correction

            # add current diameter to the taper
            taper_record.append(np.array([[bin,diameter,bin_x, bin_y, s_count, ns_count, pruned, CCI]]))

    # # get base XY coordinates from the lowest measured non-zero diameter
    # base_x = bin_x[np.argmin(diameters>0)]
    # base_y = bin_y[np.argmin(diameters>0)]

    return np.asarray(taper_record).squeeze()
    # return np.hstack((np.array([[plot_id, tree_id, x_base, y_base, z_base]]), diameters)) #  return an array of this info TODO add DBH_X and DBH_Y in parent

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

def  get_diameter_under_bark(taper, tree_height, dbh_height):

    B7 =  .7016
    B8 =  .5646
    B9 = -.6188
    Z = (tree_height-dbh_height)/tree_height

    dub = np.array(taper)**2*(B7 + B8*Z + B9*(Z**2))

    return np.sqrt(dub)

def plot_circle_points(filename,x, y, r, points, color_col, inliers=None) :
            
    fig,ax = plt.subplots(figsize=[7,7])
    # ax.set_title(f"TreeID = {output_id}, CCI = {int(CCI)}%, height = {np.around(np.mean(P[:,self.stem_dict['height_above_dtm']]),1)}m")
    ax.set_aspect('equal', adjustable='datalim')
    # ax.scatter(P[:,0], P[:,1], c=np.round(P[:,self.stem_dict['Range']]*100),  marker='.', cmap='Blues')
    ax.scatter(points[:,0], points[:,1], c=points[:,color_col],  marker='.', cmap='Blues')
    if not inliers==None:
        ax.scatter(points[inliers,0], points[inliers,1])

    # ax.scatter(DBH_x, DBH_y,  marker='+', color='k')
    circle_outline = plt.Circle((x,y), r, fill=False, edgecolor='r')
    ax.add_patch(circle_outline)
    # plt.xlim(DBH_x-DBH_taper/2-0.04, DBH_x+DBH_taper/2+0.04)
    # plt.ylim(DBH_y-DBH_taper/2-0.04, DBH_y+DBH_taper/2+0.04)
    # plt.show()
    fig.savefig(f'c:\\Temp\\{filename}.png', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def copy_canopy_image(workdir) :

    plotID = os.path.basename(workdir[:-1])
    plotID = plotID.split('_')[0]
    
    owd = os.getcwd()
    os.chdir(workdir)

    # move canopy image to plot output folder
    canopy_file = f'{plotID}_canopy.png'
    if not os.path.exists(canopy_file) : 
        print('Copying canopy image to plot folder')
        canopy_path = f'..\\Canopy\\'
        if os.path.exists(canopy_path+canopy_file) :
            try:
                shutil.copyfile(canopy_path+canopy_file, canopy_file)
            except Exception as e:
                print(f'Cannot copy {canopy_file}: {str(e)}')
    
    os.chdir(owd)

def clean_output(filepath, delete_workdir=True) :

    output_dir, plotID =  get_output_path(filepath)

    files_to_delete = [ '.html',
                        '_segmented_raw.laz',
                        '_ground_veg.laz',                            
                        '_terrain_points.laz',
                        '_stem_points_clusters.laz',
                        '_working_point_cloud.laz',
                        '_ransac_cyls.laz',
                        '_ransac_cyl_vis.laz',
                        '_stem_points_for_ransac.laz',
                        '_temp_vegetation.laz'
                        ]

    for file in files_to_delete:
        try:
            filename = f'{output_dir}/{plotID}{file}'
            if os.path.isfile(filename):
                os.remove(filename)
                print(filename, ' deleted.')
        # except FileNotFoundError or OSError:
        except OSError as e:
            print(str(e))


    if delete_workdir :
        shutil.rmtree(output_dir + 'working_directory/', ignore_errors=True)


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
    
    except Exception as e:
        print(str(e))

    os.chdir(owd)

def trace_skeleton(tree, base_id, cyl_dict, mark = False) :
    
    child_ids=[base_id]  
    next_branches = np.unique(tree[tree[:,cyl_dict['parent_branch_id']]==base_id,cyl_dict['branch_id']]) # child ids                
    all_childids=np.concatenate((child_ids, next_branches), axis = None)

    next_branches = next_branches[next_branches!=base_id]  # for the tree base parent_id is same as branch_id; remove it 
    base_branch_topz = np.max(tree[tree[:, cyl_dict['branch_id']]==base_id,2])
    
    while np.any(next_branches) : 
        # check if child is located lower than the parent's top
        if np.min(tree[tree[:, cyl_dict['branch_id']]==next_branches[0], 2]) < base_branch_topz :
            next_branches = next_branches[1:]
            continue   
        
        # find the largest radius child
        child_id=next_branches[0]
        child_radius = np.mean(tree[tree[:, cyl_dict['branch_id']]==next_branches[0], cyl_dict['radius']])
        for i in range(len(next_branches)-1) :
            if np.mean(tree[tree[:, cyl_dict['branch_id']]==next_branches[i+1], cyl_dict['radius']]) > child_radius :
                child_id=next_branches[i+1]  # current largest radius child
                child_radius = np.mean(tree[tree[:, cyl_dict['branch_id']]==next_branches[i+1], cyl_dict['radius']])
        
        child_ids.append(child_id) # add to the list of children
        
        # process/mark the child branch
        if mark:
            tree[tree[:, cyl_dict['branch_id']]==child_id, cyl_dict['main_stem']] = 1  # assign main stem

        # next iteration
        next_branches = np.unique(tree[tree[:,cyl_dict['parent_branch_id']]==child_id,cyl_dict['branch_id']])
        next_branches = next_branches[next_branches!=child_id] # just in case - for complex tree structures
        all_childids = np.concatenate((all_childids, next_branches), axis=None)

    return tree, child_ids, all_childids