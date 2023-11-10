import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import math
import pandas as pd
from scipy import stats, spatial
import time
import warnings
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from scipy.spatial.distance import euclidean
import os
from sklearn.neighbors import NearestNeighbors
from tools import load_file, save_file, get_heights_above_DTM, subsample_point_cloud
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PostProcessing:
    def __init__(self, parameters):
        self.post_processing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        
        filename = self.filename.split('/')[-1]
        filename = filename.split('_')[0]
        filename = filename.replace('.laz','')
        self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + filename + '_FT_output/'
        self.filename = filename + '.laz'

        self.noise_class_label = parameters['noise_class']
        self.terrain_class_label = parameters['terrain_class']
        self.vegetation_class_label = parameters['vegetation_class']
        self.cwd_class_label = parameters['cwd_class']
        self.stem_class_label = parameters['stem_class']

        print("Loading segmented point cloud...")
        load_headers_of_interest=self.parameters['headers'] + ['label']
        
        self.point_cloud, self.headers_of_interest = load_file(self.output_dir+'segmented_raw.laz', headers_of_interest=load_headers_of_interest)
        
        self.point_cloud = np.hstack((self.point_cloud, np.zeros((self.point_cloud.shape[0], 1))))  # Add height above DTM column
        self.headers_of_interest.append('height_above_dtm')  # Add height_above_DTM to the headers.
        self.label_index = self.headers_of_interest.index('label')
        self.point_cloud[:, self.label_index] = self.point_cloud[:, self.label_index] + 1  # index offset since noise_class was removed from inference. Check parameters for reference
        self.plot_summary = pd.read_csv(self.output_dir + 'plot_summary.csv', index_col=None)
        self.plot_radius = self.parameters['plot_radius']

    def make_DTM(self, crop_dtm=False):
        print("Making DTM...")
        full_point_cloud_kdtree = spatial.cKDTree(self.point_cloud[:, :2])
        terrain_kdtree = spatial.cKDTree(self.terrain_points[:, :2])
        xmin = np.floor(np.min(self.terrain_points[:, 0])) - 3
        ymin = np.floor(np.min(self.terrain_points[:, 1])) - 3
        xmax = np.ceil(np.max(self.terrain_points[:, 0])) + 3
        ymax = np.ceil(np.max(self.terrain_points[:, 1])) + 3
        x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters['grid_resolution'])) + 1)
        y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters['grid_resolution'])) + 1)
        grid_points = np.zeros((0, 3))

        for x in x_points:
            for y in y_points:
                radius = self.parameters['grid_resolution']*3       # hard-coded value
                indices = terrain_kdtree.query_ball_point([x, y], r=radius)
                
                while len(indices) <= 100 and radius <= self.parameters['grid_resolution'] * 5:      # hard-coded value x 2
                    radius += self.parameters['grid_resolution']
                    indices = terrain_kdtree.query_ball_point([x, y], r=radius)
               
                if len(indices) >= 100:
                    z = np.percentile(self.terrain_points[indices, 2], 20)      # hard-coded value
                    grid_points = np.vstack((grid_points, np.array([[x, y, z]])))
                else:
                    indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)
                    while len(indices) <= 100:
                        radius += self.parameters['grid_resolution']
                        indices = full_point_cloud_kdtree.query_ball_point([x, y], r=radius)

                    z = np.percentile(self.point_cloud[indices, 2], 2.5)         # hard-coded value  1 instead of 2.5 ???
                    grid_points = np.vstack((grid_points, np.array([[x, y, z]])))

        if self.plot_radius > 0:
            plot_centre = [[float(self.plot_summary['Plot Centre X'].item()), float(self.plot_summary['Plot Centre Y'].item())]]
            crop_radius = self.plot_radius + self.parameters['plot_radius_buffer']
            grid_points = grid_points[np.linalg.norm(grid_points[:, :2] - plot_centre, axis=1) <= crop_radius]

        elif crop_dtm:
            distances, _ = full_point_cloud_kdtree.query(grid_points[:, :2], k=[1])
            distances = np.squeeze(distances)
            grid_points = grid_points[distances <= self.parameters['grid_resolution']]
        print('    DTM Done')
        return grid_points

    def process_point_cloud(self):
        xyz_offsets = [self.plot_summary['Offset X'][0], self.plot_summary['Offset Y'][0], self.plot_summary['Offset Z'][0]]

        self.terrain_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.terrain_class_label]  # 2 is now the class label as we added the height above DTM column.
        
        # terrain points are used in making the DTM !!! 
        self.DTM = self.make_DTM(crop_dtm=False)
        save_file(self.output_dir + 'DTM.laz', self.DTM, offsets=xyz_offsets)

        if self.plot_radius > 0 :
            self.plot_area = np.around(np.pi*(self.plot_radius**2)/10000, 3)  # Aglika - Calculate area from plot radius        
        else:
            self.convexhull = spatial.ConvexHull(self.DTM[:, :2])
            self.plot_area = self.convexhull.volume/10000  # area is volume in 2d.
        print("Plot ground area is approximately", self.plot_area, 'ha')

        above_and_below_DTM_trim_dist = 0.2

        self.point_cloud[:,-1] = get_heights_above_DTM(self.point_cloud[:,:3], self.DTM) 
         
        # Remove underground noise
        self.point_cloud = self.point_cloud[self.point_cloud[:,-1] > -above_and_below_DTM_trim_dist]
        
        # Change Stem, Vegetation and CWD points at ground level to Terrain points
        terrain_mask = (np.abs(self.point_cloud[:,-1]) < above_and_below_DTM_trim_dist)
        self.point_cloud[terrain_mask, self.label_index] = self.terrain_class_label  # change the label
        self.terrain_points = self.point_cloud[terrain_mask]                

        self.stem_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.stem_class_label] 

        self.vegetation_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.vegetation_class_label]
     
        # subsample vegetation to 1 point per 2cm cube TODO - subsampling is faulty
        # num_points = self.vegetation_points.shape[0]
        # self.vegetation_points = subsample_point_cloud(self.vegetation_points, self.parameters['subsampling_min_spacing'], self.parameters['num_procs'])
        # print(f'Subsampling deleted {num_points - self.vegetation_points.shape[0]} points')

        # save_file(self.output_dir + 'vegetation_points.laz', self.vegetation_points, headers_of_interest=self.headers_of_interest, silent=False, offsets=xyz_offsets)

        self.cwd_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.cwd_class_label]  
        
        save_file(self.output_dir + 'terrain_points.laz', self.terrain_points, self.headers_of_interest, silent=False, offsets=xyz_offsets)
        if not self.parameters['minimise_output_size_mode']:
            save_file(self.output_dir + 'stem_points.laz', self.stem_points, headers_of_interest=self.headers_of_interest, silent=False, offsets=xyz_offsets)
            save_file(self.output_dir + 'cwd_points.laz', self.cwd_points, headers_of_interest=self.headers_of_interest, silent=False, offsets=xyz_offsets)

        if xyz_offsets[0] > 10000 : filename = "C2_E_N"
        else : filename ="C2_0_0"
        save_file(f'{self.output_dir}{filename}.laz', self.point_cloud, headers_of_interest=self.headers_of_interest, offsets=xyz_offsets)
        
        # os.remove(self.output_dir + 'segmented_raw.laz')

        self.post_processing_time_end = time.time()
        self.post_processing_time = self.post_processing_time_end - self.post_processing_time_start
        print(f"Post-processing took {np.around(self.post_processing_time/60,2)} min")
        self.plot_summary['Post processing time (s)'] = self.post_processing_time
        self.plot_summary['Num Terrain Points'] = self.terrain_points.shape[0]
        self.plot_summary['Num Vegetation Points'] = self.vegetation_points.shape[0]
        self.plot_summary['Num CWD Points'] = self.cwd_points.shape[0]
        self.plot_summary['Num Stem Points'] = self.stem_points.shape[0]
        self.plot_summary['Plot Area'] = self.plot_area
        self.plot_summary['Post processing time (s)'] = self.post_processing_time
        self.plot_summary.to_csv(self.output_dir + 'plot_summary.csv', index=False)
        print("Post processing done.")

