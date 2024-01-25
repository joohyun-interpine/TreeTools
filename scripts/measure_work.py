import math
from msvcrt import SEM_FAILCRITICALERRORS
import sys
from copy import deepcopy
from multiprocessing import get_context
import networkx as nx
# from numba import jit
import numpy as np
import pandas as pd
import os
import shutil
import glob
from scipy import spatial  # TODO Test if sklearn kdtree is faster.
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from skimage.measure import LineModelND, CircleModel, EllipseModel, ransac
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from tools import load_file, save_file, low_resolution_hack_mode, cluster_hdbscan, cluster_dbscan, trace_skeleton, \
                    get_heights_above_DTM, get_taper, volume_from_taper, get_output_path, vectors_inner_angle, get_diameter_under_bark, plot_circle_points
import time
from skspatial.objects import Plane
import scipy
from sklearn.neighbors import BallTree
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


DEBUG = 0  # use only for debugging. Use breakpoints OW it will output hundreds of image files !!!

class MeasureTree:
    # global work_dir
    # work_dir = "C:\\Temp\\"

    def __init__(self, parameters):
        self.measure_time_start = time.time()
        self.parameters = parameters

        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        workdir, self.plotID = get_output_path(parameters['point_cloud_filename'])
        self.output_dir = workdir
        if os.path.isfile(workdir + '/' + self.plotID + '_DTM.laz'):  # filenames starting with plotID
            self.output_dir =  workdir + '/' + self.plotID + '_' 
        self.workdir = workdir
        self.filename = self.plotID + '.laz'

        self.num_procs = parameters['num_procs']
        self.num_neighbours = parameters['num_neighbours']
        self.slice_thickness = parameters['slice_thickness']
        self.slice_increment = parameters['slice_increment']
        self.min_CCI = parameters['minimum_CCI']

        self.plot_summary = pd.read_csv(self.output_dir + 'plot_summary.csv', index_col=False)
        self.plot_centre = np.array([float(self.plot_summary['Plot Centre X'].item()), float(self.plot_summary['Plot Centre Y'].item())])

        # get plot coordinates offset for las files 
        self.offsets =  [self.plot_summary['Offset X'].item(), self.plot_summary['Offset Y'].item(), self.plot_summary['Offset Z'].item()]

        # try :  # clean output files from old runs
        #     fileList = glob.glob(self.output_dir + '*_cyls.*')
        #     # fileList.extend(glob.glob(self.output_dir + '*_cyl_*.laz'))
        #     for file in fileList :
        #         os.remove(file)
        # except OSError:
        #     print('error deleting file %s' % file) 


        self.parameters['plot_radius'] = float(self.plot_summary['Plot Radius'].item())
        self.parameters['plot_radius_buffer'] = float(self.plot_summary['Plot Radius Buffer'])
        self.plot_area = float(self.plot_summary['Plot Area'])

        # Load Point Cloud
        load_headers_of_interest = self.parameters['headers'] + ['label','height_above_dtm']
        
        if self.offsets[0] > 10000 : filename = "C2_E_N"
        else : filename ="C2_0_0"
        
        all_points, headers_of_interest = load_file(self.output_dir + filename + '.laz', headers_of_interest = load_headers_of_interest)
        
        # get the stem points
        self.stem_points = all_points[all_points[:, headers_of_interest.index('label')] == self.parameters['stem_class'],:]
        # self.stem_points, stem_headers_of_interest = load_file(self.output_dir + 'stem_points.laz', headers_of_interest = load_headers_of_interest)

        # add space for the tree id and update the headers
        self.stem_points = np.hstack((np.around(self.stem_points,3), np.zeros((self.stem_points.shape[0], 1))))  
        headers_of_interest.insert(len(headers_of_interest), 'tree_id')
        # self.stem_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        # self.stem_dict = dict(x=0, y=1, z=2, intensity=3, Range=4, label=5, height_above_dtm=6, tree_id=7)
        self.stem_dict = dict(zip(headers_of_interest,range(len(headers_of_interest))))

        self.vegetation_points= all_points[all_points[:, headers_of_interest.index('label')] == self.parameters['vegetation_class'],:]
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 1))))
        self.veg_dict = dict(zip(headers_of_interest,range(len(headers_of_interest))))
        
        self.terrain_points = all_points[all_points[:, headers_of_interest.index('label')] == self.parameters['terrain_class'],:]
        self.terrain_points = np.hstack((self.terrain_points, np.zeros((self.terrain_points.shape[0], 1))))

        max_stem_height = np.max(self.stem_points[:, self.veg_dict['height_above_dtm']]) - np.min(self.stem_points[:, self.veg_dict['height_above_dtm']]) 
        if self.parameters['bark_sensor_return'] == 'Normal':
            self.cluster_size_threshold = parameters['cluster_size_threshold'][1]
        elif self.parameters['bark_sensor_return'] == 'Sparse':
            self.cluster_size_threshold = parameters['cluster_size_threshold'][2]
        elif self.parameters['bark_sensor_return'] == 'Dense' :
            self.cluster_size_threshold = parameters['cluster_size_threshold'][0]

        # Remove understory vegetation and stems from further processing
        ground_veg_mask = self.vegetation_points[:, self.veg_dict['height_above_dtm']] <= self.parameters['ground_veg_cutoff_height']
        self.ground_veg = self.vegetation_points[ground_veg_mask]
        # change the label for unused points
        self.ground_veg[:,self.veg_dict['label']] = 6   # change label to unused points
        # remove from vegetation points array
        self.vegetation_points = self.vegetation_points[np.logical_not(ground_veg_mask)]
        # get near ground stem points
        ground_veg_mask = self.stem_points[:, self.stem_dict['height_above_dtm']] <= self.parameters['ground_stem_cutoff_height']
        # move them to ground vegetation but do not change the label
        self.ground_veg = np.vstack((self.ground_veg, self.stem_points[ground_veg_mask]))
        self.stem_points = self.stem_points[np.logical_not(ground_veg_mask)]
        if not self.parameters['minimise_output_size_mode'] :
            # save ground veg to file        
            save_file(self.output_dir + 'ground_veg.laz', self.ground_veg, headers_of_interest=list(self.veg_dict), offsets=self.offsets)

        # load the CWD points and do a sanity check
        if os.path.isfile(self.output_dir + 'cwd_points.laz'):
            self.cwd_points, headers_of_interest = load_file(self.output_dir + 'cwd_points.laz', headers_of_interest=load_headers_of_interest)    
            self.cwd_points = np.hstack((self.cwd_points, np.zeros((self.cwd_points.shape[0], 1))))
        else:
            self.cwd_points = np.zeros((0, self.stem_points.shape[1]))   # just a hack to fix poor code design in original 
        
        # sanity check
        if all_points[all_points[:, headers_of_interest.index('label')] == self.parameters['cwd_class']].shape[0] != self.cwd_points.shape[0]:
            print('Something is wrong with class label assignment')

        del all_points


        # create duplicate stem points for very low resolution point clouds TODO better to handle this with parameter tuning
        if self.parameters['low_resolution_point_cloud_hack_mode']:
            self.stem_points = low_resolution_hack_mode(self.stem_points,
                                                        self.parameters['low_resolution_point_cloud_hack_mode'],
                                                        self.parameters['subsampling_min_spacing'],
                                                        self.parameters['num_procs'])
            save_file(self.output_dir + self.filename[:-4] + '_stem_points_hack_mode_cloud.laz', self.stem_points, offsets=self.offsets)

        # load DTM points
        self.DTM, headers_of_interest = load_file(self.output_dir + 'DTM.laz')
        # print('DTM headers:', headers_of_interest)

        # Text Point Ploud variables
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'm', 'space', '_', '-', 'semiC',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', '_M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.character_viz = []
        for i in self.characters:
            self.character_viz.append(np.genfromtxt('../tools/numbers/' + i + '.csv', delimiter=','))

        # define column order for cylinder data
        self.cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                             tree_id=10, height_above_dtm=11, main_stem=12)
        # define column order for individual tree data
        self.tree_data_dict = dict(PlotId=0, TreeNumber=1, TreeLocation_X=2, TreeLocation_Y=3, TreeLocation_Z=4, Distance=5, Bearing=6, Height=7, 
                                    DBH=8, DBH_height=9, DBH_taper=10, DBH_bin=11, CCI=12, Volume=13, 
                                    Crown_mean_x=14, Crown_mean_y=15, Crown_top_x=16, Crown_top_y=17, Crown_top_z=18,
                                    Dub=19, Vub=20, Crown_Height=21)

        
   
    def vegetation_coverage(self):
        # Calculate Vegetation Coverage 
        print('Calculating vegetation coverage...')

        if self.cwd_points.size > 0 :
            cwd_kdtree = spatial.cKDTree(self.cwd_points[:, :2], leafsize=10000)
        else :
            cwd_kdtree=[]
       
        veg_kdtree = spatial.cKDTree(self.vegetation_points[:, :2], leafsize=10000)
        self.ground_veg_kdtree = spatial.cKDTree(self.ground_veg[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0]))
        ymin = np.floor(np.min(self.terrain_points[:, 1]))
        xmax = np.ceil(np.max(self.terrain_points[:, 0]))
        ymax = np.ceil(np.max(self.terrain_points[:, 1]))
        x_points = np.linspace(xmin, xmax,
                               int(np.ceil((xmax - xmin) / self.parameters['vegetation_coverage_resolution'])) + 1)
        y_points = np.linspace(ymin, ymax,
                               int(np.ceil((ymax - ymin) / self.parameters['vegetation_coverage_resolution'])) + 1)

        convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.ground_area = 0  # unitless. Not in m2.
        self.canopy_area = 0  # unitless. Not in m2.
        self.ground_veg_area = 0  # unitless. Not in m2.
        self.cwd_area = 0  # unitless. Not in m2.        
        for x in x_points:
            for y in y_points:
                if self.inside_conv_hull(np.array([x, y]), convexhull):
                    indices = veg_kdtree.query_ball_point([x, y],
                                                          r=self.parameters['vegetation_coverage_resolution'], p=10)
                    ground_veg_indices = self.ground_veg_kdtree.query_ball_point([x, y], 
                                                                                r=self.parameters['vegetation_coverage_resolution'], p=10)
                    if cwd_kdtree:
                        cwd_indices = cwd_kdtree.query_ball_point([x, y], r=self.parameters['vegetation_coverage_resolution'], p=10)
                    else: cwd_indices=[]

                    self.ground_area += 1
                    if len(indices) > 5:
                        self.canopy_area += 1
                    if len(ground_veg_indices) > 5:
                        self.ground_veg_area += 1
                    if len(cwd_indices) > 5:
                        self.cwd_area += 1

        print("Canopy Cover Fraction:", np.around(self.canopy_area / self.ground_area, 3))
        print("Understory Veg Fraction:", np.around(self.ground_veg_area / self.ground_area, 3))
        print("Coarse Woody Debris Fraction:", np.around(self.cwd_area / self.ground_area, 3))
        # max_z = np.max(np.hstack((self.stem_points[:, 2], self.vegetation_points[:, 2], self.cwd_points[:, 2], self.terrain_points[:, 2])))
        # min_z = np.min(np.hstack((self.stem_points[:, 2], self.vegetation_points[:, 2], self.cwd_points[:, 2], self.terrain_points[:, 2])))
        # self.z_range = max_z - min_z


    # @jit
    def interpolate_cyl(self, cyl1, cyl2, resolution):
        """
        Convention to be used
        cyl_1 is the higher cylinder
        cyl_2 is the lower cylinder
        """
        length = math.dist(cyl1[:3],cyl2[:3])
        points_per_line = int(np.ceil(length / resolution))
        
        if points_per_line > 1 :

            xyzinterp = np.linspace(cyl1[:3], cyl2[:3], points_per_line, endpoint=False, axis=0)[1:,:]
            # xyzinterp = xyzinterp[1:,:]  # remove the duplicated cyl1
            
            interpolated = np.zeros((xyzinterp.shape[0], len(cyl1)))
            interpolated[:, :3] = xyzinterp

            normal = (cyl1[:3] - cyl2[:3]) / np.linalg.norm(cyl1[:3] - cyl2[:3])

            if normal[2] < 0:  # something went wrong, we always interpolate from higher to lower
                # normal[:3] = normal[:3] * -1
                interpolated=cyl1
            else :
                interpolated[:, 3:6] = normal

                interpolated[:, self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
                interpolated[:, self.cyl_dict['branch_id']] = cyl1[self.cyl_dict['branch_id']]
                interpolated[:, self.cyl_dict['parent_branch_id']] = cyl2[self.cyl_dict['branch_id']]
                
                temp = np.array([cyl1[self.cyl_dict['radius']], cyl2[self.cyl_dict['radius']]]) # use to remove zero radii from minimum selection
                if np.max(temp) == 0 : interpolated[:, self.cyl_dict['radius']] = 0
                else : interpolated[:, self.cyl_dict['radius']] = np.min(np.atleast_2d(temp[temp>0]))

                ##  Leave CCI=0 for interpolated cylinders !!! TODO
                if len(cyl1)>self.cyl_dict['CCI']:
                    interpolated[:, self.cyl_dict['CCI']] = np.min(np.array([cyl1[self.cyl_dict['CCI']], cyl2[self.cyl_dict['CCI']]]))

                if len(cyl1)>self.cyl_dict['main_stem']:
                    interpolated[:, self.cyl_dict['main_stem']] = cyl1[self.cyl_dict['main_stem']]

                # if len(cyl1)>self.cyl_dict['segment_angle_to_horiz']:
                #     interpolated[:, self.cyl_dict['segment_angle_to_horiz']] = np.mean([cyl1[self.cyl_dict['segment_angle_to_horiz']], cyl2[self.cyl_dict['segment_angle_to_horiz']]])                    

        else: 
            middle_point = np.mean([cyl1[:3],cyl2[:3]], axis=0)
            interpolated = np.hstack((middle_point, cyl2[3:]))

        return interpolated

    @classmethod
    def compute_angle(cls, normal1, normal2=None):
        """
        Computes the angle in degrees between two 3D vectors.

        Args:
            normal1:
            normal2:

        Returns:
            theta: angle in degrees
        """
        normal1 = np.atleast_2d(normal1)
        normal2 = np.atleast_2d(normal2)

        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum('ij,ij->i', norm1, norm2), -1, 1)
        theta = np.degrees(np.arccos(dot))
        return theta

    # @jit
    def cylinder_sorting(self, cylinder_array, angle_tolerance, search_angle, distance_tolerance):
        """
        Step 1 of sorting initial cylinders into individual trees.
        For a cylinder to be joined up with another cylinder in this step, it must meet the below conditions.

        All angles are specified in degrees.
            cylinder_array:
                The Numpy array of cylinders created during cylinder fitting.

            angle_tolerance:
                Angle tolerance refers to the angle between major axis vectors of the two cylinders being queried. If
                the angle is less than "angle_tolerance", this condition is satisfied.
                TODO make sure that the inner angle is computed!!! Diff in z must be >0

            search_angle:
                Search angle refers to the angle between the major axis of cylinder 1, and the vector from cylinder 1's
                centre point to cylinder 2's centre point.
                TODO not sure if this makes sense together with angle_tolerance
				if the two cylinders/segments are long this condition will connect two separate close trees
                angle should be very small - but then may fail for leaning trees!!!
            distance_tolerance:
                Cylinder centre points must be within this distance to meet this condition. Think of a ball of radius
                "distance_tolerance".

        Returns: The sorted cylinder array.
        """

        def within_angle_tolerance(normal1, normal2, angle_tolerance):
            """Checks if normal1 and normal2 are within "angle_tolerance"
            of each other."""
            theta = self.compute_angle(normal1, normal2)
            # return abs((theta > 90) * 180 - theta) <= angle_tolerance
            return theta<=angle_tolerance

        def criteria_check(cyl1, cyls, angle_tolerance, search_angle):
            """
            Decides which cylinders in cyls should be joined to cyl1 and if they are the same tree.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            vector_array = cyls[:, :3] - np.atleast_2d(cyl1[:3])
            condition1 = within_angle_tolerance(cyl1[3:6], cyls[:, 3:6], angle_tolerance)
            # condition2 = within_angle_tolerance(cyl1[3:6], vector_array, search_angle)
            dists = np.linalg.norm(cyls[:,:2] - cyl1[:2], axis=1)
            condition2 = dists < cyl1[self.cyl_dict['radius']]*2
            condition3 = cyls[:, self.cyl_dict['radius']] < cyl1[self.cyl_dict['radius']]*1.05
            # cyls[np.logical_and(condition1, condition2, condition3), self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
            # cyls[np.logical_and(condition1, condition2, condition3), self.cyl_dict['parent_branch_id']] = cyl1[self.cyl_dict['branch_id']]
            
            # return cyls
            # return np.logical_and(condition1, condition2, condition3)
            return np.logical_and(condition1, condition3)

        max_tree_label = 1
         
        unsorted_points = cylinder_array
        sorted_points = np.zeros((0, unsorted_points.shape[1]))

        total_points = len(unsorted_points)
        while unsorted_points.shape[0] > 1:
            if sorted_points.shape[0] % 200 == 0:
                print('\r', np.around(sorted_points.shape[0] / total_points, 3), end='')

            # TODO find all assigned points and move them to sorted_points
            current_point_index = np.argmin(unsorted_points[:, 2]) # sorting the points by z-value from ground up
            current_point = unsorted_points[current_point_index] # the lowest point in the segment
            if current_point[self.cyl_dict['tree_id']] == 0:
                current_point[self.cyl_dict['tree_id']] = max_tree_label
                max_tree_label += 1
                                  
            kdtree = spatial.cKDTree(unsorted_points[:, :3], leafsize=1000)
            results = kdtree.query_ball_point(np.atleast_2d(current_point)[:, :3], r=distance_tolerance)[0]
            results.remove(current_point_index)
            mask = criteria_check(current_point,unsorted_points[results],angle_tolerance,search_angle)  # proper neighbours
            if not np.any(mask): 
                # current point has no neighbours (it's noise), therefore discard it
                unsorted_points = np.delete(unsorted_points, current_point_index, axis=0 ) # TODO: test it       
                continue
            # current point has proper neighbours - assign the current tree_id  
            results = np.array(results)       
            unsorted_points[results[mask], self.cyl_dict['tree_id']] = current_point[self.cyl_dict['tree_id']]
            unsorted_points[results[mask], self.cyl_dict['parent_branch_id']] = current_point[self.cyl_dict['branch_id']]
            
            # move the current point to Assigned array
            sorted_points = np.vstack((sorted_points, current_point))
            # neighbours are staying in the original array to be searched for more neighbours (other segments of this tree)
            
            # remove the current point from the unassigned
            unsorted_points = np.delete(unsorted_points, current_point_index, axis=0 )

        print('1.000\n')
        return sorted_points

    @classmethod
    def make_cyl_visualisation(cls, cyl):
        """Creates a 3D point cloud representation of a circle."""
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0], cyl[1], cyl[2], cyl[6])
        points = MeasureTree.rodrigues_rot(p - cyl[:3], [0, 0, 1], cyl[3:6])
        index_offset = len(cyl)-3
        points = np.hstack((points + cyl[:3], np.zeros((points.shape[0],index_offset))))
        points[:, -index_offset:] = cyl[-index_offset:]
        return points

    @classmethod
    def points_along_line(cls, x0, y0, z0, x1, y1, z1, resolution=0.05):
        """Creates a point cloud representation of a line."""
        points_per_line = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        Xs = np.atleast_2d(np.linspace(x0, x1, points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0, y1, points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0, z1, points_per_line)).T
        return np.hstack((Xs, Ys, Zs))

    @classmethod
    def create_3d_circles_as_points_flat(cls, x, y, z, r, circle_points=15):
        """Creates a point cloud representation of a horizontal circle at coordinates x, y, z. and of radius r.
        Circle points is the number of points to use to represent each circle."""
        angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
        points = np.zeros((0, 3))
        for i in angle_between_points:
            x2 = r * math.cos(i) + x
            y2 = r * math.sin(i) + y
            point = np.array([[x2, y2, z]])
            points = np.vstack((points, point))
        return points

    @classmethod
    def rodrigues_rot(cls, points, vector1, vector2):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        if points.ndim == 1:
            points = points[np.newaxis, :]

        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        k = np.cross(vector1, vector2)
        if np.sum(k) != 0:
            k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(vector1, vector2))

        P_rot = np.zeros((len(points), 3))
        for i in range(len(points)):
            P_rot[i] = points[i] * math.cos(theta) + np.cross(k, points[i]) * math.sin(theta) + k * np.dot(k, points[i]) * (
                    1 - math.cos(theta))
        return P_rot

    @classmethod
    def fit_circle_3D(cls, points, normal, P_mean):    # def fit_circle_3D(cls, points, normal):
        """
        Fits a circle using Random Sample Consensus (RANSAC) to a set of points in a plane perpendicular to vector V.

        Args:
            points: Set of points to fit a circle to using RANSAC.
            V: Axial vector of the cylinder you're fitting.

        Returns:
            cyl_output: numpy array of the format [[x, y, z, x_norm, y_norm, z_norm, radius, CCI, 0, 0, 0, 0, 0, 0]]
        """

        CCI = 0
        r = 0
        xc = 0
        yc = 0
        inliers = []

        P = points[:, :3]
        # P_mean = np.mean(P, axis=0)  Passing the original skeleton point as P_mean:
        #    This will provide a proper center at forks with wrong V. The V itself will still be wrong (TODO fix in parent call)
        P_centered = P - P_mean

        if P_centered.shape[0] > 20 :  # Hard-Coded parameter 
            # Rotate points to point their direction vector to the sky (for 2D ransac fitting)
            P_xy = MeasureTree.rodrigues_rot(P_centered, normal, [0, 0, 1])

            # remove outliers
            P = P_xy[:,:2]
            # D = np.linalg.norm(P[:, np.newaxis,:]-P[np.newaxis, :, :], axis=-1)
            # a=D[np.triu_indices_from(D,k=1)]
            
            P_xy=np.zeros([0,2])
            # convert to polar coordinates using the mean as center
            polar_d = np.linalg.norm(P, axis=1)
            polar_phi = np.around(np.degrees(np.arctan2(P[:,0],P[:,1]))) % 360
            # get the closest 3 points for every polar angle to bias ransac toward the tree core
            for phi in np.unique(polar_phi):
                mask = polar_phi == phi # points corresponding to this phi
                if np.any(mask):
                    a=list(polar_d[mask])
                    if len(a) <=4 :
                        tol = np.max(a)
                    else:
                        a.sort(reverse=True)  # smallest number at the top of the list (i.e. last member)
                        # [a.pop() for i in range(3)]  # the full list but we need only up to the 3rd smallest
                        for _ in range(4): tol = a.pop()
                P_xy = np.vstack((P_xy, P[mask,:][polar_d[mask] <= tol]))               

            # rt = (bark/100)*2  # convert to meters and double it for beam dispersion
            # if rt < 0.02: rt = 0.02  # when bark is thinner than the beam dispersion
            rt=0.03
            # with warnings.catch_warnings(): # not safe for threading. Patched with Ignore for now

            # Fit circle in new 2D coords using RANSAC          
            model_robust, inliers = ransac((P_xy[:, :2]), CircleModel, min_samples=int(P_xy.shape[0] * 0.3),
                                        residual_threshold=rt, max_trials=10000)  # test is_data_valid parameter
                                        # is_model_valid=is_CCI_good(model, samples)
                                        # residual_threshold is the maximum distance for a data point to be classified as inlier
                                        # adjust residual_threshold according to the bark thickness! Redwoods only - manual or compensate with slice thickness
                                    
                                    # TODO is_data_valid=is_data_valid,
            
            # EllipseModel TODO
            if not model_robust or inliers is None or not any(inliers):
                # if ransac fails keep the center of the cluster and assign r=0 and CCI=0
                # also assign normal vector to [0,0,1] because the calculated might be wrong (wrong PCA from previous step) and be the cause the ransac failure
                
                print('.. Failed Circle fitting')
                r = (np.max(P_xy,axis=0)-np.min(P_xy, axis=0))/2*(9/10)
                cyl_output = np.array([[P_mean[0],P_mean[1],P_mean[2], 0,0,1, r, 0,0]])
                plot_circle_points(f'{round(time.time()%1000)}', P_mean[0], P_mean[1], 0, P_xy, 2, inliers=None)
                return cyl_output
                # return None

                # except ValueError as e:  # in scikit-image 1.0 this warning will become a ValueError
                # UserWarning: Input data does not contain enough significant data points
                # UserWarning: "No inliers found"

            # model_robust.residuals(P_xy[inliers])

            xc, yc = model_robust.params[0:2]
            r = round(model_robust.params[2], 2)
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

            if DEBUG:
                # matplotlib.patches.Ellipse(xy, width, height, angle=0,)
                
                a = np.linalg.norm(P_xy, axis=1)
                # plot_fitted_circle()
                fig1 = plt.figure(figsize=(10, 6))
                ax = fig1.add_subplot(1, 2, 1)
                ax.set_title(f'distances within points')
                ax.hist(a,100)

                ax1 = fig1.add_subplot(1, 2, 2)
                ax1.set_title(f"CCI = {CCI}, radius = {r}")
                ax1.axis('equal')
                # plt.scatter(P[:,0], P[:,1],  marker='.', cmap='Greens')
                ax1.scatter(P_xy[:,0], P_xy[:,1],  marker='.', cmap='Blues')
                ax1.scatter(P_xy[inliers,0], P_xy[inliers,1],marker='.', cmap='Greens')
                circle_outline = plt.Circle(xy=(xc,yc), radius=r, fill=False, edgecolor='k', zorder=3)
                ax1.add_patch(circle_outline)
                plt.show()

            # # Refit a circle on the points inside/outsie the perimeter of the first circle 
            # TODO - add checks for NoneType
            # TODO - use second fitting for tiny trees/branches only!! radius<0.05m
            #
            # dists=np.sqrt(np.sum((P_xy[:,:2] - [xc,yc])**2, axis=1))  # find distances to the cirlce center
            # P = np.vstack((P_xy[dists >= r, :2], P_xy[inliers, :2])) # inner or outer points only
            # P_xy = P

            # # # have to reset these - not sure why - maybe the multithreading
            # r=0
            # xc=0
            # yc=0
            # inliers = []
            # model2, inliers = ransac((P_xy), CircleModel, min_samples=int(P_xy.shape[0]*0.3), residual_threshold=0.02, max_trials=1000, random_state=0)
            # xc, yc = model2.params[0:2]
            # r = model2.params[2]
            # CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])      

        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
        cyl_output = np.array([[cyl_centre[0, 0], cyl_centre[0, 1], cyl_centre[0, 2], normal[0], normal[1], normal[2],
                                    r, CCI,0]])
        return cyl_output


    def point_cloud_annotations(self, character_size, xpos, ypos, zpos, offset, text):
        """
        Point based text visualisation. Makes text viewable as a point cloud.

        Args:
            character_size:
            xpos: x coord.
            ypos: y coord.
            zpos: z coord.
            offset: Offset for the x coord. Used to shift the text depending on tree radius.
            text: The text to be displayed.

        Returns:
            nx3 point cloud of the text.
        """
        def convert_character_cells_to_points(character):
            character = np.rot90(character, axes=(1, 0))
            index_i = 0
            index_j = 0
            points = np.zeros((0, 3))
            for i in character:
                for j in i:
                    if j == 1:
                        points = np.vstack((points, np.array([[index_i, index_j, 0]])))
                    index_j += 1
                index_j = 0
                index_i += 1

            roll_mat = np.array([[1, 0, 0],
                                 [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
                                 [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
            points = np.dot(points, roll_mat)
            return points

        def get_character(char):
            if char == ':':
                return self.character_viz[self.characters.index('semiC')]
            elif char == '.':
                return self.character_viz[self.characters.index('dot')]
            elif char == ' ':
                return self.character_viz[self.characters.index('space')]
            elif char == 'M':
                return self.character_viz[self.characters.index('_M')]
            else:
                return self.character_viz[self.characters.index(char)]

        text_points = np.zeros((11, 0))
        for i in text:
            text_points = np.hstack((text_points, np.array(get_character(str(i)))))
        points = convert_character_cells_to_points(text_points)

        points = points * character_size + [xpos + 0.2 + 0.5 * offset, ypos, zpos]
        return points

    @classmethod
    def fit_cylinders(cls, skeleton_points, point_cloud, cluster_id, num_neighbours=3):
        """
        Starts by fitting a 3D line to the skeleton points cluster provided.
        Uses this line as the major axis/axial vector of the cylinder to be fitted.
        Returns a series of fitted circles perpendicular to this axis to the point cloud of this particular stem segment.

        Args:
            skeleton_points: A single sequence of skeleton points which should represent a segment of a tree/branch.
            point_cloud: The cluster of points belonging to the segment.
            num_neighbours: The number of skeleton points to use for fiinding the slope of the segment. lower numbers
                            have fewer points to fit a circle to, but higher numbers are negatively affected by curved
                            branches. Recommend leaving this as it is.

        Returns:
            cyl_array: a numpy array based representation of the fitted circles/cylinders.
        """
        # point_cloud = point_cloud[:, :3]
        # skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 9))       #h hard coded - only 8 dimensions are assigned here + 1 dimension in parent call
        
        # Find major slope vector of skeleton - This should be done for each group. Here, can be grossly wrong for curved segments.
        # line_centre = np.mean(skeleton_points[:, :3], axis=0)
        # _, evalues, vh = np.linalg.svd(line_centre - skeleton_points)
        # line_v_hat = vh[0] / np.linalg.norm(vh[0])

        # skeleton_points = skeleton_points[np.argsort(skeleton_points[:,2], :)] will be wrong when forks and multiple branching

        nn = NearestNeighbors()
        vh = np.empty(0)
        i=0 # DEBUG only
        group = skeleton_points # to be used when skeleton is <= num_neighbours

        while skeleton_points.shape[0] > 1:
            # with config_context(target_offload="gpu:0"):
            length = 0
            starting_point = skeleton_points[np.argmin(skeleton_points[:, 2])]
            
            if skeleton_points.shape[0] <= num_neighbours :                
                group = skeleton_points 
                line_centre = np.mean(group[:, :3], axis=0)
                if vh.size == 0 :
                    # find the direction vector of this group
                    _, evalues, vh = np.linalg.svd(group - line_centre)
                    # print(evalues)
                    line_v_hat = vh[0] / np.linalg.norm(vh[0])
                    line_v_hat = np.sign(line_v_hat[2]) * line_v_hat  # if direction vector is pointing down, flip it around the other way.                    

            else:
                if skeleton_points.shape[0] > num_neighbours :                    
                    nn.fit(skeleton_points)             
                    group_indices = nn.kneighbors(np.atleast_2d(starting_point), n_neighbors=num_neighbours)[1][0]
                    group = skeleton_points[group_indices]
                    line_centre = np.mean(group[:, :3], axis=0)

                    # find the direction vector of this group
                    _, _, vh = np.linalg.svd(group - line_centre)
                    line_v_hat = vh[0] / np.linalg.norm(vh[0])
                    line_v_hat = np.sign(line_v_hat[2]) * line_v_hat  # if direction vector is pointing down, flip it around the other way.                    
                    
            # line_centre = np.mean(group[:, :3], axis=0)  ## TODO try median here so we don't merge forks together        
                                    
            if np.abs(line_v_hat[2])>.7 :   # fit circle only if the lean at that point is close to vertical
            
                length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
                length = length / (num_neighbours-1)    # number of interskeleton points distances to consider for grabbing stem points 

                # Do a projection on the 1st singular vector to find the right points for circle fitting
                # mask = np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length)
                mask = np.linalg.norm(line_v_hat * (point_cloud - starting_point), axis=1) < (length/2) # the last should be ~= slice_thickness                                                                                        
                # using the starting_point instead of line_centre is the right approach here !!!

                ## fit circles only if # of points is at least min_cluster_size and the spread is at least 4 cm
                
                if  np.sum(mask) < 20: # if less than 30 points - low-confidence cylinder
                    skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0) 
                #     cylinder = np.array([[starting_point[0],starting_point[1],starting_point[2], 0,0,1, 0, 0, 0]])
                #     cyl_array = np.vstack((cyl_array, cylinder))  # NOT MUCH USE
                    continue

                plane_slice = point_cloud[mask] 
                cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat, starting_point[:3])
                
                if DEBUG and i<10:
                    fig1 = plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.scatter3D(group[:,0], group[:,1], group[:,2],color='orange')
                    ax.scatter3D(plane_slice[:,0], plane_slice[:,1], plane_slice[:,2], color='blue')
                    plt.close()
                    i+=1

                # if DEBUG :                         
                #     if i<10  :  
                #     # plotting the result for QC
                #         fig1 = plt.figure(figsize=(9, 7))
                #         ax1 = fig1.add_subplot(1, 1, 1)
                #         ax1.set_title("Fitted circle to stem points")
                #         ax1.axis('equal')
                #         circle_outline = plt.Circle(xy=(cylinder[0,0], cylinder[0,1]), radius=cylinder[0,6], fill=False, edgecolor='b', zorder=3)
                #         ax1.add_patch(circle_outline)
                #         plane_slice = line_v_hat * (plane_slice - starting_point)
                #         plt.scatter(plane_slice[:,0], plane_slice[:,1],c=plane_slice[:,4], cmap='Blues')
                #         # plt.scatter(P_xy[inliers,0], P_xy[inliers,1])
                #         fig1.savefig(f'c:/Temp/{cluster_id}_{i}circle.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
                #         plt.close()
                #         # fig1.clf()
                #         i +=1
            
            else : # low-confidence cylinder
                # cylinder = np.array([[starting_point[0],starting_point[1],starting_point[2], 0,0,1, 0, 0,0]])  
                cylinder = None

            ## add cylinder to results
            if np.any(cylinder) :
                cyl_array = np.vstack((cyl_array, cylinder))
            
            ## all done for the current point 
            skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0)
            # skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0) # use this for speedup when stem detection is good

            ## Fit to the final 2 skeleton points only if at least 1 subsection was not near horizontal
            if (skeleton_points.shape[0] == 2) and (length > 0) :
                # Only 2 skeleton points left from this section
                
                starting_point = skeleton_points[0,:]
                mask = np.linalg.norm(abs(line_v_hat * (point_cloud - starting_point)), axis=1) < (length/2) 
                if np.sum(mask) > 20 :
                    plane_slice = point_cloud[mask] 
                    cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat, starting_point[:3])   
                    if np.any(cylinder) :           
                        cyl_array = np.vstack((cyl_array, cylinder))
                skeleton_points = np.delete(skeleton_points, 0, axis=0)
            	
                # last point
                mask = np.linalg.norm(abs(line_v_hat * (point_cloud - skeleton_points)), axis=1) < (length/2) 
                if np.sum(mask) > 20 :
                    plane_slice = point_cloud[mask]
                    cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat, skeleton_points[:3])  
                    if np.any(cylinder) :           
                        cyl_array = np.vstack((cyl_array, cylinder))

        return cyl_array


    @classmethod
    def cylinder_cleaning_multithreaded(cls, args):
        def compute_frustum_volume(diameter_1, diameter_2, height):
            radius_1 = 0.5 * diameter_1
            radius_2 = 0.5 * diameter_2
            volume = (1 / 3) * np.pi * height * (radius_1**2 + radius_2**2 + radius_1*radius_2)
            return volume
        """
        Cylinder Cleaning
        Works on a single tree worth of cylinders at a time.
        Starts at the lowest (z axis) cylinder.
        Finds neighbouring cylinders within "cleaned_measurement_radius".
        If no neighbours are found, cylinder is deleted.
        If neighbours are found, find the neighbour with the highest circumferential completeness index (CCI). This is
        probably the most trustworthy cylinder in the neighbourhood.

        # Aglika - Problem - interpolated cylinders have CCI=0 - fixed with mean
        
        If there are enough neighbours, use those with CCI >= the 30th percentile of CCIs in the neighbourhood.
        Use the medians of x, y, vx, vy, vz, radius as the cleaned cylinder values.
        Use the mean of the z coords of all neighbours for the cleaned cylinder z coord.
        """
        sorted_cylinders, cleaned_measurement_radius, cyl_dict = args
        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))
		# TODO TODO read again about leaf_size !!
        tree = BallTree(sorted_cylinders[:, :3]) #, leaf_size=7   # find the six closest neighbours - above and below for midtree; 
        if sorted_cylinders.shape[0] < 7 :
            _, ind = tree.query(sorted_cylinders[:, :3], k=sorted_cylinders.shape[0])   # k=7 will get 3 below and 3 above + the current cylinder
        else:
            _, ind = tree.query(sorted_cylinders[:, :3], k=7)   # k=7 will get 3 below and 3 above + the current cylinder
                                                                # at a fork - will get 4 above and 2 below 
                                                                # therefore the median will tend towards the fork radius
        
        radii = np.array([np.median(sorted_cylinders[row, cyl_dict['radius']]) for row in ind]) # row has an array of neighbours
        sorted_cylinders[:, cyl_dict['radius']] = radii
        coords = np.array([np.median(sorted_cylinders[row, :2], axis=0) for row in ind]) # CHECK THIS TODO TODO - z should not change
        sorted_cylinders[:, :2] = coords
        coords = np.array([np.median(sorted_cylinders[row, 3:6], axis=0) for row in ind])
        sorted_cylinders[:, 3:6] = coords  # new normal vector values
        # originally keeps the CCI as it is - but it is 0 for all interpolated cylinders - I think I changed it to the average
        # so will get the mean CCI from the neighbours
        CCI = np.array([np.mean(sorted_cylinders[row, cyl_dict['CCI']]) for row in ind])
        sorted_cylinders[:, cyl_dict['CCI']] = CCI

        cleaned_cyls = sorted_cylinders     # the cleaning below does more harm than good so we are skipping it for now
        while 0:                            # lots of good tree skeletons are lost in the code below
                                            # most of the cleaning is done during assignment
                                            # further noise removal can be done in the interpolation step TODO
        # while sorted_cylinders.shape[0] >= 4 : # self.parameters["min_tree_cyls"] :
            start_point_idx = np.argmin(sorted_cylinders[:, 2])
            best_cylinder = sorted_cylinders[start_point_idx] # initialisation
            sorted_cylinders = np.delete(sorted_cylinders, start_point_idx, axis=0)
            kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
            results = kdtree.query_ball_point(best_cylinder[:3], cleaned_measurement_radius)

            neighbours = sorted_cylinders[results]
            neighbours = np.vstack((neighbours, best_cylinder))
            if neighbours.shape[0] > 0:
                if np.max(neighbours[:, cyl_dict['CCI']]) > 0:
                    best_cylinder = neighbours[np.argsort(neighbours[:, cyl_dict['CCI']])][-1]  # choose cyl with highest CCI.
                # compute a percentile of the CCIs in the neighbourhood
                percentile_thresh = np.percentile(neighbours[:, cyl_dict['CCI']], 60)  # Aglika - 50 is good for small trees!! maybe increase for large trees
                mask = neighbours[:, cyl_dict['CCI']] >= percentile_thresh
                if neighbours[mask, :2].shape[0] > 0:
                    best_cylinder[:3] = np.median(neighbours[mask, :3], axis=0)
                    best_cylinder[3:6] = np.median(neighbours[mask, 3:6], axis=0)        # normal vector
                    best_cylinder[cyl_dict['radius']] = np.mean(neighbours[mask, cyl_dict['radius']], axis=0) # orig is max
                    best_cylinder[cyl_dict['CCI']] = np.mean(neighbours[mask, cyl_dict['CCI']], axis=0)         
            
            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            sorted_cylinders = np.delete(sorted_cylinders, results, axis=0)
        
        # volume_cyls = deepcopy(cleaned_cyls)
        # volume = 0
        # while volume_cyls.shape[0] > 2:
        #     lowest_point_idx = np.argmin(volume_cyls[:, 2])
        #     lowest_point = volume_cyls[lowest_point_idx]
        #     volume_cyls = np.delete(volume_cyls, lowest_point_idx, axis=0)
        #     tree = BallTree(volume_cyls[:, :3], leaf_size=2)
        #     _, neighbour_idx = tree.query(np.atleast_2d(lowest_point[:3]), 1)
        #     neighbour_idx = int(neighbour_idx)
        #     neighbour = volume_cyls[neighbour_idx]
        #     distance = np.linalg.norm(neighbour[:3] - lowest_point[:3])
        #     volume += compute_frustum_volume(lowest_point[cyl_dict['radius']], neighbour[cyl_dict['radius']], distance)
        # cleaned_cyls[:, cyl_dict['tree_volume']] = volume
        
        return cleaned_cyls

    @staticmethod
    def inside_conv_hull(point, hull, tolerance=1e-5):
        """Checks if a point is inside a convex hull."""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


    @classmethod
    def circumferential_completeness_index(cls, fitted_circle_centre, estimated_radius, slice_points):
        """
        Computes the Circumferential Completeness Index (CCI) of a fitted circle.

        Args:
            fitted_circle_centre: x, y coords of the circle centre
            estimated_radius: circle radius
            slice_points: the points the circle was fitted to

        Returns:
            CCI
        """
        # sector_angle = 4.5  # degrees
        sector_angle = 10  # degrees
        num_sections = int(np.ceil(360 / sector_angle))
        sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

        centre_vectors = slice_points[:, :2] - fitted_circle_centre # distances from the circle center
        norms = np.linalg.norm(centre_vectors, axis=1)              # normalised distances

        centre_vectors = centre_vectors / np.atleast_2d(norms).T
        centre_vectors = centre_vectors[np.logical_and(norms >= 0.8 * estimated_radius,
                                                       norms <= 1.2 * estimated_radius)]

        sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
        CCI = np.sum([np.any(
            np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", np.atleast_2d(sector_vector), centre_vectors), -1, 1)))
            < sector_angle / 2) for sector_vector in sector_vectors]) / num_sections
        CCI = np.around(CCI*100)
        return CCI


    @classmethod
    def slice_clustering(cls, new_slice, min_cluster_size, min_samples, eps, size_threshold):
        """Helper function for clustering stem slices and extracting the skeletons of these stems."""
        # new_slice - a horizontal slice of the point cloud with height==slice_thickness
        # min_cluster_size - minimum number of points to be considered as a cluster
        # returns   - array of stem_points belonging to each cluster
        #           - array of skeleton points belonging to each cluster
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0,3))
        true_mask=np.zeros((new_slice.shape[0]))
       
        # new_slice = cluster_hdbscan(new_slice[:, :3], min_cluster_size, min_samples=min_samples, eps=0.01)
        new_slice = cluster_hdbscan(new_slice[:, :3], min_cluster_size,min_samples=min_samples, eps=eps)            
        
        if new_slice[new_slice[:,-1]>-1].shape[0] >= size_threshold: # makes sense on steep terain, etc
            # plt.clf()
            # plt.scatter(new_slice[:,0], new_slice[:,1], s=2, c=new_slice[:,-1], cmap='Reds')
            # for x in np.unique(new_slice[:,-1]): plt.annotate(x, (np.mean(new_slice[new_slice[:,-1]==x,:2], axis=0)))
            # plt.show()

            id, size = np.unique(new_slice[:,3],return_counts=True)

            ## mark small clusters as noise
            id[size<size_threshold]=-1    #  slice_thickness / height_above_dtm / scaled
            # print(f'num clusters = {sum(id>-1)}')

            # combine stem points and skeleton points into cluster_array_internal for the remaining clusters
            for cluster_id in id[id>-1]: # -1 ids are noise                    
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                centre = np.mean(cluster[:, :3], axis=0)            # use the mean because the tree is supposed to be hollow ? the median will be on the bark
                # create output
                medians = np.vstack((medians, centre))
                cluster_array_internal = np.vstack((cluster_array_internal, np.hstack((cluster[:, :3],  np.zeros((cluster.shape[0], 3)) + centre))))
                # mask of valid clusters
                true_mask = np.logical_or(true_mask, new_slice[:, -1] == cluster_id)  # add this cluster points to the valid points mask               

        return cluster_array_internal, medians, true_mask

    @classmethod
    def within_angle_tolerances(cls, normal1, normal2, angle_tolerance):
        """Checks if normal1 and normal2 are within "angle_tolerance"
        of each other."""
        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum("ij, ij->i", norm1, norm2), a_min=-1, a_max=1)
        theta = np.degrees(np.arccos(dot))
        return abs((theta > 90) * 180 - theta) <= angle_tolerance

    @classmethod
    def within_search_cone(cls, normal1, vector1_2, search_angle):
        """Checks if the angle between vector1_2 and normal1 is less than search_angle."""
        norm1 = normal1 / np.linalg.norm(normal1)
        if not (vector1_2 == 0).all():
            norm2 = vector1_2 / np.linalg.norm(vector1_2)
            dot = np.clip(np.dot(norm1, norm2), -1, 1)
            theta = math.degrees(np.arccos(dot))
            return abs((theta > 90) * 180 - theta) <= search_angle
        else:
            return False

    @classmethod
    def threaded_cyl_fitting(cls, args):
        """Helper function for multithreaded cylinder fitting."""
        skel_cluster, point_cluster, cluster_id, num_neighbours, cyl_dict = args  
        cyl_array = np.zeros((0, 9))  # only 9 dimensions are assigned here
        
        # if skel_cluster.shape[0] >= num_neighbours:
        cyl_array = cls.fit_cylinders(skel_cluster[:,:3], point_cluster[:,:3], cluster_id, num_neighbours=num_neighbours)
        cyl_array[:, cyl_dict['branch_id']] = cluster_id
        
        return cyl_array
    

    # Main Circle Fitting Starts Here
    #
    def find_skeleton(self):
        skeleton_array = np.zeros((0, 3))
        point_data = np.empty((0, 6))

        # update single_increment_height according to z values
        self.parameters['single_increment_height'] = np.min(self.stem_points[:, 2]) + self.parameters['single_increment_height']

        slice_heights = np.linspace(np.min(self.stem_points[:, 2]), self.parameters['single_increment_height'], 
            int(np.ceil(( self.parameters['single_increment_height'] - np.min(self.stem_points[:, 2])) / self.slice_increment)), endpoint=False)  # 
        
        if np.max(self.stem_points[:, 2]) > self.parameters['single_increment_height'] :
            slice_heights = np.concatenate([slice_heights, np.linspace( self.parameters['single_increment_height'], np.max(self.stem_points[:, 2]), 
                int(np.ceil((np.max(self.stem_points[:, 2]) -  self.parameters['single_increment_height']) / (self.slice_increment*2))))], axis=0)
        
        else: # plot of smaller trees - adjust cluster size values
            self.parameters['cluster_size_threshold'] = self.parameters['cluster_size_threshold'][-1]

        ransac_points = np.empty((0,self.stem_points.shape[1]))
        print("Making and clustering slices ...")
        i = 0
        max_i = slice_heights.shape[0]

        # eps - # maximum distance between neighbouring cluster points
        # default is 1cm but we want it bigger for very rough bark (redwoods OR for sparse point clouds) 
        # not much beam response from the valley points on bark or just not enough points
        # TODO  Done - added eps to initial parameters                                                               
        eps = self.parameters['eps']
        plot_slope_diff = np.percentile(self.DTM[:,2],99) - np.min(self.DTM[:,2]) # to be used as adjustment to slice height

        print('---Finding clusters of stem points---')
        for slice_height in slice_heights:  # slicing the plot horizontaly # --- Joo edited, for saving, need to be uncommented
            if i % 10 == 0:
                print('\r', i, '/', max_i, end='')
            i += 1
            slice_mask = np.logical_and(self.stem_points[:, 2] >= slice_height, self.stem_points[:, 2] < slice_height + self.slice_thickness)
            new_slice = self.stem_points[slice_mask]
                    

            ## update cluster_size_threshold as we move higher above the ground
            t_value = int(np.floor((slice_height - min(slice_heights))/(5 + plot_slope_diff)))  # !! hardcoded for now TODO
            if t_value > (np.size(self.cluster_size_threshold)-1): t_value=-1 # get the minimum cluster size for points higher than 10m above the ground

            if new_slice.shape[0] >  self.cluster_size_threshold[t_value]:
                
                ##----- Statistical outlier (removal SOR) on the slice
                if self.parameters['SOR_filter'] == True :
                    # plt.clf()
                    # plt.scatter(new_slice[:,0], new_slice[:,1], s=2, c=new_slice[:,9], cmap='Reds')
                    # plt.show()
                
                    nn = NearestNeighbors(n_neighbors=10).fit(new_slice[:, :3])
                    distances, indices = nn.kneighbors()
                    neighbours_mean_d = np.mean(distances, axis=1)
                    outliers = neighbours_mean_d > (np.mean(neighbours_mean_d) + 0.5*np.std(neighbours_mean_d))
                    new_slice = np.delete(new_slice, outliers, axis=0)
                    
                    # plt.scatter(new_slice[:,0], new_slice[:,1], s=2, c=new_slice[:,9], cmap='Purples')
                    # plt.show()

                clusters, skel, true_mask = MeasureTree.slice_clustering(new_slice, 
                                                                        self.parameters['min_cluster_size'], 
                                                                        self.parameters['min_samples'],
                                                                        eps, self.cluster_size_threshold[t_value])
                # true mask is a boolean of true and false clusters of stem points
                if np.any(true_mask) :
                    skeleton_array = np.vstack((skeleton_array, skel))
                    point_data = np.vstack((point_data, clusters)) # TODO TODO very poor data structures Skeleton_array duplicates some of point_data

                    ransac_points = np.vstack((ransac_points, new_slice[true_mask]))

        print('\r', max_i, '/', max_i, end='')
        print('\nDone\n') 
        # # save_file(self.output_dir + 'stem_points_clusters.laz', ransac_points, headers_of_interest=list(self.stem_dict), silent=False, offsets=self.offsets) # don't need at moment, Joo

        # Sanity check for number of skeleton points 
        if (skeleton_array.shape[0] < 1) :
            # sys.exit('Did not find any trees. Exiting...')  
            raise ValueError('Did not find any good clusters. Possibly a scan of very young trees!')

        print('Clustering skeletons...') # group points into tree segments
        # add a column for cluster_id
        skeleton_array = np.hstack((skeleton_array, np.zeros((skeleton_array.shape[0],1),dtype=int)))

        # split in two steps according to single_increment_height
        min_z = max(self.DTM[:,2]) # use the highest ground point as a base for height offsets
        # (1)
        mask = skeleton_array[:,2] < (min_z + self.parameters['single_increment_height'] + self.slice_increment)
        ids = cluster_dbscan(skeleton_array[mask,:3], eps=self.parameters['gap_connect']) #  eps=self.slice_increment * 2.02
        skeleton_array[mask,3] = ids.T
        skeleton_array = np.delete(skeleton_array, skeleton_array[:, -1] < 0, axis=0)
        # skeleton_array = skeleton_array[skeleton_array[:, -1] >-1 , :] # discard non-clustered skeleton points
        # (2)  To watch out - ids will start again from -1 
        max_id = max(ids)
        mask = skeleton_array[:,2] >= (min_z + self.parameters['single_increment_height'] + self.slice_increment)
        if sum(mask)>0:
            ids = cluster_dbscan(skeleton_array[mask,:3], eps=self.parameters['gap_connect']*2) #  eps=self.slice_increment * 2.02
            skeleton_array[mask,3] = (ids.T + max_id + 2)
            # skeleton_array = skeleton_array[skeleton_array[:, -1] != (max_id+1) , :] # corresponds to -1 in the second group of ids
            skeleton_array = np.delete(skeleton_array, skeleton_array[:, -1] == (max_id+1), axis=0)

        ## remove very small clusters -  Not good for Native or any bushy forest:        
        ids, indices, counts = np.unique(skeleton_array[:,-1], return_inverse=True, return_counts=True)
        ids = ids[counts>=self.parameters['num_neighbours']]
        skeleton_array = skeleton_array[np.in1d(indices, ids) , :] 

        print("---Saving skeleton and cluster array---")
        # save_file(self.output_dir + 'skeleton_cluster_visualisation.laz', skeleton_array, ['X', 'Y', 'Z', 'cluster'], offsets=self.offsets)  # --- Joo edited, for saving, need to be uncommented

        # split execution here TODO TODO

        # # print("Deleting isolated skeleton points...")
        # skel_tree = BallTree(skeleton_array, leaf_size=1000)     
        # results = (np.where(np.hstack(skel_tree.query_radius(skeleton_array, r=0.5, count_only=True))!=1))[0]
        # skeleton_array = skeleton_array[results,:]

        print("Making initial tree skeleton sections...")
        ransac_points = np.empty((0,3))
        input_data = []
        max_i = int(np.max(skeleton_array[:, -1]) + 1)

        # # point_data is XYZ of stem points and XYZ of skeleton points, i.e. [n,6]
        # #  partition space of skeletons (mean values of the stem clusters) 
        # stem_points_kdtree = spatial.cKDTree(point_data[:, 3:], leafsize=100000)

        # The code below matches skeleton points from point_data to points in skeleton_array
        # Skeleton_array has been filtered and has less members by now TODO TODO better data structures are needed 
        medians_kdtree = spatial.cKDTree(point_data[:, 3:], leafsize=100000) 
        
        # loop through tree segments
        for cluster_id in np.unique(skeleton_array[:, -1]):
            if int(cluster_id) % 100 == 0:
                print('+ \r', int(cluster_id), '/', max_i, end='')
        
            # assign stem points to a tree skeleton segment
            # TODO check if we can fit Hough - not here but either in hdbscan or circle fitting

            skel_cluster = skeleton_array[skeleton_array[:, -1] == cluster_id, :3]
            skel_tree = spatial.cKDTree(skel_cluster, leafsize=100000)
            results = np.unique(np.hstack(skel_tree.query_ball_tree(medians_kdtree, r=0.0001))) #  a list of skeleton points, not stem points 
            point_data_clean = point_data[results, :3]        # these are stem points       
            point_data_clean = np.unique(point_data_clean, axis=0)                                     
            input_data.append([skel_cluster[:, :3], point_data_clean[:, :3], cluster_id, self.num_neighbours,
                            self.cyl_dict]) 

            ransac_points = np.vstack((ransac_points, point_data_clean[:,:3])) 

        print('\r', max_i, '/', max_i, end='') 
        print('\nDone\n') 
        # save_file(self.output_dir + 'stem_points_for_ransac.laz', ransac_points, headers_of_interest=list(self.stem_dict)[:3], offsets=self.offsets) # Joo edited, 


        # TODO try to send data to GPU
        print("Starting cirlce fitting... This can take a while.")
        j = 1
        max_j = len(input_data)
        outputlist = []
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting, input_data):
                # print(i)          
                outputlist.append(i)
                print('\r', j, '/', max_j, end='')
                j += 1
        print('\nDone')
        
        full_cyl_array = np.vstack(outputlist)      
        print("Saving cylinder array...")
        # save_file(self.output_dir + 'ransac_cyls.laz', full_cyl_array, headers_of_interest=list(self.cyl_dict)[:full_cyl_array.shape[1]], offsets=self.offsets)  # --- Joo edited, for saving, need to be uncommented



    def stem_smoothing(self, trees_cyl_array) :
        # Ensure diameters are getting smaller as we move up the tree. Replace any bad diameter with the mean of the lower three good diameters TODO
        # Applies a median filter on the main stem to remove single outlier diameters
        # trees_cyl_array, headers = load_file(self.output_dir + 'all_trees_skeletons.laz', headers_of_interest=list(self.cyl_dict))

        # add 'main_stem' column if does not exist already
        if trees_cyl_array.shape[1] < self.cyl_dict['main_stem']+1 :
            trees_cyl_array = np.hstack((trees_cyl_array, np.zeros((trees_cyl_array.shape[0],1))))
        
        ## find trees that are split in two and put them together - use a data frame !!
        # for tree_id in np.unique(trees_cyl_array[:, self.cyl_dict['tree_id']]):
        #     this_tree_mask = trees_cyl_array[:,self.cyl_dict['tree_id']]==tree_id
        #     tree = deepcopy(trees_cyl_array[this_tree_mask])
        #     lowest_skeleton_point = tree[np.argmin(tree[:,2]),:]

        # # put together tree forks who are given unique id and assign parent_number
        # for tree_id in np.unique(trees_cyl_array[:, self.cyl_dict['tree_id']]):

        max_treeid = np.max(trees_cyl_array[:, self.cyl_dict['tree_id']])

        for tree_id in np.unique(trees_cyl_array[:, self.cyl_dict['tree_id']]):
            ## remove outliers - defined by a bigger diameter compared to lower diameteres
            # print(tree_id)
            if tree_id==8:
                print(tree_id)
            
            # also replace x, y, vx, vy, and vz
            this_tree_mask = trees_cyl_array[:,self.cyl_dict['tree_id']]==tree_id
            tree = deepcopy(trees_cyl_array[this_tree_mask])
            lowest_skeleton_point = tree[np.argmin(tree[:,2]),:]

            # discard the tree if it is too small (noise and bushes), too short (non-crop) or does not reach the ground (hanging branch, etc)
            # np.mean(tree[:,self.cyl_dict['CCI']]) < 75 or \
            if (tree.shape[0] < self.parameters['min_tree_cyls'] or \
                    lowest_skeleton_point[self.cyl_dict['height_above_dtm']] > self.parameters['tree_base_cutoff_height'] or \
                    np.max(tree[:,self.cyl_dict['height_above_dtm']]) < self.parameters['tree_stem_min_height']) : 
            
                trees_cyl_array = np.delete(trees_cyl_array, this_tree_mask, axis=0)                
                continue
            
            # get all cylinders in the DBH section +-30cm            
            dbh_section = tree[np.abs(tree[:,self.cyl_dict['height_above_dtm']] - self.parameters['dbh_height']) < .3, :]
            if len(dbh_section) == 0 :
                mask = tree[:, self.cyl_dict['height_above_dtm']]>self.parameters['dbh_height']
                # find the 2 lowest cylinders instead (section height = 2*slice_thickness)
                lowest_cyl = np.min(tree[mask,self.cyl_dict['height_above_dtm']])
                dbh_section = tree[np.logical_and(mask, tree[:,self.cyl_dict['height_above_dtm']] < lowest_cyl+.15), :]

            # # remove trees that are outside the plot radius, # delete after vegetation assignment to have a cleaner final point cloud !!    
            # if self.parameters['plot_radius_buffer'] > 0:
            #     if math.dist(np.mean(dbh_section[:,:2], axis=0), self.plot_centre) >  (self.parameters['plot_radius'] +.1) : 
            #         # using a 10cm margin to allow for a difference between this calculation and the one in tree_metrics (the final)
            #         # trees_cyl_array = np.delete(trees_cyl_array, this_tree_mask, axis=0) # delete after point assignment to have a cleaner final point cloud !!               
            #         continue

            ## Find multiple leaders at breast height
            branch_id_array = np.unique(dbh_section[:,self.cyl_dict['branch_id']])

            # discard some small leaders
            max_r = 0
            for branch_id in branch_id_array:                
                leader = tree[tree[:, self.cyl_dict['branch_id']] == branch_id, :]
                mean_r = np.mean(leader[:, self.cyl_dict['radius']]) 
                if mean_r > max_r : max_r = mean_r
                if (mean_r*2 < self.parameters['minimum_DBH']) or (mean_r*2.5 < max_r) :   # Note that the current max_r may not be the actual largest leader.
                    branch_id_array = np.delete(branch_id_array, branch_id_array==branch_id)  # Therefore, some small leaders may remain after this loop.
                                                                                                # These will be taken care of later.
            
            # If not a valid tree, keep it for now so we can remove its vegetation as well (in tree_metrics)
            if np.size(branch_id_array) == 0 : 
                # trees_cyl_array = np.delete(trees_cyl_array, this_tree_mask, axis=0) 
                # continue
                branch_id_array = np.unique(dbh_section[:,self.cyl_dict['branch_id']])  # restore branch_id array so we can find the highest leader
                max_m = 0
                for branch_id in branch_id_array:                
                    branch_len = np.sum(tree[:, self.cyl_dict['branch_id']] == branch_id)
                    if branch_len > max_m :
                        max_m = branch_len
                        leader = branch_id      # save this branch as the highest
                branch_id_array = leader                
        
            if np.size(branch_id_array) == 1 : # single stem
               
                tree[tree[:, self.cyl_dict['branch_id']]==branch_id_array, self.cyl_dict['main_stem']] = 1
                
                ## trace to top and mark as main stem;
                tree, _,_ = trace_skeleton(tree, branch_id_array, self.cyl_dict, mark=True)                
                trees_cyl_array[this_tree_mask,:] = deepcopy(tree)
                continue  # done with this tree

            
            
            ## check if one stem is split into two branch_ids
            elif np.size(branch_id_array) == 2 :                 
                leader = tree[tree[:,self.cyl_dict['branch_id']]==branch_id_array[1], :]                               
                base = tree[tree[:,self.cyl_dict['branch_id']]==branch_id_array[0], :] # smaller branch_id means a lower section
                base_branch_topxy = base[np.argmax(base[:,2]),:2]
                
                # check if interrupted dbh section into two vertical segments
                condition =  np.max(base[:,2]) < np.min(leader[:,2]) and \
                            math.dist(base_branch_topxy, np.median(leader[:,:2], axis=0)) < .1  # Hard-coded 10cm !!!
                if condition :                     
                    # mark both as main stem and continue with the leader (higher segment)
                    tree[tree[:, self.cyl_dict['branch_id']] == branch_id_array[1], self.cyl_dict['main_stem']] = 1
                    tree[tree[:, self.cyl_dict['branch_id']] == branch_id_array[0], self.cyl_dict['main_stem']] = 1
                    branch_id_array=branch_id_array[1]    
                else :  # discard the branch
                    tree[tree[:, self.cyl_dict['branch_id']] == branch_id_array[0], self.cyl_dict['main_stem']] = 1
                    branch_id_array=branch_id_array[0]
                
                # trace to top, mark priority and copy to original array
                tree, _,_ = trace_skeleton(tree, branch_id_array, self.cyl_dict, mark=True)
                trees_cyl_array[this_tree_mask,:] = deepcopy(tree)
                continue        # Done with this tree            

            ## multiple leaders             
            largest_radius = 0
            largest_leader = min(branch_id_array) # the lowest segment  
            for branch_id in branch_id_array :
                # check if a leader is a short branch/stump/etc
                if np.sum(tree[:, self.cyl_dict['branch_id']]==branch_id) > self.parameters['min_tree_cyls'] or \
                    max(tree[tree[:, self.cyl_dict['branch_id']] == branch_id, self.cyl_dict['height_above_dtm']]) > self.parameters['tree_stem_min_height'] :
                    # true leaders - find the largest                        
                    mean_r = np.mean(dbh_section[dbh_section[:, self.cyl_dict['branch_id']]==branch_id, self.cyl_dict['radius']])   
                    if mean_r > largest_radius:
                        largest_radius = mean_r
                        largest_leader = branch_id
                else:
                    # remove this branch from further leader processing
                    branch_id_array = branch_id_array[branch_id_array!=branch_id]                    
            # print(f'Largest leader id: {largest_leader}')    
            
            ##----- assign leader priorities
            priority = 1 
            if np.size(branch_id_array) > 0 :
                base_id = min(branch_id_array)
            
            else :  # # No valid leaders left - must be a bush                 
                # continue OR  keep for now and delete during QC : 
                base_id = largest_leader
                branch_id_array = largest_leader
            
            ##----- process each leader
            for branch_id in branch_id_array :  
                leader_mask = tree[:, self.cyl_dict['branch_id']]==branch_id

                if branch_id == base_id :    # tree_base
                    tree[leader_mask, self.cyl_dict['main_stem']] = 1   # will be traced to top at the end
                                                                        # after confirming it is not just the base under real leaders
                else: # not just the base
                    leader = tree[leader_mask,:]
                    mean_r = np.mean(leader[:, self.cyl_dict['radius']]) 
                    
                    #  check if big enough to be measured ...
                    if ((mean_r*2.5) > largest_radius) and (mean_r*2 > self.parameters['minimum_DBH']) :  
                        ## trace this leader to the top; 
                        tree, child_ids, all_childids = trace_skeleton(tree, branch_id, self.cyl_dict, mark=False)

                        # child_ids=[branch_id]                        
                        # next_branches = np.unique(tree[tree[:,self.cyl_dict['parent_branch_id']]==branch_id,self.cyl_dict['branch_id']]) # child ids                        
                        # all_childids=np.concatenate((child_ids, next_branches), axis = None)
                        # next_branches = next_branches[next_branches!=branch_id]             
                    
                        # base_branch_topz = np.max(tree[tree[:, self.cyl_dict['branch_id']]==branch_id,2])
                        # while np.any(next_branches) : 
                        #     # check if child is located lower than the parent's top
                        #     if np.min(tree[tree[:, self.cyl_dict['branch_id']]==next_branches[0], 2]) < base_branch_topz :
                        #         next_branches = next_branches[1:]
                        #         continue   

                        #     # find the largest radius child
                        #     child_id=next_branches[0]
                        #     child_radius = np.mean(tree[tree[:, self.cyl_dict['branch_id']]==next_branches[0], self.cyl_dict['radius']])
                        #     for i in range(len(next_branches)-1) :
                        #         if np.mean(tree[tree[:, self.cyl_dict['branch_id']]==next_branches[i+1], self.cyl_dict['radius']]) > child_radius :
                        #             child_id=next_branches[i+1]
                        #             child_radius = np.mean(tree[tree[:, self.cyl_dict['branch_id']]==next_branches[i+1], self.cyl_dict['radius']])
                            
                        #     child_ids.append(child_id) # add to the list of children
                        #     #                   ...do something on the child branch
                        #     # next iteration
                        #     next_branches = np.unique(tree[tree[:,self.cyl_dict['parent_branch_id']]==child_id,self.cyl_dict['branch_id']])
                        #     # next_branches = next_branches[next_branches!=child_id] # just in case ( should only happen for a tree base)
                        #     all_childids = np.concatenate((all_childids, next_branches), axis=None)

                        #--- check if tall enough to be measured 
                        leader_height = max(max(leader[:, self.cyl_dict['height_above_dtm']]), max(tree[tree[:,self.cyl_dict['branch_id']]==child_ids[-1], self.cyl_dict['height_above_dtm']]))
                        if (leader_height >= self.parameters['tree_stem_min_height']) :  

                            priority +=1            # assign stem priority
                            max_treeid +=1          # assign new tree_id
                            print(f'priority {priority}')

                            for child in child_ids :  # main skeleton of this leader 
                                tree[tree[:,self.cyl_dict['branch_id']]==child, self.cyl_dict['main_stem']] = priority
                            for child in all_childids :  # skeleton of side branches of this leader
                                tree[tree[:,self.cyl_dict['branch_id']]==child, self.cyl_dict['tree_id']] = max_treeid                                                     

            ## End of leader detection
            
            if priority == 1 : # one leader left as true leader    
                ## trace to top to assign main stem for taper
                tree, _ , _ = trace_skeleton(tree, base_id, self.cyl_dict, mark=True)

            elif priority > 2 : # at least 3 true leaders - only then we can have the tree base problem
                ## Check if the lowest leader is only the tree base 
                mask = tree[:,self.cyl_dict['branch_id']]==base_id    # get the base
                if max(tree[mask,self.cyl_dict['height_above_dtm']]) < self.parameters['tree_stem_min_height'] and \
                    (base_id in np.unique(tree[tree[:,self.cyl_dict['main_stem']] == 2, self.cyl_dict['parent_branch_id']])) and \
                    (base_id in np.unique(tree[tree[:,self.cyl_dict['main_stem']] == 3, self.cyl_dict['parent_branch_id']])) : 
                    
                    # too short and at least 2 leaders are its children => a tree base
                    tree[mask, self.cyl_dict['main_stem']] = 0   # set main stem to zero to be excluded from taper measurements
                    tree[mask, self.cyl_dict['tree_id']] = max_treeid  # set tree id to the last leader tree_id

                    # change the old treeId to the new for all remaining branches/sections
                    tree[tree[:, self.cyl_dict['tree_id']] == tree_id, self.cyl_dict['tree_id']] = max_treeid


            #  Set the tree_id of any leftover branches to their nearest leader
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(tree[tree[:,self.cyl_dict['main_stem']]>0, :3])  # find the nearest leader in 3d
            for branch in np.unique(tree[tree[:,self.cyl_dict['main_stem']] == 0, self.cyl_dict['branch_id']]) :
                mask = tree[:, self.cyl_dict['branch_id']] == branch
                lowest_point = np.argmin(tree[mask, 2])
                _, nb = nn.kneighbors(np.reshape(tree[mask,:3][lowest_point],[1,3]), n_neighbors=1)                            
                tree[mask, self.cyl_dict['tree_id']] = tree[tree[:,self.cyl_dict['main_stem']]>0, self.cyl_dict['tree_id']][nb]
                    

            ## Sanity check for multiple leaders
            if not np.any(tree[:, self.cyl_dict['main_stem']] > 0):
                tree[:, self.cyl_dict['main_stem']] = 1

            ## copy the modified tree back to the full array
            trees_cyl_array[this_tree_mask,:] = deepcopy(tree)
        
        # TODO move next to a function :
        ##----- smoothing the main stems of all trees - leaders have unique ids now
        for tree_id in np.unique(trees_cyl_array[:, self.cyl_dict['tree_id']]):       
            # print(tree_id)
            
            # # if a special case occured and there isn't any main stem, mark the whole tree as main stem
            # if np.all(trees_cyl_array[trees_cyl_array[:,self.cyl_dict['tree_id']]==tree_id, self.cyl_dict['main_stem']]==0) :
            #     trees_cyl_array[trees_cyl_array[:,self.cyl_dict['tree_id']]==tree_id, self.cyl_dict['main_stem']] = 1

            ## use median filter to remove outliers from the main stem cylinders
            this_leader_mask = np.logical_and(trees_cyl_array[:,self.cyl_dict['tree_id']]==tree_id,
                                              trees_cyl_array[:,self.cyl_dict['main_stem']]>0)
            
            tree_stem = deepcopy(trees_cyl_array[this_leader_mask,:])

            radii = tree_stem[:,self.cyl_dict['radius']]
            sort_mask = np.argsort(tree_stem[:,2])       # Ascending sort by z coordinate
            # difs = radii[sort_mask][1:]-radii[sort_mask][:-1]
            
            ## clean single outliers and smooth the radii of the skeleton
            # smoothed = scipy.ndimage.filters.convolve() np.full((3), 1.0/3)               
            # smoothed = ndimage.median_filter(radii[sort_mask], size=5, mode='mirror', origin=-2) # negative origin practically ignores the start point
            smoothed = ndimage.median_filter(radii[sort_mask], size=5, mode='reflect') 
            
            ## copy the cleaned radii back to cylinder array
            tree_stem[sort_mask, self.cyl_dict['radius']] = np.around(smoothed,3)
            
            x = ndimage.median_filter(tree_stem[sort_mask,0], size=7, mode='reflect') 
            y = ndimage.median_filter(tree_stem[sort_mask,1], size=7, mode='reflect')
            tree_stem[sort_mask, 0] = x
            tree_stem[sort_mask, 1] = y

            trees_cyl_array[this_leader_mask] = deepcopy(tree_stem)
                         
            ## interpolate from the lowest cylinder to the ground
            lowest_cyl = tree_stem[sort_mask,:][0]
            z_tree_base = lowest_cyl[2] - lowest_cyl[self.cyl_dict['height_above_dtm']] 
            # make a new cylinder at ground level
            to_ground = deepcopy(lowest_cyl)    # copy attributes from lowest cylinder
            to_ground[2] = z_tree_base + .2     # set z to ground level + 20cm terrain&ground_vegetation margin
            to_ground[self.cyl_dict['height_above_dtm']] = .2 
            ground_section = self.interpolate_cyl(lowest_cyl, to_ground,resolution=self.slice_increment)
            # add to cylinder array
            trees_cyl_array = np.vstack((trees_cyl_array, ground_section))

        # save_file(self.output_dir + 'cleaned_stems.laz', trees_cyl_array, headers_of_interest=list(self.cyl_dict), offsets=self.offsets)
        return trees_cyl_array


    # # split execution here - Aglika: 
    # @jit(nopython=True)
    def cylinder_assignment(self):        
        
        full_cyl_array, _ = load_file(self.output_dir + 'ransac_cyls.laz', headers_of_interest=list(self.cyl_dict)[:9])
        print("Deleting cyls with CCI less than:", self.parameters['minimum_CCI'])
        full_cyl_array = full_cyl_array[np.logical_or(full_cyl_array[:, self.cyl_dict['CCI']] >= self.parameters['minimum_CCI'],
                                                      full_cyl_array[:, self.cyl_dict['CCI']] == 0)]
        # delete oversized circles
        full_cyl_array = full_cyl_array[full_cyl_array[:,self.cyl_dict['radius']]*2 < self.parameters['maximum_DBH']] 
        # delete small circles
        full_cyl_array = full_cyl_array[full_cyl_array[:,self.cyl_dict['radius']]*2 > .04] 
        # delete cylinders with big lean
        # full_cyl_array = full_cyl_array[full_cyl_array[:,self.cyl_dict['nz']] > .7] # already done 

        if 1:
            print("Making initial_cyl visualisation...")
            j = 0
            initial_cyl_vis = []
            max_j = np.shape(full_cyl_array)[0]
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, full_cyl_array):
                    initial_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            initial_cyl_vis = np.vstack(initial_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')
            print("\nSaving cylinder visualisation...")
            # save_file(self.output_dir + 'ransac_cyl_vis.laz', initial_cyl_vis, headers_of_interest=list(self.cyl_dict)[:initial_cyl_vis.shape[1]], offsets=self.offsets) # --- Joo edited, for saving, need to be uncommented
        
        

        full_cyl_array = np.hstack((full_cyl_array, np.zeros((full_cyl_array.shape[0],2)))) # adding more dimensions for the last cyl_dict items

        # TODO TODO  - losing whole trees here - fixed 
        # print("Sorting Cylinders...")       # Assigning cylinders to trees
        # full_cyl_array = self.cylinder_sorting(full_cyl_array,
        #                                        angle_tolerance=self.parameters['sorting_angle_tolerance'],
        #                                        search_angle=self.parameters['sorting_search_angle'],
        #                                        distance_tolerance=self.parameters['sorting_search_radius'])        
        # save_file(self.output_dir + 'assigned_cyls.laz', full_cyl_array, headers_of_interest=list(self.cyl_dict), offsets=self.offsets)

        ############
        # TODO     - losing height   ?? - fixed: 2-step segment connection
        print('Connecting segments into trees...')
        
        sorted_cyl_array = np.zeros((0, full_cyl_array.shape[1]))   
        max_search_radius = self.parameters['max_search_radius']        
        max_search_angle = self.parameters['max_search_angle']
        max_branch_id = np.max(full_cyl_array[:, self.cyl_dict['branch_id']])

        new_id = 0
        high_cyl_id = 0
        
        for branch_id in np.unique(full_cyl_array[:, self.cyl_dict['branch_id']]):          
            if int(branch_id) % 10 == 0:
                print('\r','Branch ID', int(branch_id), '/', int(max_branch_id),end='')

            if branch_id == 119 :   # for debugging
                print(branch_id)
            # print(high_cyl_id)

            # get the current tree segment and its lowest point
            tree_mask = (full_cyl_array[:, self.cyl_dict['branch_id']] == int(branch_id))
            
            #  delete segment if consisting of less than num_neghbours number of points
            if np.sum(tree_mask) < self.num_neighbours :
                full_cyl_array = full_cyl_array[~tree_mask]
                continue

            tree = full_cyl_array[tree_mask]   
            valid_mask=[]

            ## segment direction vector
            segment_main_direction = np.median(tree[:,3:6], axis=0)
            # if segment_main_direction[2] < .7

            lowest_point = tree[np.argmin(tree[:, 2]),:]
            second2lowest_point = tree[np.argmin(tree[tree[:,2]>lowest_point[2], 2]),:] # use to reduce the effect of noisy end point
            # lowest_point = np.mean([lowest_point[:3], second2lowest_point[:3]], axis=0)

            # split the space of remaining segments
            full_cyl_array_copy = deepcopy(full_cyl_array[np.logical_not(tree_mask)])
            tree_kdtree = spatial.cKDTree(full_cyl_array_copy[:, :3], leafsize=1000)

            # first do wide angle search at short distance  (1)
            neighbours = full_cyl_array_copy[tree_kdtree.query_ball_point(lowest_point[:3], r=.5), :] # TODO hard-coded search radius 1/2m
            neighbours = neighbours[neighbours[:,2] < (lowest_point[2]-self.parameters['slice_increment'])]   # keep those that are below
            if neighbours.size > 0 : 
                verticals = lowest_point[:3] - neighbours[:, :3]
                angles = vectors_inner_angle(verticals, lowest_point[3:6]) # search in the direction of the lowest point
                valid_mask = (angles < 90)

            # if nothing is found by the closest search try long distance search  (2)
            if not np.any(valid_mask) : 

                # find cylinder points within a radius of max_search_radius
                neighbours = full_cyl_array_copy[tree_kdtree.query_ball_point(lowest_point[:3], r=max_search_radius),:]
                # neighbours = neighbours[(neighbours[:, None] != lowest_point).any(-1).flatten(),:]   # remove the lowest point from the neighbours array
                neighbours = neighbours[neighbours[:,2] < (lowest_point[2]-self.parameters['slice_increment'])]   # keep those that are below
                if neighbours.shape[0] > 0 :                
                    verticals = np.mean([lowest_point[:3], second2lowest_point[:3]], axis=0) - neighbours[:, :3]
                    ## find the angle between the verticals and the sky
                    angles = vectors_inner_angle(verticals, np.array([0,0,1]))
                    ## search within cone in the direction of the segment
                    # angles = vectors_inner_angle(verticals, segment_main_direction)

                    valid_mask = (angles <= max_search_angle)

            if not np.any(valid_mask) : # no valid cylinders below this one, it is a tree base, a distant segment, or noisy segment
                
                # if neighbours.shape[0] > 0 :
                # ## check if noisy: using just the lowest point is very singular and prone to noise
                # # dist = np.linalg.norm(lowest_point[:3]-neighbours[:,:3], axis=1)
                #     dist = np.min(lowest_point[2]-neighbours[:,2])
                #     if dist < (self.parameters['gap_connect']*2) : # if there is a close-by point, leave for the 2nd round
                #         high_cyl_id -=1 
                #         tree[:, self.cyl_dict['tree_id']] = high_cyl_id
                #         sorted_cyl_array = np.vstack((sorted_cyl_array, tree)) # to be used in 2nd round
                #         # propagate the new tree id to the original array of cylinders so that remaining segments can be connected to it
                #         full_cyl_array[full_cyl_array[:, self.cyl_dict['branch_id']] == int(branch_id), self.cyl_dict['tree_id']] = high_cyl_id 
                #         continue
                
                # A tree base - check if lowest point near ground
                lowest_point_h = lowest_point[2] - griddata((self.DTM[:, 0], self.DTM[:, 1]), self.DTM[:, 2],
                                                lowest_point[0:2], method='linear', fill_value=np.median(self.DTM[:, 2]))
                if lowest_point_h < self.parameters['tree_base_cutoff_height']:
                    # tree id assigned here will be in the final set of trees                    
                    new_id +=1
                    tree[:, self.cyl_dict['tree_id']] = new_id
                    tree[:, self.cyl_dict['parent_branch_id']] = branch_id
                    sorted_cyl_array = np.vstack((sorted_cyl_array, tree))
                    # propagate the new tree id to the original array of cylinders to be used in the remaining iterations
                    full_cyl_array[full_cyl_array[:, self.cyl_dict['branch_id']] == int(branch_id), self.cyl_dict['tree_id']] = new_id
                
                else: # distant high segments
                    # keep for the second round if not leaning too much TODO replace with angle_to_sky
                    if segment_main_direction[2] > .5 :
                    # if np.max(tree[:,2])-lowest_point[2] > self.parameters['gap_connect']: # don't consider if the whole section is near horizontal

                        high_cyl_id -=1 
                        tree[:, self.cyl_dict['tree_id']] = high_cyl_id
                        sorted_cyl_array = np.vstack((sorted_cyl_array, tree))
                        # propagate the new tree id to the original array of cylinders so that remaining segments can be connected to it
                        full_cyl_array[full_cyl_array[:, self.cyl_dict['branch_id']] == int(branch_id), self.cyl_dict['tree_id']] = high_cyl_id 

            else : # valid neighbours -  connect this segment to a lower one
                
                # # connect to the point at the smallest vertical angle from the lowest segment point
                # best_parent_point = neighbours[np.argmin(angles[valid_mask])]   # not the best when current segment is a branch
                
                if tree.shape[0] > (self.parameters['num_neighbours']*2) :
                    ## connect to the closest neighbour WITHIN the cone
                    neighbours = neighbours[valid_mask,:]           
                    dist = np.linalg.norm(lowest_point[:3]-neighbours[:,:3], axis=1)
                    best_parent_point = neighbours[np.argmin(dist)]
                else:
                    ## OW connect to The closest neighbour - better for branches
                    dist = np.linalg.norm(lowest_point[:3]-neighbours[:,:3], axis=1)
                    if min(dist) < self.parameters['stem_sorting_range']:
                        best_parent_point = neighbours[np.argmin(dist)]
                    else:
                        # discard this segment - either vegetation labeled as stem, or distant branch that cannot be connected reliably
                        full_cyl_array = full_cyl_array[~tree_mask]
                        continue


                # TODO - connect branches separately from main stems - TEST
                # if (np.mean(tree[:, self.cyl_dict['radius']]) > 0.06) and (lowest_point[self.cyl_dict['nz']] > .9) : 
                #     # np.degrees(np.arccos(.9)) = 25.8 degrees maximum lean
                #     best_parent_point = neighbours[np.argmin(angles[valid_mask])] # not the best when current segment is a branch
                #     ## connect to the point of the smallest angle - weighted by the distance
                #     # best_parent_point = neighbours[np.argmin(angles*(dist**2))]                                                                         
                # else :
                #     ## connect to the closest point in 3d - for bends and forks
                #     best_parent_point = neighbours[np.argmin(dist)]

                tree = np.vstack((tree, self.interpolate_cyl(lowest_point, best_parent_point,  
                                                                resolution=self.slice_increment)))
                tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                tree[:, self.cyl_dict['parent_branch_id']] = best_parent_point[self.cyl_dict['branch_id']]

                # put this segment back in the full array
                sorted_cyl_array = np.vstack((sorted_cyl_array, tree))
                # propagate the new tree id to the original array of cylinders to be used in the remaining iterations
                full_cyl_array[full_cyl_array[:, self.cyl_dict['branch_id']] == int(branch_id), self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]

        del full_cyl_array, full_cyl_array_copy

        # connect distant high segments (tree_id < 0)

        # # this is good for redwoods or big trees with small relative lean
        # max_search_radius *=2
        # max_search_angle /=2

        # adjust the max_search_radius and max_search_angle to connect heavy bends and forks - eucalypts, etc.
        max_search_radius = self.parameters['gap_connect']*2       # TODO: remove the hardcoding - tree species parameter
        max_search_angle = 120       # 1*np.sin(np.radians(20)) = 0.342m radius of the search cone
        
        mask = sorted_cyl_array[:, self.cyl_dict['tree_id']] < 0  # 
        while np.any(mask):
            for tree_id in np.unique(sorted_cyl_array[mask, self.cyl_dict['tree_id']]) :

                # use a search cone following the direction of the segment
                tree_mask = sorted_cyl_array[:, self.cyl_dict['tree_id']] == int(tree_id)
                tree = sorted_cyl_array[tree_mask]   
                lowest_point = tree[np.argmin(tree[:, 2]),:]           
                
                sorted_cyl_array = sorted_cyl_array[np.logical_not(tree_mask)]  # temporarily remove the current tree from the results

                tree_kdtree = spatial.cKDTree(sorted_cyl_array[:, :3], leafsize=1000)
                neighbours = sorted_cyl_array[tree_kdtree.query_ball_point(lowest_point[:3], r=max_search_radius), :] # TODO be smarter with the search radius (max_dbh+lean??)
                neighbours = neighbours[neighbours[:,2] < (lowest_point[2]-self.parameters['slice_increment'])]   # keep those that are below

                if neighbours.size > 0 : # connect segment only if there is a close-by tree - 1m, hardcoded
                    
                    # # TODO - use the normals of the segments to find best parent point. Then, Search radius can be bigger
                    verticals = lowest_point[:3] - neighbours[:, :3]

                    # find the angle between the verticals and the sky
                    # angles = vectors_inner_angle(verticals, np.array([0,0,1]))
                    angles = vectors_inner_angle(verticals, lowest_point[3:6])                    
                    valid_mask = angles < max_search_angle
                
                    if np.any(valid_mask) :
                        angles= angles[valid_mask]
                        neighbours = neighbours[valid_mask]

                        # best_parent_point = neighbours[np.argmin(angles)]
                        # best_parent_point = neighbours[np.argmin(np.linalg.norm(lowest_point[:3]-neighbours[:,:3], axis=1))] 

                        dist = np.linalg.norm(lowest_point[:3]-neighbours[:,:3], axis=1)
                        best_parent_point = neighbours[np.argmin(dist)]                    

                        tree = np.vstack((tree, self.interpolate_cyl(lowest_point, best_parent_point, resolution=self.slice_increment)))
                        tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                        tree[:, self.cyl_dict['parent_branch_id']] = best_parent_point[self.cyl_dict['branch_id']]             
                        sorted_cyl_array = np.vstack((sorted_cyl_array, tree))  # a distant segment, put it back into the results

                # else: 
                #     high_cyl_id -=1
                #     tree[:, self.cyl_dict['tree_id']] = high_cyl_id
                #     sorted_cyl_array = np.vstack((sorted_cyl_array, tree))  # put it back into the results
            
            mask = sorted_cyl_array[:, self.cyl_dict['tree_id']] < 0
        
        if sorted_cyl_array.shape[0]==0:
            # sys.exit('Cannnot measure any tree. Exiting..')
            raise ValueError('Cannot connect into tree skeleton. Possibly a scan of very young trees!')
        
        headers = list(self.cyl_dict)[:sorted_cyl_array.shape[1]]
        # save_file(self.output_dir + 'initial_trees_cyls.laz', sorted_cyl_array, headers_of_interest=headers, offsets=self.offsets) # --- Joo edited, for saving, need to be uncommented
        print("\n--------Saving cylinder data...")
        pd.DataFrame(sorted_cyl_array, columns=headers).to_csv(self.output_dir + 'initial_trees_cyls.csv', index=False, sep=',')

        # get the heights above ground
        sorted_cyl_array = np.hstack((sorted_cyl_array, np.zeros((sorted_cyl_array.shape[0],1))))
        sorted_cyl_array[:,self.cyl_dict['height_above_dtm']] = get_heights_above_DTM(sorted_cyl_array[:,:3], self.DTM)

        trees_cyl_array = self.stem_smoothing(sorted_cyl_array) # this is the moment for error
        if trees_cyl_array.shape[0]==0 :
            # sys.exit('Cannnot measure any tree. Exiting..')
            raise ValueError('Stem smoothing did not ouput any trees. Young trees or algorithm failure!')
        save_file(self.output_dir + 'clean_trees_skeletons.laz', trees_cyl_array, headers_of_interest=list(self.cyl_dict), offsets=self.offsets)
        print("\n--------Saving clean cylinder data...")
        pd.DataFrame(trees_cyl_array, columns=list(self.cyl_dict)).to_csv(self.output_dir + 'clean_trees_cyls.csv', index=False, sep=',')        
   



    # not relevant at the moment. Will need modification when put back into the workflow!!
    def connect_branches(self) :

        sorted_cyl_array, headers = load_file(self.output_dir + 'clean_trees_skeletons.laz', headers_of_interest=list(self.cyl_dict))
        
        # add Height
        sorted_cyl_array = np.hstack((sorted_cyl_array, np.zeros((sorted_cyl_array.shape[0],1))))
        sorted_cyl_array[:,-1] = get_heights_above_DTM(sorted_cyl_array[:,:3], self.DTM)
        # add Angle to horizon
        v1 = sorted_cyl_array[:, 3:6]
        v2 = np.vstack((sorted_cyl_array[:, 3],   #TODO
                        sorted_cyl_array[:, 4],
                        np.zeros((sorted_cyl_array.shape[0]))+0.001)).T  # to avoid division by zero
        sorted_cyl_array = np.hstack((sorted_cyl_array, np.zeros((sorted_cyl_array.shape[0],1))))
        sorted_cyl_array[:, self.cyl_dict['segment_angle_to_horiz']] = self.compute_angle(v1, v2)

        # # clean branches and tiny trees
        mask = sorted_cyl_array[:,self.cyl_dict['radius']] > self.parameters['minimum_DBH']/2 # hardcoded
        sorted_cyl_array = sorted_cyl_array[mask]
        mask = sorted_cyl_array[:,self.cyl_dict['radius']] < self.parameters['maximum_DBH']/2 # hardcoded
        sorted_cyl_array = sorted_cyl_array[mask]
        
        # delete points based on their lean - mostly branches
        if 'segment_angle_to_horiz' in self.cyl_dict and np.any(sorted_cyl_array[:,self.cyl_dict['segment_angle_to_horiz']]):
                sorted_cyl_array = sorted_cyl_array[sorted_cyl_array[:,self.cyl_dict['segment_angle_to_horiz']]>60]     #hardcoded

        # Interpolate within remaining points in tree
        print("Cylinder interpolation...")
        tree_list = []
        interpolated_full_cyl_array = np.zeros((0, sorted_cyl_array.shape[1]))        
        
        max_tree_id = np.max(sorted_cyl_array[:, self.cyl_dict['tree_id']])
        # for tree_id in np.unique(sorted_cyl_array[:, self.cyl_dict['tree_id']]):
        while 0:  # !! Connecting branches 
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), '/', int(max_tree_id))
            current_tree = sorted_cyl_array[sorted_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]
            if current_tree.shape[0] >= self.parameters['min_tree_cyls']:
                interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, current_tree))
                
                _, individual_branches_indices = np.unique(current_tree[:, self.cyl_dict['branch_id']], return_index=True)
                tree_list.append(nx.Graph())
                
                for branch in current_tree[individual_branches_indices]:
                    branch_id = branch[self.cyl_dict['branch_id']]
                    parent_branch_id = branch[self.cyl_dict['parent_branch_id']]
                    
                    tree_list[-1].add_edge(int(parent_branch_id), int(branch_id))
                    current_branch = current_tree[current_tree[:, self.cyl_dict['branch_id']] == branch_id]
                    parent_branch = current_tree[current_tree[:, self.cyl_dict['branch_id']] == parent_branch_id]

                    current_branch_copy = deepcopy(current_branch[np.argsort(current_branch[:, 2])])
                    while current_branch_copy.shape[0] > 1:
                        lowest_point = current_branch_copy[0]
                        current_branch_copy = current_branch_copy[1:]
                        # find nearest point. if nearest point > increment size, interpolate.
                        distances = np.abs(np.linalg.norm(current_branch_copy[:, :3] - lowest_point[:3], axis=1))
                        if distances[distances > 0].shape[0] > 0:
                            if np.min(distances[distances > 0]) > self.slice_increment:
                                interp_to_point = current_branch_copy[distances > 0]
                                if interp_to_point.shape[0] > 0:
                                    interp_to_point = interp_to_point[np.argmin(distances[distances > 0])]

                                # Interpolates a single branch.
                                if interp_to_point.shape[0] > 0:
                                    interpolated_cyls = self.interpolate_cyl(interp_to_point, lowest_point,
                                                                             resolution=self.slice_increment)
                                    current_branch = np.vstack((current_branch, interpolated_cyls))
                                    interpolated_full_cyl_array = np.vstack(
                                            (interpolated_full_cyl_array, interpolated_cyls))

                    if parent_branch.shape[0] > 0:
                        parent_centre = np.mean(parent_branch[:, :3])
                        closest_point_index = np.argmin(np.linalg.norm(parent_centre - current_branch[:, :3]))
                        closest_point_of_current_branch = current_branch[closest_point_index]
                        kdtree = spatial.cKDTree(parent_branch[:, :3])
                        parent_points_in_range = parent_branch[
                            kdtree.query_ball_point(closest_point_of_current_branch[:3], r=self.parameters['max_search_radius'])]
                        lowest_point_of_current_branch = current_branch[np.argmin(current_branch[:, 2])]

                        same_points_mask = np.linalg.norm(lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3], axis=1).T == 0
                        parent_points_in_range = parent_points_in_range[~same_points_mask]
                        if parent_points_in_range.shape[0] > 0:

                            angles = MeasureTree.compute_angle(lowest_point_of_current_branch[3:6],
                                                               lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3])
                            angles = angles[angles <= self.parameters['max_search_angle']]
                            if angles.shape[0] > 0:
                                best_parent_point = parent_points_in_range[np.argmin(angles)]
                                # Interpolates from lowest point of current branch to smallest angle parent point.
                                interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, 
                                    self.interpolate_cyl(lowest_point_of_current_branch, best_parent_point,resolution=self.slice_increment)))
               
                # Aglika - do not need measurements below taper_measurement_height_min
                #           In any case they will be just a copy of the lowest fitted circle
                #
                # current_tree = get_heights_above_DTM(current_tree, self.DTM)
                # lowest_10_measured_stem_points = deepcopy(current_tree[np.argsort(current_tree[:, -1])][:10])
                # lowest_measured_tree_point = np.median(lowest_10_measured_stem_points, axis=0)
                # tree_base_point = deepcopy(current_tree[np.argmin(current_tree[:, self.cyl_dict['height_above_dtm']])])
                # tree_base_point[2] = tree_base_point[2] - tree_base_point[self.cyl_dict['height_above_dtm']]              
                # interpolated_to_ground = self.interpolate_cyl(lowest_measured_tree_point, tree_base_point,       
                #                                               resolution=self.slice_increment)
                # interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, interpolated_to_ground))

        interpolated_full_cyl_array = sorted_cyl_array
        interpolated_full_cyl_array[:,self.cyl_dict['height_above_dtm']] = get_heights_above_DTM(interpolated_full_cyl_array[:,:3], self.DTM)
        
        # v1 = interpolated_full_cyl_array[:, 3:6]
        # v2 = np.vstack((interpolated_full_cyl_array[:, 3],   #TODO
        #                 interpolated_full_cyl_array[:, 4],
        #                 np.zeros((interpolated_full_cyl_array.shape[0]))+0.001)).T      # to avoid division by zero
        # interpolated_full_cyl_array = np.hstack((interpolated_full_cyl_array, np.zeros((interpolated_full_cyl_array.shape[0],1))))        
        # interpolated_full_cyl_array[:, self.cyl_dict['segment_angle_to_horiz']] = self.compute_angle(v1, v2)

        # save_file(self.output_dir + 'interpolated_full_cyl_array.laz', interpolated_full_cyl_array, 
        #             headers_of_interest=list(self.cyl_dict)[:interpolated_full_cyl_array.shape[1]], offsets=self.offsets)

        print(interpolated_full_cyl_array.shape)
        if 1:
            print("Making interp_cyl visualisation...")
            j = 0
            interpolated_cyl_vis = []
            max_j = np.shape(interpolated_full_cyl_array)[0]
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, interpolated_full_cyl_array):
                    interpolated_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            interpolated_cyl_vis = np.vstack(interpolated_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')

            print("\nSaving cylinder visualisation...")
            save_file(self.output_dir + 'interpolated_cyl_vis.laz', interpolated_cyl_vis,
                      headers_of_interest=list(self.cyl_dict)[:interpolated_cyl_vis.shape[1]], offsets=self.offsets)


        # print("Cylinder Outlier Removal...")
        input_data = []
        i = 0
        tree_id_list = np.unique(interpolated_full_cyl_array[:, self.cyl_dict['tree_id']])
        
        if tree_id_list.shape[0] > 0:
            max_tree_id = int(np.max(tree_id_list))
            
            for tree_id in tree_id_list: # Formatting cylinders for the cylinder_cleaning_multithreaded function
                if tree_id % 10 == 0:
                    print('\r', tree_id, '/', max_tree_id, end='')
                i += 1
               
                single_tree = interpolated_full_cyl_array[interpolated_full_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]

                # # delete branches
                if single_tree.size > self.parameters['min_tree_cyls'] :
                #     single_tree = np.delete(single_tree, single_tree[:,self.cyl_dict['radius']]<0.051, axis=0)
                # # delete more branches
                # if single_tree.shape[0]:
                #     if 'segment_angle_to_horiz' in self.cyl_dict and np.any(single_tree[:,self.cyl_dict['segment_angle_to_horiz']]):
                #         single_tree = np.delete(single_tree, single_tree[:,self.cyl_dict['segment_angle_to_horiz']]<60, axis=0)
    
                # if (single_tree.size > self.parameters['min_tree_cyls'] and 
                #     (np.max(single_tree[:,self.cyl_dict['height_above_dtm']]) > self.parameters['tree_stem_min_height'])) :  # discard undergrowth and stumps
                #     # single_tree = self.fix_outliers(single_tree)
                    input_data.append([single_tree, self.parameters['cleaned_measurement_radius'], self.cyl_dict])

            print('\r', max_tree_id, '/', max_tree_id, end='')
            print('\nDone\n')

            print("Starting multithreaded cylinder smoothing for tree ID ", tree_id, "...")
            j = 0
            max_j = len(input_data)

            cleaned_cyls_list = []
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(MeasureTree.cylinder_cleaning_multithreaded, input_data):

                    cleaned_cyls_list.append(i)
                    if j % 11 == 0:                             
                        print('\r', j, '/', max_j, end='')
                    j += 1
            cleaned_cyls = np.vstack(cleaned_cyls_list)

            del cleaned_cyls_list
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')

            cleaned_cyls = np.unique(cleaned_cyls, axis=0)

            headers = list(self.cyl_dict)[:cleaned_cyls.shape[1]]
            save_file(self.output_dir + 'cleaned_cyls.laz', cleaned_cyls, headers_of_interest=headers, offsets=self.offsets)
            pd.DataFrame(cleaned_cyls, columns=headers).to_csv(self.output_dir + 'cleaned_cyls.csv', index=False)



    def tree_metrics(self):

        if not os.path.isdir(f'{self.workdir}taper'): os.makedirs(f'{self.workdir}taper')
        # dbh_height = np.float(self.plot_summary['DBH_height'])
        dbh_height = float(self.parameters['dbh_height'])

        self.text_point_cloud = np.zeros((0, 3)) 
        tree_data = np.zeros((0, len(self.tree_data_dict)))

        cleaned_cyls, _ = load_file(self.output_dir + 'clean_trees_skeletons.laz', headers_of_interest=list(self.cyl_dict)) 
        if cleaned_cyls.shape[1] < self.cyl_dict['height_above_dtm']+1 :
            cleaned_cyls = np.hstack((cleaned_cyls, np.zeros((cleaned_cyls.shape[0],1))))
            cleaned_cyls[:,-1] = get_heights_above_DTM(cleaned_cyls, self.DTM)


        # offset tree_ids so ordered ids can be assigned from 1 to n
        cleaned_cyls[:, self.cyl_dict['tree_id']] +=10000

        ## subsample point cloud for faster processing and smaller output files
        # Raw diameter measurements are done. Height will not be affected by subsampling
        # self.stem_points = subsample_point_cloud(self.stem_points, 0.01, self.num_procs)  - This function needs a fix TODO TODO

        # Back to point cloud
        tree_id_col = self.stem_dict['tree_id']

        if os.path.isfile(self.output_dir + 'temp_vegetation.laz'):
            temp_vegetation,_= load_file(self.output_dir + 'temp_vegetation.laz', headers_of_interest=list(self.stem_dict))
            extra_dims = len(self.stem_dict) - temp_vegetation.shape[1]
            temp_vegetation = np.hstack((temp_vegetation, np.zeros((temp_vegetation.shape[0],extra_dims))))
            temp_vegetation[:,-2] = get_heights_above_DTM(temp_vegetation, self.DTM)
            temp_vegetation = temp_vegetation[temp_vegetation[:,-1] > max(self.parameters['tree_stem_min_height'],
                                                                        self.parameters['tree_base_cutoff_height']), :]
            temp_vegetation[:,-3] = self.parameters['vegetation_class'] # set label to vegetation
            self.vegetation_points = np.vstack((self.vegetation_points, temp_vegetation))


        print("Stem assignment...")  
        # Aglika -  Simple 2d nearest neighbours vegetation sorting
        #
        # TODO - use gaps along the z axis of the vegetation points to calculate tree height
        #        2d mapping gets the vegetation points from the neighbouring trees\

        # # assign stem points to trees using the IDs in cleaned_cyls in 3D
        kdtree = spatial.KDTree(cleaned_cyls[:, :3], leafsize=1000)  # TODO - use 3d points when searching for neighbours
        results = kdtree.query(self.stem_points[:, :3], k=1)
        mask = (results[0] <= self.parameters['stem_sorting_range'])   # discard points that are too far from any tree - keep for later
        unsorted_stems = self.stem_points[~mask]                     # keep unassigned points for the final las output
        self.stem_points = self.stem_points[mask]
        results = results[1][mask]
        self.stem_points[:, tree_id_col] = cleaned_cyls[results, self.cyl_dict['tree_id']] # assign tree_id to stem points 

        ## for unsorted stems use a 2d assignment below

        print("Vegetation assignment...")  
        # assign vegetation points to trees using the IDs in cleaned_cyls  in 3D 
        results = kdtree.query(self.vegetation_points[:, :3], k=1)
        mask = results[0] <= self.parameters['veg_sorting_range']
        unsorted_vegetation = self.vegetation_points[~mask]
        self.vegetation_points = self.vegetation_points[mask]
        results = results[1][mask]
        self.vegetation_points[:, tree_id_col] = cleaned_cyls[results, self.cyl_dict['tree_id']]

        ## 2d assignement of remaining stems 
        cyl_range = sorted(cleaned_cyls[:,2])
        
        # TODO - add the highest stem points to the cleaned_cyls to get better veg assignment for leaning trees
        # t = np.percentile(cleaned_cyls[:, 2], 50) # for lower Complexity use the already sorted array instead
        t = cyl_range[int(len(cyl_range)*(50/100))]
        top_cyls = cleaned_cyls[cleaned_cyls[:,2]>t]
        kdtree = spatial.KDTree(top_cyls[:, :2], leafsize=1000) 
        results = kdtree.query(unsorted_stems[:, :2], k=1)
        mask = results[0] <= self.parameters['stem_sorting_range']   # discard points that are too far from any tree
        results = results[1][mask]
        unsorted_stems[mask, tree_id_col] = top_cyls[results, self.cyl_dict['tree_id']] # copy tree_id from cylinders to veg points
        self.stem_points = np.vstack((self.stem_points, unsorted_stems[mask])) # move the newly assigned points to self
        unsorted_stems = np.delete(unsorted_stems, mask, axis=0) # delete these from the unsorted array


        t1=len(cyl_range)
        for i,p_level in enumerate([90,80,70]) :  #[90,80,70]

            t2 = cyl_range[int(len(cyl_range)*(p_level/100))]
            top_cyls = cleaned_cyls[np.logical_and(cleaned_cyls[:,2] <= t1, cleaned_cyls[:,2] > t2)]

            kdtree = spatial.KDTree(top_cyls[:,:2], leafsize=1000)
            results = kdtree.query(unsorted_vegetation[:, :2], k=1)
            mask = results[0] <= (self.parameters['veg_sorting_range'] - (.5*i))
            results = results[1][mask]  # filter results to keep only valid assignements
            unsorted_vegetation[mask, tree_id_col] = top_cyls[results, self.cyl_dict['tree_id']] # copy tree_id from cylinders to veg points
            self.vegetation_points = np.vstack((self.vegetation_points, unsorted_vegetation[mask])) # move the newly assigned points to self
            unsorted_vegetation = np.delete(unsorted_vegetation, mask, axis=0) # delete these from the unsorted array

            t1=t2 # get ready for the next iteration

            if unsorted_vegetation.shape[0]<10000: break # stop if most vegetation points have been assigned
        
        ## 2d assignement of remaining vegetation
        top_cyls = cleaned_cyls[cleaned_cyls[:,2]<t2]
        kdtree = spatial.KDTree(top_cyls[:, :2], leafsize=1000) 
        results = kdtree.query(unsorted_vegetation[:, :2], k=1)
        mask = results[0] <= self.parameters['veg_sorting_range']   # discard points that are too far from any tree
        results = results[1][mask]
        unsorted_vegetation[mask, tree_id_col] = top_cyls[results, self.cyl_dict['tree_id']] # copy tree_id from cylinders to veg points
        self.vegetation_points = np.vstack((self.vegetation_points, unsorted_vegetation[mask])) # move the newly assigned points to self
        unsorted_vegetation = np.delete(unsorted_vegetation, mask, axis=0) # delete these from the unsorted array

        # taper range and bins            
        taper_height_max = self.parameters['taper_measurement_height_max']
        taper_height_min = self.parameters['taper_measurement_height_min']
        taper_height_increment = self.parameters['taper_measurement_height_increment']
        
        if np.max(cleaned_cyls[:, self.cyl_dict['height_above_dtm']]) < taper_height_max :
            taper_height_max = np.max(cleaned_cyls[:, self.cyl_dict['height_above_dtm']])

        self.taper_measurement_heights = np.arange(taper_height_min*10, taper_height_max*10,taper_height_increment*10) /10   # Aglika - made flexible bin borders  NEEDS a fix - line 1104
        # self.taper_measurement_heights = np.arange(np.floor(taper_height_min*taper_meas_height_increment)/taper_meas_height_increment,
        #                                            (np.floor(taper_meas_height_max*taper_meas_height_increment)/taper_meas_height_increment)+taper_meas_height_increment,
        #                                            taper_meas_height_increment)
        
        # taper_array = np.zeros((0, self.taper_measurement_heights.shape[0] + 5))           #Aglika - another hard-coded value TODO 

        taper_cols = ['Plot_ID','Tree_ID','Base_X','Base_Y','Height','Height_Bin','MA_Diameter','Bin_X','Bin_Y','S_Count','NS_Count','Pruned', 'CCI']
        taper_dict = dict(height_bin=0, diameter=1, bin_x=2, bin_y=3, stem_count=4, non_stem_count=5, pruned=6, CCI=7)
        taper_summary=np.zeros((0,9))

        # make directory for saving dBH fit plots
        dbh_fit_dir = f'{self.workdir}/DBH_fit'
        try:
            shutil.rmtree(dbh_fit_dir)
            os.mkdir(dbh_fit_dir)
        except FileNotFoundError:
            os.mkdir(dbh_fit_dir)
        taper_dir = f'{self.workdir}/taper'            
        try:
            shutil.rmtree(taper_dir)
            os.mkdir(taper_dir)
        except Exception as e:
            print(f'Error: {str(e)}. Ignored.')
            os.mkdir(taper_dir)
		
        # prepare tree_ids to be output sorted by bearing
        ids = np.unique(cleaned_cyls[:, self.cyl_dict['tree_id']])
        bearings = np.zeros(len(ids))
        for i,tree_id in enumerate(ids):
            tree_skeleton = cleaned_cyls[cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id]
            lowest_cyl = tree_skeleton[np.argmin(tree_skeleton[:,self.cyl_dict['z']])]
            dx = lowest_cyl[0] - self.plot_centre[0]
            dy = lowest_cyl[1] - self.plot_centre[1]

            bearings[i] = round(np.degrees(np.arctan2(dx, dy))) % 360
        # arrange the tree ids according to the sorted bearing
        inorder_ids = deepcopy(ids[np.argsort(bearings)])

        print('Measure DBH and taper')    
        rejected_id=-1  # ids for rejected trees
        output_id=0     # the final id after removing false positives
        idmap = []      # storing the final tree_ids ordered bt bearing
        rejectedmap=[]  # id map of rejected trees
        final_cylinders = np.zeros((0, cleaned_cyls.shape[1]))  # to store final approved trees

        for tree_id in inorder_ids:
            # 1. use skeleton to find the lowest tree point
            # 2. use vegetation and stem points to find the tree height (missclassification gets worse with height)
            #          cleaned_cyls = [x, y, z, ...., height, ..]
            tree_skeleton = cleaned_cyls[cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id]
            # if np.sum(tree_skeleton[:,self.cyl_dict['main_stem']]==1) >= self.parameters['min_tree_cyls'] :
            #     tree_skeleton = tree_skeleton[tree_skeleton[:,self.cyl_dict['main_stem']]==1]
            if tree_id == 10045:
                print(tree_id)
                        
            lowest_skeleton_point = deepcopy(tree_skeleton[np.argmin(tree_skeleton[:, self.cyl_dict['height_above_dtm']])])
            z_tree_base = lowest_skeleton_point[2] - lowest_skeleton_point[self.cyl_dict['height_above_dtm']] # z coordinate of the ground at the tree base
            x_tree_base = lowest_skeleton_point[0]   # the tree base straight down from the lowest cylinder
            y_tree_base = lowest_skeleton_point[1]

            tree_stems = deepcopy(self.stem_points[self.stem_points[:, tree_id_col] == tree_id])

            # # when the tree is on a slope, we measure from the highest ground around the stem
            ground_sp =  tree_stems[(tree_stems[:, self.stem_dict['height_above_dtm']] - np.min(tree_stems[:, self.stem_dict['height_above_dtm']])) <.01, :] # stem points at ground
            if ground_sp.shape[0]<self.parameters['min_cluster_size']: 
                ground_sp =  tree_stems[(tree_stems[:, self.stem_dict['height_above_dtm']] - np.min(tree_stems[:, self.stem_dict['height_above_dtm']])) <.04, :] # stem points at ground
            
            # z_tree_base = max(ground_sp[:,2]) - .01
            highest_ground_point = ground_sp[np.argmax(ground_sp[:,2])]
            z_tree_base = highest_ground_point[2] - highest_ground_point[self.stem_dict['height_above_dtm']]
            
            # threshold = np.percentile(ground_sp[:,self.stem_dict['height_above_dtm']], 10)
            # highest_stem_side = ground_sp[ground_sp[:,self.stem_dict['height_above_dtm']] < threshold, :] 
            # z_tree_base = np.mean(highest_stem_side[:,2]) - np.mean(highest_stem_side[:,self.stem_dict['height_above_dtm']])
            

            # DBH_mh = 0 # height of calculated DBH
            # DBH_cyls_slice=np.array([])
            # DBH_z = z_tree_base + dbh_height  # this will be replaced by the z of where the DBH is measured

            # # Calculate DBH
            # low_border = DBH_z - (self.parameters['slice_increment']*2)
            # high_border = DBH_z + (self.parameters['slice_increment']*2)

            # while (DBH_cyls_slice.shape[0] < 1) :
            #     DBH_cyls_slice = tree_skeleton[np.logical_and
            #                     (tree_skeleton[:, 2] > low_border,
            #                     tree_skeleton[:, 2] < high_border,
            #                     tree_skeleton[:, self.cyl_dict['main_stem']] == 1)]
                
            #     # move lower while still above ground_stem_cutoff_height
            #     if (low_border-z_tree_base) > self.parameters['ground_stem_cutoff_height']:
            #         low_border -= (self.parameters['slice_increment'])  
            #     # move the high border until it reaches tree_base_cutoff_height
            #     high_border += (self.parameters['slice_increment'])
                
            # # if self.parameters['mode'] == 'ALS' :
            # #     while (DBH_cyls_slice.shape[0] < 2) :
            # #         if high_border > DBH_z + 2 : break
            # #         # increase measurement area while trying to stay symmetrical around the dbh_height
            # #         if (np.min(DBH_cyls_slice[:,2]) + np.max(DBH_cyls_slice[:,2]))/2 > DBH_z and low_border > np.min(tree_skeleton[:,2]) :
            # #             low_border -= (self.parameters['slice_increment'])  
            # #         else :
            # #             high_border += (self.parameters['slice_increment'])
            # #         DBH_cyls_slice = tree_skeleton[np.logical_and
            # #                         (tree_skeleton[:, 2] > low_border,
            # #                         tree_skeleton[:, 2] < high_border,
            # #                         tree_skeleton[:, self.cyl_dict['main_stem']] > .02)]
                        
            # DBH_points_slice = tree_stems[np.logical_and(tree_stems[:, 2] > low_border,
            #                                                 tree_stems[:, 2] < high_border)]

            # if tree_id == 22:
            #     print('this tree')   # Debugging

            # DBH = np.around(np.mean(DBH_cyls_slice[:, self.cyl_dict['radius']]) * 2, 3) 
            # DBH_x, DBH_y = np.median(np.atleast_2d(DBH_cyls_slice[:, :2]), axis=0)
           
            # DBH_z = np.mean(DBH_cyls_slice[:,2]) # DBH_z where DBH was actually measured. 
            # DBH_mh = np.around(np.mean(DBH_cyls_slice[:,self.cyl_dict['height_above_dtm']]), 1)
            
            # CCI_at_BH = np.around(self.circumferential_completeness_index([DBH_x, DBH_y], DBH/2, DBH_points_slice[:, :2]))

            ## Discard the tree if its stem center at BH is outside the plot
            if (self.parameters['plot_radius'] > 0) and \
                (math.dist([x_tree_base,y_tree_base], self.plot_centre) >  self.parameters['plot_radius']) :                    
                    self.vegetation_points = np.delete(self.vegetation_points, self.vegetation_points[:, tree_id_col] == tree_id, axis=0)
                    self.stem_points = np.delete(self.stem_points, self.stem_points[:, tree_id_col] == tree_id, axis=0)  
                    rejected_id -=1     # to clean any leftover references to this id - Not needed if code is correct
                    rejectedmap.append(tree_id)
                    continue                        

            
            # # get tree vegetation points
            tree_vegetation = deepcopy(self.vegetation_points[self.vegetation_points[:, tree_id_col] == tree_id])
            # # height metrics
            if tree_vegetation.shape[0] > 0 :
                tree_height = deepcopy(np.max([np.max(tree_vegetation[:,self.veg_dict['height_above_dtm']]), np.max(tree_stems[:,self.stem_dict['height_above_dtm']])]))
                tree_max_point = np.vstack((tree_vegetation[np.argmax(tree_vegetation[:,2]), :3], tree_stems[np.argmax(tree_stems[:,2]), :3]))
                tree_max_point = deepcopy(tree_max_point[np.argmax(tree_max_point[:,2]),:]) 
            else :
                tree_height = deepcopy(np.max(tree_stems[:,self.stem_dict['height_above_dtm']]))   # dead trees, no vegetation TODO
                tree_max_point = deepcopy(tree_stems[np.argmax(tree_stems[:,2]), :3])

            tree_height = np.around(tree_height,2)

            if tree_vegetation.shape[0]>0 :
            # tree_mean_position = np.mean(np.vstack((tree_vegetation[:, :2], tree_stems[:, :2])), axis=0) 
                crown_mean_position = np.around(np.mean(np.vstack((tree_vegetation[:, :2])), axis=0) , 3)
            else: crown_mean_position=[np.nan, np.nan]

            # Taper outputs - [height_bin, diameter, bin_x, bin_y, stem_count, non_stem_count, pruned, CCI ]
            #
            taper = get_taper(tree_skeleton[tree_skeleton[:, self.cyl_dict['main_stem']]>0,:], 
                              self.taper_measurement_heights, z_tree_base, 
                              tree_stems[:,:3], tree_vegetation[:,:3],
                              self.parameters['MA_margin'], self.cyl_dict, self.parameters['dbh_correction_mm']/1000)    # self.plot_summary['PlotId'].item())

            # TODO 
            # Correct diameteres by adding < bark_thickness * sensor_noise >
            # taper[:,[taper_dict['diameter']]] = taper[:,taper_dict['diameter']] + 2*(self.parameters['bark_roughness_cm'] * sensor_noise_cm/3)/10


            if np.atleast_2d(taper).size > 0 :
                taper[:,taper_dict['diameter']] += self.parameters['dbh_correction_mm']
                # get the taper measurements at dbh_height
                DBH_taper_row = taper[taper[:, 0] == (dbh_height), :] 
                if DBH_taper_row.shape[0] == 0 : # no measurement at the required height
                    DBH_taper_row = taper[np.abs(np.argmin(np.abs(taper[:,0] - (dbh_height)))),:]
                else :  DBH_taper_row=DBH_taper_row[0]
                DBH_taper = DBH_taper_row[taper_dict['diameter']] # gets the diameter at the nearest available height
                
                # # correct for noise around multiple leaders
                # if taper[taper[:,0] == 2, 1] > DBH_taper :
                #     DBH_taper_row = taper[taper[:,0] == 2,:][0]
                #     DBH_taper = DBH_taper_row[1] 
                    
                DBH_bin = DBH_taper_row[taper_dict['height_bin']]
                DBH_x = DBH_taper_row[taper_dict['bin_x']] 
                DBH_y = DBH_taper_row[taper_dict['bin_y']]
                CCI = DBH_taper_row[taper_dict['CCI']]

                # if DBH_bin < DBH_mh :  # essentially will put a bigger weight (*3) to the lowest measurement of DBH_cyls_slice
                #                         # assuming 2 cylinders -> (3*a + b)/4
                #     DBH_mh = np.around((DBH_bin + DBH_mh) / 2, 1)
                #     DBH = (DBH + DBH_taper) /2

                bins = taper[:,taper_dict['diameter']]
                if np.any((taper[:,taper_dict['pruned']] == 1)): # any non-pruned height
                    crown_height = bins[np.max(np.where(taper[:,taper_dict['pruned']] == 1))]
                else: crown_height = 0
            
            else:
                DBH_taper = 0 
            
            # discard the tree if the DBH is too small - non-crop tree
            if (DBH_taper < self.parameters['minimum_DBH']): 
                # set ID to -1 for visualisation and QC helps to identify rejected trees from not detected ones
                rejected_id -=1
                rejectedmap.append(tree_id)
                # self.vegetation_points = np.delete(self.vegetation_points, self.vegetation_points[:, tree_id_col] == tree_id, axis=0)
                # self.stem_points = np.delete(self.stem_points, self.stem_points[:, tree_id_col] == tree_id, axis=0)  
                # cleaned_cyls = np.delete(cleaned_cyls, cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id , axis=0)
                continue              
            

            # A real crop tree
            #
            
            
            ## Creating output files starts here !!!
            output_id +=1
            
            # Save the original tree ID to idmap
            idmap.append(tree_id)

            tree_skeleton[:, self.cyl_dict['tree_id']] = output_id
            if self.parameters['generate_output_point_cloud']:
                final_cylinders = np.vstack((final_cylinders, tree_skeleton))
            
            # print("Saving DBH fit image...")
            dbh_section = taper[np.abs(taper[:,taper_dict['height_bin']]-dbh_height)<.15, :]
            best_height = dbh_height
            if dbh_section.size == 0:
                dbh_section = taper[:2, :]
                # dbh_section = (dbh_height+.01) - dbh_section[:,taper_dict['height_bin']]
                best_height = dbh_section[np.argmax(dbh_section[:,taper_dict['CCI']]),taper_dict['height_bin']]

            P = tree_stems[np.abs(tree_stems[:, self.stem_dict['height_above_dtm']] - best_height) < .05, :]
            if P.shape[0] < self.parameters['min_cluster_size']:
                P = tree_stems[np.abs(tree_stems[:, self.stem_dict['height_above_dtm']] - best_height) < .1, :]
            
            fig,ax = plt.subplots(figsize=[7,7])
            ax.set_title(f"PlotID = {self.plotID}, TreeID = {output_id}, CCI = {int(CCI)}%, height = {np.around(np.mean(P[:,self.stem_dict['height_above_dtm']]),1)}m, DBH = {DBH_taper}m")
            # ax.axis('equal')
            ax.set_aspect('equal', adjustable='datalim')
            # ax.scatter(P[:,0], P[:,1], c=np.round(P[:,self.stem_dict['Range']]*100),  marker='.', cmap='Blues')
            ax.scatter(P[:,0], P[:,1], c=P[:,2],  marker='.', cmap='Blues')
            # ax.scatter(DBH_x, DBH_y,  marker='+', color='k')
            circle_outline = plt.Circle((DBH_x,DBH_y), DBH_taper/2, fill=False, edgecolor='r')
            ax.add_patch(circle_outline)
            # plt.xlim(DBH_x-DBH_taper/2-0.04, DBH_x+DBH_taper/2+0.04)
            # plt.ylim(DBH_y-DBH_taper/2-0.04, DBH_y+DBH_taper/2+0.04)
            # plt.show()
            # Add the grid
            ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 0.05))
            ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.05))
            ax.grid(True, which='both', linewidth=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.tick_params(axis='x', rotation=90)
            plt.close()
            fig.savefig(f'{dbh_fit_dir}/{self.plotID}_DBH_tree{round(output_id)}.png', dpi=1000, bbox_inches='tight', pad_inches=0.0)
            
            # plot_fitted_circle(DBH_x, DBH_y, DBH_taper, P[:,:2],P[:,self.stem_dict['intensity']])


            print(f'Saving taper for tree {output_id}')
            # create the full taper csv
            if taper.size > 0 :
                # taper_cols = ['Plot_ID','Tree_ID','Base_X','Base_Y','Height','Height_Bin','MA_Diameter','Bin_X','Bin_Y','S_Count','NS_Count','Pruned','CCI']
                dummy = np.zeros((taper.shape[0],1))
                taper_df = np.concatenate(( dummy + output_id,
                                                dummy + taper[0,2], 
                                                dummy + taper[0,3],
                                                dummy + tree_height,
                                                taper), axis=1 )

                taper_df = pd.DataFrame(taper_df, columns = taper_cols[1:]) # convert to data frame
                taper_df.insert(loc=0, column=taper_cols[0], value='')      # add plotID column
                taper_df[taper_cols[0]] = str(self.plotID) # add plot_id value
                taper_df = taper_df.astype({"Tree_ID":"int","Pruned":"int", "CCI":"int"})    # format cols
                taper_df.to_csv(f'{taper_dir}/{self.plotID}_{round(output_id)}.csv', index=None, sep=',') 

                taper_summary = np.vstack((taper_summary, np.hstack((output_id, taper[-1,:]))))

                taper = taper_df[['Height_Bin','MA_Diameter']].values
                volume_1, volume_2, bin = volume_from_taper(taper, tree_height, dbh_height)
                dub = get_diameter_under_bark(taper[:,1], tree_height, dbh_height)
                
                
                taper[:,1] = dub # not consequential, to be used for vub only
                dub = np.atleast_2d(taper[taper[:,0] == bin,1])
                vub, _,_ =  volume_from_taper(taper, tree_height, dbh_height)
                vub = np.atleast_2d(vub)
                

            dx = DBH_x - self.plot_centre[0] 
            dy = DBH_y - self.plot_centre[1] #float(self.plot_summary['Plot Centre Y'])
            tree_dist = round(np.sqrt(dx**2 + dy**2),2)
            tree_bearing = round(np.degrees(np.arctan2(dx, dy))) % 360
            # mean_vegetation_density_in_5m_radius = 0
            # mean_understory_height_in_5m_radius = 0
            # nearby_understory_points = self.ground_veg[self.ground_veg_kdtree.query_ball_point([x_tree_base, y_tree_base], r=2)]
            # if nearby_understory_points.shape[0] > 0:
            #     mean_understory_height_in_5m_radius = np.around(np.nanmean(nearby_understory_points[:, self.veg_dict['height_above_dtm']]), 2)
                            
            description = '\n Tree ' + str(int(output_id))
            description = description + '\nDistance:   ' + str(tree_dist) + ' m'
            description = description + '\nBearing:    ' + str(tree_bearing) + ' degrees'
            description = description + '\nDBH height: ' + str(DBH_bin) + ' m'  
            description = description + '\nDBH:        ' + str(DBH_taper*1000) + ' mm'                           
            description = description + '\nHeight:     ' + str(np.around(tree_height, 3)) + ' m'
            description = description + '\nVolume:     ' + str(np.around(volume_1, 3)) + ' m^3'
            description = description + '\nVolume Cone: ' + str(np.around(volume_2, 3)) + ' m^3'
            # print(description)
            
            # making tree_data.csv file
            this_trees_data = np.zeros((1, len(self.tree_data_dict)), dtype='object')  
            this_trees_data[:, self.tree_data_dict['PlotId']] = self.plotID
            this_trees_data[:, self.tree_data_dict['TreeNumber']] = int(output_id)
            # this_trees_data[:, self.tree_data_dict['DBH_height']] = np.around(DBH_mh,1)
            this_trees_data[:, self.tree_data_dict['TreeLocation_X']] = np.around(DBH_x, 3)
            this_trees_data[:, self.tree_data_dict['TreeLocation_Y']] = np.around(DBH_y, 3)
            # this_trees_data[:, self.tree_data_dict['TreeLocation_Z']] = np.around(DBH_z, 3)
            # this_trees_data[:, self.tree_data_dict['DBH']] = DBH*1000                           # report DBH in mm
            this_trees_data[:, self.tree_data_dict['DBH_taper']] = np.around(DBH_taper*1000)    # report DBH in mm
            this_trees_data[:, self.tree_data_dict['DBH_bin']] = np.around(DBH_bin, 1)
            this_trees_data[:, self.tree_data_dict['CCI']] = np.around(CCI)
            this_trees_data[:, self.tree_data_dict['Distance']] = tree_dist                     # Aglika - added distance
            this_trees_data[:, self.tree_data_dict['Bearing']] = tree_bearing                   # Aglika - added bearing
            this_trees_data[:, self.tree_data_dict['Height']] = np.around(tree_height, 2)
            this_trees_data[:, self.tree_data_dict['Volume']] = np.around(volume_1, 3)
            # this_trees_data[:, self.tree_data_dict['Volume_2']] = np.around(volume_2, 3)
            this_trees_data[:, self.tree_data_dict['Crown_mean_x']] = crown_mean_position[0]
            this_trees_data[:, self.tree_data_dict['Crown_mean_y']] = crown_mean_position[1]
            this_trees_data[:, self.tree_data_dict['Crown_top_x']] = np.around(tree_max_point[0], 3)
            this_trees_data[:, self.tree_data_dict['Crown_top_y']] = np.around(tree_max_point[1], 3)
            this_trees_data[:, self.tree_data_dict['Crown_top_z']] = np.around(tree_max_point[2], 3)
            this_trees_data[:, self.tree_data_dict['Dub']] = np.around(dub*1000)    
            this_trees_data[:, self.tree_data_dict['Vub']] = np.around(vub,3)    # volume_under_bark
            this_trees_data[:, self.tree_data_dict['Crown_Height']] = np.around(crown_height,3)  

            text_size = 0.00256
            line_height = 0.025
            
            DBH_mh = DBH_bin
            treeID = self.point_cloud_annotations(text_size, DBH_x, DBH_y + 2 * line_height, DBH_mh + 2 * line_height, DBH_taper * 0.2, '     TREE ID: ' + str(int(output_id)))
            line0 = self.point_cloud_annotations(text_size, DBH_x, DBH_y, DBH_mh, DBH_taper * 0.2, '        DIAM: ' + str(np.around(DBH_taper*1000)) + 'mm') # Aglika - mm, no rounding
            line1 = self.point_cloud_annotations(text_size, DBH_x, DBH_y - line_height, DBH_mh - line_height, DBH_taper * 0.2, '   CCI AT BH: ' + str(np.around(CCI)))
            line2 = self.point_cloud_annotations(text_size, DBH_x, DBH_y - 2 * line_height, DBH_mh - 2 * line_height, DBH_taper * 0.2, '      HEIGHT: ' + str(np.around(tree_height, 2)) + 'm')
            line3 = self.point_cloud_annotations(text_size, DBH_x, DBH_y - 3 * line_height, DBH_mh - 3 * line_height, DBH_taper * 0.2, '      VOLUME 1: ' + str(np.around(volume_1, 2)) + 'm3')
            line4 = self.point_cloud_annotations(text_size, DBH_x, DBH_y - 4 * line_height, DBH_mh - 4 * line_height, DBH_taper * 0.2, '      VOLUME 2: ' + str(np.around(volume_2, 2)) + 'm3')

            # height_measurement_line = self.points_along_line(DBH_x, DBH_y, z_tree_base, DBH_x, DBH_y, z_tree_base + tree_height, resolution=0.025)
            height_measurement_line = self.points_along_line(DBH_x, DBH_y, 0, DBH_x, DBH_y, 0 + tree_height, resolution=0.025)

            dbh_circle_points = self.create_3d_circles_as_points_flat(DBH_x, DBH_y, DBH_mh, DBH_taper / 2, circle_points=100)
           
            tree_data = np.vstack((tree_data, this_trees_data))
            
            self.text_point_cloud = np.vstack((self.text_point_cloud, treeID, line0, line1, line2,
                                                height_measurement_line, dbh_circle_points))                                                                   
            
            if self.parameters['split_by_tree']:
                # find ground vegetation points around this tree
                kdtree = spatial.KDTree(tree_skeleton[:, :2], leafsize=1000)
                results = kdtree.query(self.ground_veg[:, :2], k=1)
                ground_veg = self.ground_veg[results[0] <= self.parameters['veg_sorting_range']]
                # find terrain points around this tree
                results = kdtree.query(self.terrain_points[:, :2], k=1)
                ground = self.terrain_points[results[0] <= self.parameters['veg_sorting_range']]
                # put together all points belong to this tree
                tree_points = np.vstack((tree_stems, tree_vegetation, ground_veg, ground))
                ## save tree to laz
                save_file(f'{self.workdir}taper/{self.plotID}_{round(output_id)}.laz', tree_points, headers_of_interest=list(self.stem_dict), offsets=self.offsets)
            
        ## End of single tree loop         


        ## Plot level output
        ##
        save_file(self.output_dir + 'text_point_cloud.laz', self.text_point_cloud, offsets=self.offsets)
        tree_data = pd.DataFrame(tree_data, columns=list(self.tree_data_dict))
        tree_data.to_csv(self.output_dir + 'tree_data.csv', index=None, sep=',') 
        
        # taper_data = pd.DataFrame(taper_array, columns=['PlotId', 'TreeNumber', 'x_base', 'y_base', 'z_base'] + [str(i) for i in self.taper_measurement_heights])
        # taper_cols = ['Plot_ID','Tree_ID','Base_X','Base_Y','Height','Height_Bin','MA_Diameter','Bin_X','Bin_Y','S_Count','NS_Count','Pruned']
        # col_names = ['bin','diameter','bin_x','bin_y','stem_count','vegetation_count','pruned']
        taper_data = pd.DataFrame(taper_summary, columns=[['TreeNumber'] + taper_cols[5:]])
        taper_data.to_csv(self.output_dir + 'taper_summary.csv', index=None, sep=',') 

        # put together stem points and vegetation points
        tree_points = np.vstack((self.stem_points, self.vegetation_points)) 
        
        # Update treeIDs in final_cylinders, stem_points and vegetation_points
        if len(idmap) > 0 :
            for newi, oldi in enumerate(idmap): # idmap stores the final trees ordered by bearing
                final_cylinders[final_cylinders[:, self.cyl_dict['tree_id']]==oldi, self.cyl_dict['tree_id']] = newi+1
                tree_points[tree_points[:, tree_id_col]==oldi, tree_id_col] = newi+1
        if len(rejectedmap) > 0 :        
            for newi, oldi in enumerate(rejectedmap): # idmap stores the final trees ordered by bearing
                final_cylinders[final_cylinders[:, self.cyl_dict['tree_id']]==oldi, self.cyl_dict['tree_id']] = rejected_id
                tree_points[tree_points[:, tree_id_col]==oldi, tree_id_col] = rejected_id


        if final_cylinders.shape[0]>0 :
            print("Making final cylinder visualisation...")
            j = 0
            cleaned_cyl_vis = []
            max_j = np.shape(final_cylinders)[0]
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, final_cylinders):
                    cleaned_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            cleaned_cyl_vis = np.vstack(cleaned_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print("\n--------Saving final cylinder visualisation...")
            save_file(self.output_dir + 'final_cyl_vis.laz', cleaned_cyl_vis,
                        headers_of_interest=list(self.cyl_dict)[:cleaned_cyl_vis.shape[1]], offsets=self.offsets)
            print("\n--------Saving final cylinder data...")
            pd.DataFrame(final_cylinders, columns=list(self.cyl_dict)).to_csv(self.output_dir + 'final_cyls.csv', index=False, sep=',')
            

        #  height normalise the final cylinders
        final_cylinders[:,2] = final_cylinders[:,self.cyl_dict['height_above_dtm']]
        if final_cylinders.shape[0]>0 :
            print("Making final cylinder visualisation...")
            j = 0
            final_cyl_vis = []
            max_j = np.shape(final_cylinders)[0]
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, final_cylinders):
                    final_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            final_cyl_vis = np.vstack(final_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print("\nSaving cylinder visualisation...")
            save_file(self.output_dir + 'final_cyl_vis_hnom.laz', final_cyl_vis,
                        headers_of_interest=list(self.cyl_dict)[:final_cyl_vis.shape[1]], offsets=self.offsets)


        if self.parameters['generate_output_point_cloud']:
            unsorted_points = np.round(np.vstack((unsorted_stems, unsorted_vegetation, self.cwd_points, self.terrain_points, self.ground_veg)), 3)
            
            if self.parameters['plot_radius'] > 0:
                
                # Crop unassigned point cloud to plot radius
                # move rejected trees to unsorted points
                mask = tree_points[:,self.stem_dict['tree_id']]<0
                unsorted_points = np.vstack((unsorted_points, tree_points[mask]))
                tree_points = np.delete(tree_points, mask, axis=0)
                # for points that belong to rejected trees keep only the ones within the plot radius    
                unsorted_points = unsorted_points[np.linalg.norm(unsorted_points[:, :2]-self.plot_centre, axis=1) < self.parameters['plot_radius']]
                
                # Crop the DTM
                self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - self.plot_centre, axis=1) < self.parameters['plot_radius']]
                save_file(self.output_dir + 'cropped_DTM.laz', self.DTM, offsets=self.offsets)                    
           
            all_points = np.round(np.vstack((tree_points, unsorted_points)), 3)
            
            # # No need of duplicates removal, saving the next for debugging purposes
            # _, indices = np.unique(all_points[:,:3], axis=0, return_index=True)
            # all_points = all_points[indices, :]
            # print(f'Deleted {all_points.shape[0]-len(indices)} duplicate points')

            # Height Classsification
            if not ('classification' in self.stem_dict.keys()):
                all_points = np.hstack((all_points, np.zeros((all_points.shape[0],1))))
                class_col=-1
            else:
                class_col=self.stem_dict['classification']
   
            height_col = self.stem_dict['height_above_dtm']
            mask = all_points[:,height_col]<=0.1 # ground   
            all_points[mask,class_col] = 2  # ground class - 10 cm above DTM 
            mask = np.logical_and(all_points[:,height_col]>.1, all_points[:,height_col]<=0.3)
            all_points[mask,class_col] = 3  # low vegetation - 10 to 30cm above DTM
            mask = np.logical_and(all_points[:,height_col]>0.3, all_points[:,height_col]<=2)
            all_points[mask,class_col] = 4  # medium vegetation - 30cm to 2m above dtm
            mask = all_points[:,height_col]>2
            all_points[mask,class_col] = 5  # high vegetation - above 2m 
            
            ## don't save Ring and gps time TODO
            ## headers_to_save = list(self.stem_dict)+['classification']
            # save_file(self.output_dir + 'temp.laz', all_points, headers_of_interest=list(self.stem_dict)+['classification'], offsets=self.offsets)
            # command = f'c:\\lastools\\bin\\lasoptimize64 -i {self.output_dir}temp.laz -do_not_set_nice_offset -o {self.output_dir}C2_0_0_treeid.laz -olaz'
            # print(command)
            # if os.system(command)==0 :
            #     os.remove(f'{self.output_dir}temp.laz')

            if self.offsets[0] > 10000 : filename = "C2_E_N"
            else : filename ="C2_0_0"

            if 1:
                # Height Normalization
                all_points[:,2] = np.around(all_points[:,self.stem_dict['height_above_dtm']], 3)
                all_points = all_points[all_points[:,2]>=0,:]  # remove underground points
                # all_points = np.delete(all_points, self.stem_dict['height_above_dtm'], axis=1) ...
                save_file(self.output_dir + 'temp.laz', all_points, headers_of_interest=list(self.stem_dict)+['classification'], offsets=self.offsets)

                command = f'c:\\lastools\\bin\\lasoptimize64 -i {self.output_dir}temp.laz -do_not_set_nice_offset -o {self.output_dir}{filename}_hnom.laz -olaz'
                print(command)
                failed = os.system(command)
                if not failed:
                    os.remove(f'{self.output_dir}temp.laz')               
                else:
                    print('lasoptimize failed')


        if not self.parameters['minimise_output_size_mode']:
            save_file(self.output_dir + 'stem_points_sorted.laz',   
                        tree_points[tree_points[:,self.stem_dict['label']] == 4,:], 
                        headers_of_interest=list(self.stem_dict), offsets=self.offsets)
            save_file(self.output_dir + 'veg_points_sorted.laz', 
                        tree_points[tree_points[:,self.stem_dict['label']] == 2,:], 
                        headers_of_interest=list(self.veg_dict), offsets=self.offsets)

        # Calculate final statistics and update plot_summary output
        # 
        plane = Plane.best_fit(self.DTM)
        avg_gradient = self.compute_angle(plane.normal, [0, 0, 1])
        avg_gradient_x = self.compute_angle(plane.normal[[0, 2]], [0, 1])
        avg_gradient_y = self.compute_angle(plane.normal[[1, 2]], [0, 1])
        self.measure_time_end = time.time()
        self.measure_total_time = np.around(self.measure_time_end - self.measure_time_start, 2)

        self.plot_summary['Measurement Time (s)'] = self.measure_total_time
        self.plot_summary['Total Run Time (s)'] = round(self.plot_summary['Preprocessing Time (s)'] + self.plot_summary['Semantic Segmentation Time (s)'] + self.plot_summary['Post processing time (s)'] + self.plot_summary['Measurement Time (s)'],2)

        self.plot_summary['Num Trees in Plot'] = tree_data.shape[0]
        self.plot_summary['Stems/ha'] = np.around(tree_data.shape[0] / self.plot_area, 1)

        if tree_data.shape[0] > 0:
            self.plot_summary['Mean DBH'] = np.mean(tree_data['DBH'])
            self.plot_summary['Median DBH'] = np.median(tree_data['DBH'])
            self.plot_summary['Min DBH'] = np.min(tree_data['DBH'])
            self.plot_summary['Max DBH'] = np.max(tree_data['DBH'])

            self.plot_summary['Mean Height'] = np.mean(tree_data['Height'])
            self.plot_summary['Median Height'] = np.median(tree_data['Height'])
            self.plot_summary['Min Height'] = np.min(tree_data['Height'])
            self.plot_summary['Max Height'] = np.max(tree_data['Height'])

            self.plot_summary['Mean Volume'] = np.mean(tree_data['Volume'])
            self.plot_summary['Median Volume'] = np.median(tree_data['Volume'])
            self.plot_summary['Min Volume'] = np.min(tree_data['Volume'])
            self.plot_summary['Max Volume'] = np.max(tree_data['Volume'])
            self.plot_summary['Total Volume'] = np.sum(tree_data['Volume'])

            # self.plot_summary['Mean Volume 2'] = np.mean(tree_data['Volume_2'])
            # self.plot_summary['Median Volume 2'] = np.median(tree_data['Volume_2'])
            # self.plot_summary['Min Volume 2'] = np.min(tree_data['Volume_2'])
            # self.plot_summary['Max Volume 2'] = np.max(tree_data['Volume_2'])
            # self.plot_summary['Total Volume 2'] = np.sum(tree_data['Volume_2'])

            # self.plot_summary['Canopy Cover Fraction'] = np.around(self.canopy_area / self.ground_area, 3)

        else:
            self.plot_summary['Mean DBH'] = 0
            self.plot_summary['Median DBH'] = 0
            self.plot_summary['Min DBH'] = 0
            self.plot_summary['Max DBH'] = 0

            self.plot_summary['Mean Height'] = 0
            self.plot_summary['Median Height'] = 0
            self.plot_summary['Min Height'] = 0
            self.plot_summary['Max Height'] = 0

            self.plot_summary['Mean Volume'] = 0
            self.plot_summary['Median Volume'] = 0
            self.plot_summary['Min Volume'] = 0
            self.plot_summary['Max Volume'] = 0
            self.plot_summary['Canopy Cover Fraction'] = 0

        self.plot_summary['Avg Gradient'] = avg_gradient
        self.plot_summary['Avg Gradient X'] = avg_gradient_x
        self.plot_summary['Avg Gradient Y'] = avg_gradient_y

        # self.plot_summary['Understory Veg Coverage Fraction'] = float(self.ground_veg_area) / float(self.ground_area)
        # self.plot_summary['CWD Coverage Fraction'] = float(self.cwd_area) / float(self.ground_area)

        self.plot_summary.to_csv(self.output_dir + 'plot_summary.csv', index=False)
        print(f"Measuring plot took {np.around(self.measure_total_time/60,2)} min")
        print("Measuring plot done.")
