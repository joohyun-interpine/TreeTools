import numpy as np
import time
import random
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import spatial
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
from sklearn.neighbors import NearestNeighbors
import threading
from tools import load_file, save_file, make_folder_structure, subsample_point_cloud, low_resolution_hack_mode, copy_canopy_image
import os
from multiprocessing import get_context
import subprocess


class Preprocessing:
    def __init__(self, parameters):
        self.preprocessing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        self.directory = os.path.dirname(os.path.realpath(self.filename)) + '/'

        # workdir, plotID = get_output_path(self.filename)

        self.filename = self.filename.split('/')[-1]
        self.box_dimensions = np.array(self.parameters['box_dimensions'])
        self.box_overlap = np.array(self.parameters['box_overlap'])
        self.min_points_per_box = self.parameters['min_points_per_box']
        self.max_points_per_box = self.parameters['max_points_per_box']
        self.num_procs = parameters['num_procs']

        self.output_dir, self.working_dir = make_folder_structure(self.directory, self.filename)

        copy_canopy_image(self.output_dir)

        self.point_cloud, headers, self.num_points_orig = load_file(filename=self.directory + self.filename,
                                                                    plot_centre=self.parameters['plot_centre'],
                                                                    plot_radius=self.parameters['plot_radius'],
                                                                    plot_radius_buffer=self.parameters['plot_radius_buffer'],
                                                                    headers_of_interest=self.parameters['headers'],
                                                                    return_num_points=True)

        
        self.offsets=[0,0,0] # default for centered point cloud
        
        if self.parameters['plot_centre'] is None:      # otherwise
            mins = np.min(self.point_cloud[:, :3], axis=0)
            maxes = np.max(self.point_cloud[:, :3], axis=0)
            self.parameters['plot_centre'] = np.around((mins[:2] + maxes[:2]) / 2, 3)
            self.offsets[:2] = self.parameters['plot_centre']     # set offsets so plot center is at 0,0
            self.offsets[2] = np.around(mins[2],3)
            self.plot_centre = np.hstack((self.parameters['plot_centre'], np.around(mins[2],3)))

        ##--- Denoising
        print("Denoising point cloud")

        # Delete points classified as noise
        if 'classification' in headers:
            # self.point_cloud = self.point_cloud[self.point_cloud[:,headers.index('classification')] != 7,:]
            self.point_cloud = np.delete(self.point_cloud, np.where(self.point_cloud[:,headers.index('classification')] == 7), axis=0)
            print(f'Deleted {self.num_points_orig-self.point_cloud.shape[0]} noise points (classification=7)')

        self.num_points_trimmed = self.point_cloud.shape[0]

        if self.parameters['denoise_stem_points']: # using the same parameter for convinience
        
            # Delete Low Confidence Returns
        
            # remove second-return points (return number == 1) - noisier due to beam dispersion and first return occlusions


            if 'return_number' in headers:

                if np.min(self.point_cloud[:,headers.index('return_number')]) == 0:  # MLS pc
                    self.point_cloud = np.delete(self.point_cloud, np.where(self.point_cloud[:,headers.index('return_number')] > 0), axis=0)
                    print(f'Deleted {self.num_points_trimmed-self.point_cloud.shape[0]} points with return number > 0')                
                    self.num_points_trimmed = self.point_cloud.shape[0]

                elif np.max(self.point_cloud[:,headers.index('return_number')]) > 1 :   # ALS pc
                    # for ALS point clouds return number 1 is mostly noise and the highest section of canopy
                    # Removing it make segmentation more accurate and faster             
                    max_z = max(self.point_cloud[:,2]) 
                    noise_mask = np.logical_and(self.point_cloud[:,headers.index('return_number')] < 2,
                                                self.point_cloud[:,2] < (max_z-5))  # keep the top section intact
                    self.point_cloud = self.point_cloud[~noise_mask,:]    
                    print(f'Deleted {self.num_points_trimmed-self.point_cloud.shape[0]} points with return number > 1')
                    self.num_points_trimmed = self.point_cloud.shape[0]
        
            # # remove points with intensity < 7 -- most of these are noise in a regular plantation
            if 'intensity' in headers:  
                # # delete low-intensity points to help segmentation
                noise_mask = self.point_cloud[:,headers.index('intensity')] < self.parameters['noise_intensity_threshold']
                temp_vegetation = self.point_cloud[noise_mask,:]
                self.point_cloud = self.point_cloud[~noise_mask,:]
                save_file(self.output_dir + 'temp_vegetation.laz', temp_vegetation, headers_of_interest=headers, offsets = self.offsets)
                
                # mask = self.point_cloud[:,headers.index('intensity')] > 7
                # self.point_cloud = self.point_cloud[mask,:]     
                # mask = self.point_cloud[:,headers.index('intensity')] < 220
                # self.point_cloud = self.point_cloud[mask,:]     

                print(f"Deleted {self.num_points_trimmed - self.point_cloud.shape[0]} points with intensity < {self.parameters['noise_intensity_threshold']}")
                self.num_points_trimmed = self.point_cloud.shape[0]
        
            
            if 'Range' in headers : # # remove far range point
        
                # keep only close range for z < 5  TODO the next should be replaced with a SOR filter on the slices. Done!!
                min_z = min(self.point_cloud[:,2]) # not ideal, underground noise and slope will fail this
                noise_mask = np.logical_and(self.point_cloud[:,headers.index('Range')] > 20,
                                            self.point_cloud[:,2] < (min_z+5))
                self.point_cloud = self.point_cloud[~noise_mask,:]
                print(f'Deleted {self.num_points_trimmed - self.point_cloud.shape[0]} points with range > 20 in the DBH section')
                self.num_points_trimmed = self.point_cloud.shape[0]                        
        

            # TODO - use for ALS only
            # if 'number_of_returns' in headers:
            #     # save points which are last returns - must be big objects like ground and stem
            #     mask = (self.point_cloud[:,headers.index('return_number')] == self.point_cloud[:,headers.index('number_of_returns')])
            #     self.point_cloud = self.point_cloud[mask]
            #     save_file(f'{self.output_dir}/last_returns.laz', self.point_cloud, headers_of_interest=headers, offsets=self.offsets)            

            
        #### end of de-noising


        # remove duplicate XYZ if any
        # self.point_cloud = np.vstack(list(tuple{row for row in self.point_cloud}))  # nice but slow
        _,indices = np.unique(np.round(self.point_cloud[:,:3],3), axis=0, return_index=True)
        self.point_cloud = self.point_cloud[indices,:] # keep the unique points (by XYZ)
        print(f'Deleted {self.num_points_trimmed-self.point_cloud.shape[0]} duplicate xyz points')
        
        self.num_points_subsampled = 0
        # Subsample - does not work correctly
        if self.parameters['subsample']:            
            self.point_cloud = subsample_point_cloud(self.point_cloud,
                                                     self.parameters['subsampling_min_spacing'],
                                                     self.num_procs)
            self.num_points_subsampled = self.point_cloud.shape[0]
            print(f'Subsampling deleted {self.num_points_trimmed-self.num_points_subsampled} points')


        # save the clipped, denoised and subsampled point cloud - the starting point for analysis although 
        #                                                         it will loose more points during 'box-ing'
        save_file(f'{self.output_dir}/working_point_cloud.laz', self.point_cloud, headers_of_interest=headers, offsets = self.offsets)

        
        # Global shift the point cloud to avoid loss of precision during segmentation.
        self.point_cloud[:, :2] = self.point_cloud[:, :2] - self.parameters['plot_centre']
        # self.point_cloud[:, :3] = self.point_cloud[:, :3] - self.plot_centre

        self.point_cloud = self.point_cloud[:, :3]  # Trims off unneeded dimensions if present.        


    @staticmethod
    def threaded_boxes(point_cloud, box_size, min_points_per_box, max_points_per_box, path, id_offset, point_divisions):

        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size
        i = 0
        pds = len(point_divisions)
        while i < pds:
            box = point_cloud
            box = box[np.logical_and(np.logical_and(np.logical_and(box[:, 0] >= box_centre_mins[i, 0],
                                                                   box[:, 0] < box_centre_maxes[i, 0]),
                                                    np.logical_and(box[:, 1] >= box_centre_mins[i, 1],
                                                                   box[:, 1] < box_centre_maxes[i, 1])),
                                     np.logical_and(box[:, 2] >= box_centre_mins[i, 2],
                                                    box[:, 2] < box_centre_maxes[i, 2]))]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    indices = list(range(0, box.shape[0]))
                    random.shuffle(indices)
                    random.shuffle(indices)
                    box = box[indices[:max_points_per_box], :]
                    box = np.asarray(box, dtype='float64')

                box[:, :3] = box[:, :3]
                np.save(path + str(id_offset + i).zfill(7) + '.npy', box)
            i += 1
        return 1

    def preprocess_point_cloud(self):
        print("Pre-processing point cloud...")
        point_cloud = self.point_cloud  # [self.point_cloud[:,4]!=5]
        Xmax = np.max(point_cloud[:, 0])
        Xmin = np.min(point_cloud[:, 0])
        Ymax = np.max(point_cloud[:, 1])
        Ymin = np.min(point_cloud[:, 1])
        Zmax = np.max(point_cloud[:, 2])
        Zmin = np.min(point_cloud[:, 2])       

        X_range = Xmax - Xmin
        Y_range = Ymax - Ymin
        Z_range = Zmax - Zmin

        num_boxes_x = int(np.ceil(X_range / self.box_dimensions[0]))
        num_boxes_y = int(np.ceil(Y_range / self.box_dimensions[1]))
        num_boxes_z = int(np.ceil(Z_range / self.box_dimensions[2]))

        x_vals = np.linspace(Xmin, Xmin + (num_boxes_x * self.box_dimensions[0]),
                             int(num_boxes_x / (1 - self.box_overlap[0])) + 1)
        y_vals = np.linspace(Ymin, Ymin + (num_boxes_y * self.box_dimensions[1]),
                             int(num_boxes_y / (1 - self.box_overlap[1])) + 1)
        z_vals = np.linspace(Zmin, Zmin + (num_boxes_z * self.box_dimensions[2]),
                             int(num_boxes_z / (1 - self.box_overlap[2])) + 1)

        box_centres = np.vstack(np.meshgrid(x_vals, y_vals, z_vals)).reshape(3, -1).T

        point_divisions = []
        for thread in range(self.num_procs):
            point_divisions.append([])

        points_to_assign = box_centres

        while points_to_assign.shape[0] > 0:
            for i in range(self.num_procs):
                point_divisions[i].append(points_to_assign[0, :])
                points_to_assign = points_to_assign[1:]
                if points_to_assign.shape[0] == 0:
                    break
        threads = []
        prev_id_offset = 0
        for thread in range(self.num_procs):
            id_offset = 0
            for t in range(thread):
                id_offset = id_offset + len(point_divisions[t])
            print('Thread:', thread, prev_id_offset, id_offset)
            prev_id_offset = id_offset
            t = threading.Thread(target=Preprocessing.threaded_boxes, args=(self.point_cloud,
                                                                            self.box_dimensions,
                                                                            self.min_points_per_box,
                                                                            self.max_points_per_box,
                                                                            self.working_dir,
                                                                            id_offset,
                                                                            point_divisions[thread],))
            threads.append(t)
        print("Starting threads")
        for x in threads:
            x.start()
        print("Joining threads output")
        for x in threads:
            x.join()

        self.preprocessing_time_end = time.time()
        self.preprocessing_time_total = self.preprocessing_time_end - self.preprocessing_time_start
        print(f"Preprocessing took {np.around(self.preprocessing_time_total/60,2)} min")
        plot_summary_headers = ['PlotId',
                                'Point Cloud Filename',
                                'Plot Centre X',
                                'Plot Centre Y',
                                'Plot Centre Z',
                                'Offset X',
                                'Offset Y',
                                'Offset Z',
                                'Plot Radius',
                                'Plot Radius Buffer',
                                'Plot Area',
                                'DBH_height',
                                'Num Trees in Plot',
                                'Stems/ha',
                                'Mean DBH',
                                'Median DBH',
                                'Min DBH',
                                'Max DBH',
                                'Mean Height',
                                'Median Height',
                                'Min Height',
                                'Max Height',

                                'Mean Volume',
                                'Median Volume',
                                'Min Volume',
                                'Max Volume',
                                'Total Volume',

                                'Mean Volume 2',
                                'Median Volume 2',
                                'Min Volume 2',
                                'Max Volume 2',
                                'Total Volume 2',

                                'Avg Gradient',
                                'Avg Gradient X',
                                'Avg Gradient Y',
                                'Canopy Cover Fraction',
                                'Understory Veg Coverage Fraction',
                                'CWD Coverage Fraction',
                                'Num Points Original PC',
                                'Num Points Trimmed PC',
                                'Num Points Subsampled PC',
                                'Num Terrain Points',
                                'Num Vegetation Points',
                                'Num CWD Points',
                                'Num Stem Points',
                                'Preprocessing Time (s)',
                                'Semantic Segmentation Time (s)',
                                'Post processing time (s)',
                                'Measurement Time (s)',
                                'Total Run Time (s)']

        plot_summary = pd.DataFrame(np.zeros((1, len(plot_summary_headers))), columns=plot_summary_headers)

        plot_summary['Preprocessing Time (s)'] = self.preprocessing_time_total
        plot_summary['PlotId'] = self.filename[:-4]   #removing the '.laz' part
        plot_summary['Point Cloud Filename'] = self.parameters['point_cloud_filename']
        #plot_summary['Plot Centre X'] = self.plot_centre[0]
        #plot_summary['Plot Centre Y'] = self.plot_centre[1]
        #plot_summary['Plot Centre Z'] = self.plot_centre[2]
        plot_summary['Plot Radius'] = self.parameters['plot_radius']
        plot_summary['Plot Radius Buffer'] = self.parameters['plot_radius_buffer']
        plot_summary['DBH_height'] = self.parameters['dbh_height']
        plot_summary['Num Points Original PC'] = self.num_points_orig
        plot_summary['Num Points Trimmed PC'] = self.num_points_trimmed
        plot_summary['Num Points Subsampled PC'] = self.num_points_subsampled
        plot_summary['Offset X'] = self.offsets[0]
        plot_summary['Offset Y'] = self.offsets[1]       
        plot_summary['Offset Z'] = self.offsets[2]

        plot_summary.to_csv(self.output_dir + 'plot_summary.csv', index=False)
        print("Preprocessing done\n")
