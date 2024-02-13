# Don't change these unless you really understand what you are doing with them/are learning how the code works.
# These have been tuned to work on most high resolution forest point clouds without changing them, but you may be able
# to tune these better for your particular data. Almost everything here is a trade-off between different situations, so
# optimisation is not straight-forward.
class BaseParameters:
    """
    The parameters of deep learning model architecture
    """
    def __init__(self) -> None:
        pass         
    
    def other_parameters(self):
            other_parameters = dict(model_filename='FC_upperstem_branch_4classes_Epoch555_ValEpochAcc-0p68.pth',
                                    # model_filename='model_original.pth',
                            box_dimensions=[6, 6, 6],  # Dimensions of the sliding box used for semantic segmentation. # Joo-Hyun Testing
                            # box_dimensions=[4,4,10],
                            # box_dimensions=[1,1,1],
                            ##box_dimensions=[4,4,10],
                            box_overlap=[0.5, 0.5, 0.5],  # Overlap of the sliding box used for semantic segmentation.
                            # box_overlap=[0.5, 0.5, 0.5],
                            min_points_per_box=1000,  # Minimum number of points for input to the model. Too few points and it becomes near impossible to accurately label them (though assuming vegetation class is the safest bet here).
                            # max_points_per_box=20000,  # Maximum number of points for input to the model. The model may tolerate higher numbers if you decrease the batch size accordingly (to fit on the GPU), but this is not tested.
                            max_points_per_box=200000,  # Maximum number of points for input to the model. The model may tolerate higher numbers if you decrease the batch size accordingly (to fit on the GPU), but this is not tested.
                            noise_class=0,  # Don't change
                            terrain_class=1,  # Don't change
                            vegetation_class=2,  # Don't change
                            cwd_class=3,  # Don't change
                            stem_class=4,  # Don't change
                            grid_resolution=0.5,  # Resolution of the DTM. !! Do not change before revising the make_dtm function, which is full of hard-coded values!!
                            vegetation_coverage_resolution=0.2,
                                                    
                            num_neighbours=5,

                            # HDBSCAN clustering parameters -   # these are affected by the slice_thickness!!! :(
                            min_cluster_size=40,  # Used for HDBSCAN clustering step to find clusters of stem points. Recommend 30 for general use (3D).
                                                    # Aglika - use bigger value for larger trees - 50
                                                    # attempt to fit a cirle will be done on any cluster of min_cluster_size points
                            min_samples=10,       # used in a combination with min_cluster_size to keep clusters of dense points 
                                                    # these are the number of dense points that might represent a stem/branch section 
                            eps=0.03,            # the distance
                        
                            cluster_size_threshold = [[250, 100, 50],[100, 50, 20],[30,10,10]],  # !!! hard-coded for now
                                                                    # [250, 100, 50], for older trees, e.g. DBH > 200mm
                                                                    # [100, 50, 20], for younger trees, e.g. DBH ~= 100mm, height < 20m, Species dependant
                            sorting_search_angle=20,    # Currently not used
                            sorting_search_radius=3,    #   " "                 
                            sorting_angle_tolerance=40, #   " "
                            
                            max_search_radius=2,    # orig = 3 -- Search radius for interpolating tree segments 
                            max_search_angle=20,

                            # cleaned_measurement_radius=0.1,  # During cleaning, this will leave only 1 cylinder in a sphere of this value
                            # subsampling_min_spacing=0.01,  # The point cloud will be subsampled such that the closest any 2 points can be is 0.01 m.
                            
                            minimum_CCI=40,  # Minimum valid Circumferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
                            min_tree_cyls=10,  # Deletes any trees with fewer than 10 cylinders (before the cylinder interpolation step).

                            low_resolution_point_cloud_hack_mode=0)  # Very ugly hack that can sometimes be useful on point clouds which are on the borderline of having not enough points to be functional with FSCT. Set to a positive integer. Point cloud will be copied this many times (with noise added) to artificially increase point density giving the segmentation model more points.
        
            return other_parameters