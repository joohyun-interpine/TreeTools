class Parameters:
    """
    Each functions that contains parameters for implementing TreeTools below belongs to each client,
    so in the 'run.py' can call this function depends on clients.
    """
    def __init__(self) -> None:
        pass        
    
    def abp(self, point_cloud_filename):
        """
        Client: Australian Bluegum Plantations, Aka = ABP

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 14.1,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=2.9, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.035,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .8,
                         
                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=4,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time

                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=False,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 0  # default = 0
                          )
        
        return parameters
    
    
    def dpi(self, point_cloud_filename):
        """
        Client: New South Wales Department of Primary Industries, Aka = DPI

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 17.82,       # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=1,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = .09,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = 1.,
                          tree_stem_min_height=4.1, # trees must have a cylinder above this height to be kept - takes care of stumps, undergrowth and non-crop trees

                          # Set these appropriately for your hardware.
                          batch_size=10,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.12,  # thickness of point cloud slice to be used for circle fitting
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.08,  # Distance between slices

                          #denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          
                          # dependant on the size of the trees, forest density and undergrowth height
                          #tree_base_cutoff_height=4.1,  # A tree must have a diameter measurement below this height to be kept. This filters unsorted branches from being called individual trees.
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                          
                          # taper measurement
                          taper_measurement_height_min=.8,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. #Aglika - needs a FIX
                          
                          delete_working_directory=False,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 0  # default = 0
                          )
        
        return parameters
    
    def fcnsw(self, point_cloud_filename):
        """
        Client: Forestry Corporation of New South Wales, Aka = FCNSW

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 13.82,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=1.9, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode". It was 1.9m but it's too small for leaning trees outside. 

                          # DBH height - height above ground to take DBH mea surement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.1,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = 0.5,
                          tree_stem_min_height = 0.1, # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)                         
                          tree_base_cutoff_height = 5, # upper bound of the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)                          
                          dbh_correction_mm = 0.0,  # default is 0.0 Adds a correction to every measured diameter up to 2m above the ground. A positive or a negative number in millimeters

                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 
                         
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.2,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.2,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=4,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time

                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          )
        
        return parameters
    
    def hqp(self, point_cloud_filename):
        """
        Client: HQPlanation, Aka = HQP

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre = None,      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 0,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=0, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.08,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .4,
                          tree_stem_min_height = 1.3, # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)                            
                          tree_base_cutoff_height = 2.2, # upper bound of the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)                          
                          dbh_correction_mm = 0,  # default is 0.0 Adds a correction to every measured diameter up to 2m above the ground. A positive or a negative number in millimeters.

                         
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=4,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time


                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          )
        
        return parameters
    
    def lucas(self, point_cloud_filename):
        """
        Client: Land Use and Carbon Analysis System, Aka = LUCAS

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 13.85,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=3.15, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.4,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.08,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .8,
                          tree_stem_min_height = 2., # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)                          
                          tree_base_cutoff_height = 2.8, # upper bound of the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)                          
                          dbh_correction_mm = 0,  # default is 0.0 Adds a correction to every measured diameter up to 2m above the ground. A positive or a negative number in millimeters.

                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 

                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.

                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=20,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Moving average margin for taper output. The average of the diameteres within +/- MA_margin are used for taper measurement at a given height. 


                          # Set these appropriately for your hardware.
                          batch_size=2,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices.
                          single_increment_height = 15, # doubled increment is used above this height to save time

                          # Output control
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          )
        
        return parameters
    
    def MPI(self, point_cloud_filename):
        """
        Client: Ministry for Primary Industries, Aka = MPI

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=None,      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 13.82,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=3, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode". It was 1.9m but it's too small for leaning trees outside. 

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.4,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.06,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = 1.0,
                          tree_stem_min_height = 1.4, # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)
                          tree_base_cutoff_height = 3.2, # upper bound of the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)                          
                          dbh_correction_mm = 0,  # default is 0.0 Adds a correction to every measured diameter up to 2m above the ground. A positive or a negative number in millimeters.

                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 

                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.3,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=20,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Moving average margin for taper output. The average of the diameteres within +/- MA_margin are used for taper measurement at a given height. 
                          

                          # Set these appropriately for your hardware.
                          batch_size=2,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 15, # doubled increment is used above this height to save time

                          # Output control
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          )
        
        return parameters
    
    def redwoods(self, point_cloud_filename):
        """
        Client: Redwoods Forest, Whakarewarewa, Aka = Redwoods

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 13.8,       # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=1.2,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.4,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = .4,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = 1.6,
                          tree_stem_min_height=3.1, # trees must have a stem point above this height to be kept - takes care of stumps, undergrowth and non-crop trees

                          # Set these appropriately for your hardware.
                          batch_size=10,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=20,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.14,  # thickness of point cloud slice to be used for circle fitting
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.2,  # Distance between slices

                          #denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          
                          # dependant on the size of the trees, forest density and undergrowth height
                          #tree_base_cutoff_height=4.1,  # A tree must have a diameter measurement below this height to be kept. This filters unsorted branches from being called individual trees.
                          ground_veg_cutoff_height=.3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.5,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                          
                          # taper measurement
                          taper_measurement_height_min=.8,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. #Aglika - needs a FIX
                          delete_working_directory=False,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 10
                          )
        
        return parameters
    
    def treeiso(self, point_cloud_filename):
        """
        Client: TreeISO, Aka = TreeISO

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=None,      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 0,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=0, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode". It was 1.9m but it's too small for leaning trees outside. 

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.04,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .9,
                          tree_stem_min_height = 2., # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)                          
                          tree_base_cutoff_height = 2.8, # upper bound of the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)                          
                          dbh_correction_mm = 0,  # default is 0.0 Adds a correction to every measured diameter up to 2m above the ground. A positive or a negative number in millimeters.

                          bark_sensor_return = "Sparse",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 

                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.5,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=20,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Moving average margin for taper output. The average of the diameteres within +/- MA_margin are used for taper measurement at a given height. 
                          

                          # Set these appropriately for your hardware.
                          batch_size=2,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters - default is 0.15 (15cm)
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices. This is a tradoff between speed and accuracy in the taper
                          single_increment_height = 15, # doubled increment is used above this height to save time

                          # Output control
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          )
        
        return parameters
    
    def uoi(self, point_cloud_filename):
        """
        Client: UOI, Aka = UOI

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 13.57,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=6.43, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.035,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .7,
                         
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=2,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time

                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 0  # default = 0
                          )
        
        return parameters
    
    def ofo(self, point_cloud_filename):
        """
        Client: One forty One, Aka = OFO

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=[0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 20,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=2.9, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.035,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .8,
                         
                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=4,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time

                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=False,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 0  # default = 0
                          )
        
        return parameters
    
    def joohyun(self, point_cloud_filename):
        """
        Client: Joo-Hyun Testing, Aka = joohyun

        Args:
            point_cloud_filename (str): the file name of .laz file

        Returns:
            dictionary: parameters
        """
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre = [0,0],      # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
                          plot_radius = 20,    # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer = 22.9, # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # DBH height - height above ground to take DBH measurement
                          dbh_height = 1.3,      # 1.4m for New Zealand, 1.3m for the rest of the world
                          minimum_DBH = 0.035,  # (metres) trees having DBH smaller than minimum_DBH will be deleted
                          maximum_DBH = .8,
                         
                          bark_sensor_return = "Normal",  # values can be "Dense", "Normal" or "Sparse"
                                                          # "Dense" for australian Bluegum and mature Redwoods
                                                          # "Sparse" for young trees and poor semantic segmentation 
                          # dependant on the size of the trees, forest density and undergrowth height
                          ground_veg_cutoff_height=.8,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          ground_stem_cutoff_height=.4,  # Any stem points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=2.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=1.2,  # Stem points can be, at most, this far away from a cylinder center in 3D to be matched to a particular tree.
                         
                          # Set these appropriately for your hardware.
                          batch_size=2,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=12,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
                         
                          # Optional settings - Generally leave as they are. Speed vs accuracy.
                          slice_thickness=0.15,  # thickness of point cloud slice to be used for finding clusters
                          # If your point cloud is really dense, you may get away with 0.1.
                          slice_increment=0.1,  # Distance between slices
                          single_increment_height = 10, # doubled increment is used above this height to save time

                          # denoise_stem_points = True, # if True, stem points of Height<5 AND Range>11 will be discarded
                                                      # Also, this parameter is used as a denoising flag during preprocessing
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                           
                          # taper measurement
                          taper_measurement_height_min=.5,  # Lowest height to measure diameter for taper output.
                          taper_measurement_height_max=50,  # Highest height to measure diameter for taper output.
                          taper_measurement_height_increment=0.1,  # diameter measurement increment. #Aglika - always uses 0 as a start point - needs a FIX - done
                          MA_margin=0.30,  # Cylinder measurements within +/- taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used. 
                                            #Aglika - needs a FIX - currently a moving average in the range +-30cm
                          
                          delete_working_directory=False,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished. 
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=1,  # Will not write a few non-essential outputs to reduce storage use.
                          # QC=''  # use QC='_QC' to generate plot reports from QC tables. Will look for {plotID}_QC_data.csv file
                          split_by_tree = False, # if True, outputs a las file for each detected tree in the plot (outputs to the 'taper' directory)
                          dbh_correction_mm = 0  # default = 0
                          )
        
        return parameters