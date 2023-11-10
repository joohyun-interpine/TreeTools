from run_tools import FSCT, directory_mode, file_mode
from other_parameters import other_parameters
import glob


if __name__ == '__main__':
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FT_output" in their names.
    
    File mode will allow you to select multiple .las files within a directory.
    
    Alternatively, you can just list the point cloud file paths.
    
    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    # point_clouds_to_process = directory_mode(r'V:\Succes_Study\DC61A_01\09_TREEtools')
    # point_clouds_to_process = directory_mode()
    # point_clouds_to_process = ['full_path_to_your_point_cloud.laz', 'full_path_to_your_second_point_cloud.laz', etc.]
    point_clouds_to_process = file_mode()
    # point_clouds_to_process = [r'K:\Success_Study\09_Forest_Tool\plot2.laz']

    SCAN = 'MLS' # "MLS" for hovermap, "ALS" for drone, 
    # main parameters to change according to type of scan
    if SCAN == 'ALS':
      scan_param_dict ={"denoise_stem_points":False, # Aerial data has fewer stem points so we are keeping then all
                          "tree_base_cutoff_height":13.1, # # to ignore unassigned branches
                          "gap_connect":.5, #used in dbscan to connect skeleton points into tree skeleton
                          "subsample":1,  # Aerial data has very dense foliage - processing takes ages, therefore subsample
                          "headers":['x', 'y', 'z', 'intensity','return_number','number_of_returns','classification', '']
                          }
    else:  # standard hovermap scans
      scan_param_dict={"denoise_stem_points":True, 
                       "noise_intensity_threshold":2,
                       "SOR_filter" : True, # applies Statistical outlier removal on the slices of stem points
                        "gap_connect":.205, # used in dbscan to connect skeleton points into tree skeleton
                        "subsample":0,
                        "headers":['x', 'y', 'z', 'intensity','return_number','gps_time','Ring','Range']
                        }

    for point_cloud_filename in point_clouds_to_process:
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

        parameters.update(other_parameters)
        parameters = parameters | scan_param_dict # Math Union - the second operand overwrites the first!

        # if 1:
        try:

           FSCT(parameters=parameters,
             # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
             # For standard use, just leave them all set to 1 except "clean_up_files".
             preprocess   = 0,  # Preparation for semantic segmentation.
             segmentation = 0,  # Deep learning based semantic segmentation of the point cloud.
             postprocessing=0,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
             measure_plot = 0,  # The bulk of the plot measurement happens here.
             make_report  = 1,  # Generates a plot report, plot map, and some other figures.
             clean_up_files=0)  # Optionally deletes most of the large point cloud outputs to minimise storage requirements.

        except Exception as e:
          print ("Error working on file " + point_cloud_filename)
          print(f"The error raised is: {str(e)}")
             