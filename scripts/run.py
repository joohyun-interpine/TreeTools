from run_tools import FSCT, directory_mode, file_mode
from dependencies.other_parameters import *
from dependencies.configs_by_mls_or_als import *
from dependencies.paramters_by_clients import *

import glob


if __name__ == '__main__':
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FT_output" in their names.
    
    File mode will allow you to select multiple .las files within a directory.
    
    Alternatively, you can just list the point cloud file paths.
    
    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    # point_clouds_to_process = directory_mode(r'K:\ABP_2023\09_TREEtools')
    # point_clouds_to_process = directory_mode()
    # point_clouds_to_process = ['full_path_to_your_point_cloud.laz', 'full_path_to_your_second_point_cloud.laz', etc.]
    point_clouds_to_process = file_mode()
    # point_clouds_to_process = [r'K:\Success_Study\09_Forest_Tool\plot2.laz']
    
    # main parameters to change according to type of scan
    
    machine_config_obj = Dependencies()
    base_param_obj = BaseParameters().other_parameters()
    client_param_obj = Parameters()
    SCAN = 'MLS' # "MLS" for hovermap, "ALS" for drone, 
    if SCAN == 'ALS':
      scan_param_dict = machine_config_obj.airborne_laser_scanning()
     
    else:  # standard hovermap scans
      scan_param_dict = machine_config_obj.mobile_laser_scanning()

    for point_cloud_filename in point_clouds_to_process:
        parameters = client_param_obj.abp(point_cloud_filename)
        parameters.update(base_param_obj)
        parameters = parameters | scan_param_dict # Math Union - the second operand overwrites the first!

        # if 1:
        try:
           FSCT(parameters=parameters,
             # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
             # For standard use, just leave them all set to 1 except "clean_up_files".
             preprocess   = 1,  # Preparation for semantic segmentation.
             segmentation = 1,  # Deep learning based semantic segmentation of the point cloud.
             postprocessing = 0,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
             measure_plot = 0,  # The bulk of the plot measurement happens here.
             make_report  = 0,  # Generates a plot report, plot map, and some other figures.
             nolabel_laz_writer = 0, # Write the .laz files under 'ToClient' folder that do not have label info.
             clean_up_files=0)  # Optionally deletes most of the large point cloud outputs to minimise storage requirements.

        except Exception as e:
          print ("Error working on file " + point_cloud_filename)
          print(f"The error raised is: {str(e)}")
             