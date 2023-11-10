import os
import sys
import yaml
from run_tools import FSCT, directory_mode, file_mode
from run_tools import file_mode
from other_parameters import other_parameters
import pprint
import argparse

default_parameters = {
    'ALS_scan': { 
                'denoise_stem_points': False,
                'gap_connect': 0.5,
                'headers': [
                            'x',
                            'y',
                            'z',
                            'intensity',
                            'return_number',
                            'number_of_returns',
                            'classification',
                            ''
                        ],
                'subsample': 1,
                'tree_base_cutoff_height': 13.1
                  },
    'MLS_scan': { 
                'denoise_stem_points': True,
                'gap_connect': 0.205,
                'headers': [
                            'x',
                            'y',
                            'z',
                            'intensity',
                            'return_number',
                            'gps_time',
                            'Ring',
                            'Range'
                          ],
                'subsample': 0,
                'tree_base_cutoff_height': 2.8
                  },
    'general_parameters': { 
                            'MA_margin': 0.3,
                            'dbh_correction_mm': 0,
                            'batch_size': 4,
                            'dbh_height': 1.3,
                            'delete_working_directory': True,
                            'generate_output_point_cloud': 1,
                            'ground_stem_cutoff_height': 0.4,
                            'ground_veg_cutoff_height': 0.8,
                            'maximum_DBH': 0.7,
                            'minimise_output_size_mode': 1,
                            'minimum_DBH': 0.07,
                            'num_procs': 12,
                            'plot_centre': [0, 0],
                            'plot_radius': 14.1,
                            'plot_radius_buffer': 1.9,
#                            'point_cloud_filename': 'my_point_cloud.laz',
                            'slice_increment': 0.1,
                            'single_increment_height': 15,
                            'slice_thickness': 0.15,
                            'sort_stems': 1,
                            'split_by_tree': False,
                            'stem_sorting_range': 1.5,
                            'taper_measurement_height_increment': 0.1,
                            'taper_measurement_height_max': 20,
                            'taper_measurement_height_min': 0.5,
                            'tree_stem_min_height': 3.0,
                            'use_CPU_only': False,
                            'veg_sorting_range': 2.5
                            }
}

def export_dict_to_yaml(yaml_file, dict_data):
    """Export dictionary as YAML"""    
    with open(yaml_file, 'w') as outfile:
        yaml.dump(dict_data, outfile, default_flow_style=False)

def import_yaml(yaml_file):
    """Import YAML file and return a dictionary"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='run_with_config.py',
                    )
    
    parser = argparse.ArgumentParser(description='Run HoverMap LiDAR processing.',epilog='Copyright Interpine Group Ltd 2023')
    
    parser.add_argument('scantype',   help='Scantype: ALS or MLS.')             # positional argument
    parser.add_argument('foldername', help='Location of the LAZ files to be processed.')  # positional argument
    parser.add_argument('-p', '--preprocessing',  action='store_true', help='Preparation for semantic segmentation.')
    parser.add_argument('-s', '--segmentation',   action='store_true', help='Deep learning based semantic segmentation of the point cloud.') 
    parser.add_argument('-o', '--postprocessing', action='store_true', help='Creates the DTM and applies some simple rules to clean up the segmented point cloud.')
    parser.add_argument('-m', '--measureplot',    action='store_true', help='The bulk of the plot measurement happens here.') 
    parser.add_argument('-w', '--wrapup',         action='store_true', help='Rename output files, create a merged las')
    parser.add_argument('-r', '--makereport',     action='store_true', help='Generates a plot report, plot map, and some other figures.') 
    parser.add_argument('-c', '--cleanupfiles',   action='store_true', help='Optionally deletes most of the large point cloud outputs to minimise storage requirements.') 

    args = parser.parse_args()
    pprint.pprint(args)                

    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FT_output" in their names.
    
    File mode will allow you to select multiple .las files within a directory.
    
    Alternatively, you can just list the point cloud file paths.
    
    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """
    point_clouds_to_process = directory_mode(args.foldername)
    # point_clouds_to_process = directory_mode()
    # point_clouds_to_process = ['full_path_to_your_point_cloud.laz', 'full_path_to_your_second_point_cloud.laz', etc.]
    # point_clouds_to_process = file_mode()
    # point_clouds_to_process = [r'K:\ABP_2023\09_TREEtools\LIR566small_0_0.laz']

    for point_cloud_filename in point_clouds_to_process:
        #try: 
        config_file = os.path.join(os.path.dirname(point_cloud_filename),os.path.basename(point_cloud_filename).split("_")[0]+".yaml")
        
        print(config_file)
    
        if(os.path.exists(config_file)):
            print("Found config file {0} - importing...".format(config_file))
            #config_parameters = import_yaml("config.yaml")
            config_parameters = import_yaml(config_file)
        else:
            config_parameters = default_parameters
        
        #pprint.pprint(config_parameters)

        if args.scantype == 'ALS':
            scan_param_dict = config_parameters["ALS_scan"]
        else:  # standard hovermap scans
            scan_param_dict = config_parameters["MLS_scan"]
    
        parameters = dict(point_cloud_filename=point_cloud_filename) | config_parameters["general_parameters"]

        parameters.update(other_parameters)
        parameters = parameters | scan_param_dict # Math Union - the second operand overwrites the first!

        # Prints a nicely formatted dictionary
        pprint.pprint(parameters)

        if 1:
            try:

                FSCT(parameters=parameters,
                    # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
                    # For standard use, just leave them all set to 1 except "clean_up_files".
                    preprocess     = args.preprocessing   ,  # Preparation for semantic segmentation.
                    segmentation   = args.segmentation    ,  # Deep learning based semantic segmentation of the point cloud.
                    postprocessing = args.postprocessing  ,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
                    measure_plot   = args.measureplot     ,  # The bulk of the plot measurement happens here.
                    make_report    = args.makereport      ,  # Generates a plot report, plot map, and some other figures.
                    clean_up_files = args.cleanupfiles       # Optionally deletes most of the large point cloud outputs to minimise storage requirements.
                    )  

            except Exception as e:
                print ("Error working on file " + point_cloud_filename)
                print(f"The error raised is: {str(e)}")
         