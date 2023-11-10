import arcpy
from arcpy import da
import os
import yaml
import pprint
def export_dict_to_yaml(yaml_file, dict_data, overwrite=False):
    """Export dictionary as YAML"""
    mode = 'w'
    if overwrite == False:
        mode = 'a'
    with open(yaml_file, mode) as outfile:
        yaml.dump(dict_data, outfile, default_flow_style=False)
        

if(len(sys.argv)!=4):
    print("Usage:")
    print("")
    print("     get_config_from_Survey123.py <Survey123 GDB> <Feature Class Name> <Output Folder>")
    print("")
    exit(1)
        
gdb           = sys.argv[1]
fc_name       = sys.argv[2]
output_folder = sys.argv[3]

from arcpy import env

env.workspace = gdb

datasetList = arcpy.ListFeatureClasses("*")

for dataset in datasetList:
    if(dataset==fc_name):
        survey_form = dataset
        print("Found feature class {0}".format(survey_form))

print("Extracting configs from {0} to {1}".format(survey_form,output_folder))

field_list = [
            'plotdatetime',                     # 0
            'plotid',                           # 1
            'projectid',                        # 2
            'inventoryprovider',                # 3
            'crew_leader_initials',             # 4
            'plot_shape',                       # 5
            'trimble',                          # 6
            'dbhclear',                         # 7
            'projectid',                        # 8
            'tree1dbh',                         # 9
            'tree1height',                      #10
            'tree1bearing',                     #11
            'tree1distance',                    #12
            't1crowndistancea',                 #13
            't1crowndistanceb',                 #14
            't1crowndistancec',                 #15
            't1crowndistanced',                 #16
            'mindbh',                           #17
            'maxdbh',                           #18
            'tree_stem_min_height',             #19
            'ground_veg_cutoff_height',         #20
            'centrepeg_gps_here',               #21
            'latitude',                         #22
            'longitude',                        #23
            'plot_notes',                       #24
            'created_date',                     #25
            'created_user',                     #26
            'last_edited_date',                 #27
            'last_edited_user'                  #28
            ]
            
            
als_scan_dict = {
    "denoise_stem_points":False, 
    "tree_base_cutoff_height":13.1, # to ignore unassigned branches
    "gap_connect":0.5,              # used in dbscan to connect skeleton points into tree skeleton
    "subsample":1,                  # Aerial data has very dense foliage - processing takes ages, therefore subsample
    "headers":['x', 'y', 'z', 'intensity','return_number','number_of_returns','classification', '']
}
mls_scan_dict={
    "denoise_stem_points":True, 
    "tree_base_cutoff_height":2.8,   # the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)
    "gap_connect":0.205,             # used in dbscan to connect skeleton points into tree skeleton
    "subsample":0,
    "headers":['x', 'y', 'z', 'intensity','return_number','gps_time','Ring','Range']
}
    
with da.SearchCursor(survey_form, field_list) as cursor:
    for item in cursor:
        survey_date = item[0]
        plot_id     = item[1]
        project_id  = item[2]
        plot_shape  = item[5]
        filename = str(plot_id) + ".yaml"
        plot_header_dict = {"plot_id": plot_id, "project_id": project_id, "survey_date": survey_date, "plot_shape": plot_shape}
        general_parameters_dict = {
        
            # To give a 20mm margin for error do
            # minimum_DBH = <Croptree_min DBH> - .02
            # If trees of too small dbh are detected, they can be deleted manually from the table with the results.
            "minimum_DBH": round((item[17]/1000)-0.02,2),   # To give a 20mm margin for error do minimum_DBH = <Croptree_min DBH> - .02

            # The error margin can be larger here because usually we need to find all big trees 
            # compared to the minimum_DBH where we want to discard bushes, undergrowth, wildlings, etc. 
            # This parameter is used by the algorithm to discard erroneous Ransac results that happen 
            # near the ground or up in the crown. Therefore, I would use 100mm margin for bigger trees 
            # (dbh=400mm) and maybe 50mm for smaller (dbh=200mm). In other word add 1/4th to the expected DBH.
            "maximum_DBH": round(((item[18]/1000)*5)/4,2),

            # The Treetools parameter is the stem height and not the total tree height. 
            # From my observations I think that we can reliably fit circles up to 1/3 of the tree height. 
            # (This is not true if the trees are unpruned.) This parameter is used to discard tree stumps 
            # or short non crop trees that fit into the DBH requirements. Very dense bushes can appear as thick-stem trees. For ABP use:
            "tree_stem_min_height": round(item[19]/3,2),

            "MA_margin": 0.3,
            "dbh_correction_mm": 0,
            "batch_size": 4,
            "dbh_height": 1.3,
            "delete_working_directory": True,
            "generate_output_point_cloud": 1,
            "ground_stem_cutoff_height": 0.4,
            "ground_veg_cutoff_height": item[20],
            "minimise_output_size_mode": 1,
            "num_procs": 12,
            "plot_centre": [ 0, 0 ],
            "plot_radius": 14.1,
            "plot_radius_buffer": 1.9,
            "slice_increment": 0.1,
            "single_increment_height": 10,
            "slice_thickness": 0.15,
            "sort_stems": 1,
            "split_by_tree": False,
            "stem_sorting_range": 1.2,
            "taper_measurement_height_increment": 0.1,
            "taper_measurement_height_max": 50,
            "taper_measurement_height_min": 0.5,
            "use_CPU_only": False,
            "veg_sorting_range": 2.5
        }
        
        config_dict = { "ALS_scan": als_scan_dict, "MLS_scan": mls_scan_dict, "general_parameters": general_parameters_dict }
        export_dict_to_yaml(os.path.join(output_folder,filename),plot_header_dict,overwrite=True)
        export_dict_to_yaml(os.path.join(output_folder,filename),config_dict)
