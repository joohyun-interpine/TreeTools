from modulefinder import IMPORT_NAME
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
from preprocessing import Preprocessing
from inference_new import SemanticSegmentation
from post_segmentation_script import PostProcessing
from measure_work import MeasureTree
from tools import add_plotid_to_name, clean_output
import report
from drop_label_laz import *
import glob
import tkinter as tk
import tkinter.filedialog as fd


def FSCT(parameters, 
         preprocess=True, 
         segmentation=True, 
         postprocessing=True, 
         measure_plot=True, 
         make_report=False, 
         nolabel_laz_writer=False, 
         clean_up_files=False):
    
    print(parameters['point_cloud_filename'])

    if preprocess:
        preprocessing = Preprocessing(parameters)
        preprocessing.preprocess_point_cloud()
        del preprocessing

    if segmentation:
        sem_seg = SemanticSegmentation(parameters)
        sem_seg.inference()
        del sem_seg

    if postprocessing:
        object_1 = PostProcessing(parameters)
        object_1.process_point_cloud()
        del object_1

    if measure_plot:
        measure1 = MeasureTree(parameters)
        measure1.vegetation_coverage()
        measure1.find_skeleton()  # slow
        measure1.cylinder_assignment()
            # # measure1.cylinder_smoothing()  # don't use
            # #  measure1.connect_branches()   # don't use
        measure1.tree_metrics()
        del measure1
        add_plotid_to_name(parameters['point_cloud_filename'])          # !! carefull with this one - not easy to undo     
        
    
    if make_report:
        report.make_report(parameters)

        # report_writer = ReportWriter(parameters)
        # report_writer.make_report()          
        # del report_writer
    
    if nolabel_laz_writer:
        laz_writer_obj = LazFileWriter(parameters)            
        # laz_writer_obj.create_subfolder()
        # laz_writer_obj.get_laz_contains_label()
        # laz_writer_obj.is_label_contain_laz()
        laz_writer_obj.write_laz_file_without_label()
   
    if clean_up_files:
        clean_output(parameters['point_cloud_filename'], parameters['delete_working_directory'])


def directory_mode(directory=None):
    # root = tk.Tk()
    point_clouds_to_process = []
    # directory = fd.askdirectory(parent=root, title='Choose directory')
    # unfiltered_point_clouds_to_process = glob.glob(directory + '/**/*.laz', recursive=False)  # the /**/ pattern

    unfiltered_point_clouds_to_process = glob.glob(directory + '/*.laz', recursive=False)

    for i in unfiltered_point_clouds_to_process:
        if 'FT_output' not in i:  # use to process files that don't have an FT_output folder
        # if 'FT_output' in i:    # use this to re-process already existing FT_output folders                                    
            point_clouds_to_process.append(i)
    # root.destroy()                                          # close interactive mode
    return point_clouds_to_process


def file_mode():
    root = tk.Tk()
    point_clouds_to_process = fd.askopenfilenames(parent=root, title='Choose files',
                                                  filetypes=[("LAZ", "*.laz"), ("LAS", "*.las"), ("CSV", "*.csv")])
    root.destroy()                          # close interactive mode
    return point_clouds_to_process
