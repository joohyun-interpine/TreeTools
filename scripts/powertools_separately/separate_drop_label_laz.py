import laspy
import os


def create_subfolder(parent_path):
    """
    Create a folder for delivery to the client that will contain multiple .laz files which does not have 'label' colmun 

    Args:
        parent_path (str): a parent folder path that will contain a sub folder which will be used as a delivery folder to client.
    """
    
    sub_folder_name = 'ToClient'
    client_folder_path = os.path.join(parent_path, 'ToClient')
    
    if os.path.exists(client_folder_path) == False:
        os.mkdir(client_folder_path)
    else:
        print("The folder {} is already there".format(sub_folder_name))
    
    return client_folder_path


def get_laz_contains_label(parent_path):
    """
    Finding .laz files under the given parent path which is usually project's output folder

    Args:
        parent_path (str): a parent folder path that will contain a sub folder which will be used as a delivery folder to client.

    Returns:
        list: the entire file paths of the .laz files under the given parent folder
    """
    # Get a list of all files in the folder
    all_contents = os.listdir(parent_path)
    # Filter files with .laz extension
    laz_files = [file for file in all_contents if file.endswith('.laz')]
    # Full paths to laz files
    laz_file_paths = [os.path.join(parent_path, file) for file in laz_files]    
    
    return laz_file_paths

def is_label_contain_laz(laz_file_paths):
    """
    Finding laz files that contains 'label' column,
    then make them as a list with full paths

    Args:
        laz_file_paths (list): all .laz files' full paths

    Returns:
        list: only gives .laz file' paths that contains 'label'
    """
    label_contained_laz_list = []
    
    for each_laz_file in laz_file_paths:
        las_file = laspy.read(each_laz_file)
        point_format = las_file.point_format
        # Access the names of the fields (attribute names)
        field_names = list(point_format.dimension_names)
        
        if 'label' in field_names:
            label_contained_laz_list.append(each_laz_file)
    
    return label_contained_laz_list
    
def write_laz_file_without_label(client_folder_path, label_contained_laz_list):
    """
    Write a .laz file that exclude 'label' values from the original .laz file which has 'label' values    

    Args:
        client_folder_path (str): the subfolder path for delivery to a client
        label_contained_laz_list (list): full paths for laz files that do not have 'label'
    """
    # This for loop will save the each laz file into a new .laz file which does not contain 'label'.
    for each_laz_file in label_contained_laz_list:        
        las_file_with_extension = os.path.basename(each_laz_file)
        las_file_without_extension = os.path.splitext(las_file_with_extension)[0]
        las_file_extension = os.path.splitext(las_file_with_extension)[-1]
        name_for_client = '_client' # You can change whatever you want to, but please remeber that it iwwl be shown to the client.
        # new_laz_file_name = las_file_without_extension + name_for_client + las_file_extension
        new_laz_file_name = las_file_without_extension + las_file_extension
        new_laz_output_file_path = os.path.join(client_folder_path, new_laz_file_name)
        
        # Load the laz file with laspy lib
        las_file = laspy.read(each_laz_file)

        # # Access the point format to get information about the attributes
        point_format = las_file.point_format
        # # Access the names of the fields (attribute names)
        field_names = list(point_format.dimension_names)
        
        # Create a list of field names excluding 'label'
        fields_to_keep = [field_name for field_name in field_names if field_name != 'label']
  
        # Create a new LAS file without the 'label' field
        header = las_file.header
        header.data_format_id = 1  #  The new LAS file should be in a standard LAS format
        header.point_format_id = 3  # The new LAS point format with X, Y, Z coordinates and intensity
        header.point_count = len(las_file.points)

        # Write a laz file with the given header which does not have 'label'
        out_las = laspy.LasData(header)
        for field_name in fields_to_keep:
            setattr(out_las, field_name, getattr(las_file, field_name))

        out_las.write(new_laz_output_file_path)
    

def main():
    parent_path = r'G:\Hancock Queensland Plantation\Current\Southern_Pine_genetics_trial\HQP_Red_Ridge_220\Reports_and_Presentations\StraightnessScore_Material\Plot95_FT_output\Trees\Tree_with_Ground'
    client_folder_path = create_subfolder(parent_path)
    laz_file_paths = get_laz_contains_label(parent_path)    
    label_contained_laz_list = is_label_contain_laz(laz_file_paths)
    write_laz_file_without_label(client_folder_path, label_contained_laz_list)
    
main()
