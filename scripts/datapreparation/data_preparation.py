from data_selection import *
from data_splitter import *
from data_removing_ambigous_area import *


class DataPrep:
    """
    This is an integrated data preparation process,
    1. Remove ambigous area - applied
    2. data selection - applied
    3. lid off outliers - yet
    4. over and down sampling - yet
    5. data splitting - applied
    
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def removing_ambigous_area(self, las2las_path):
        """
        Removing the ambigous area between stem and canopy

        Args:
            las2las_path (str): a path where the las2las.exe file exist

        Returns:
            str: a path for unique_nominated_folder_path
            str: a path for unique_dropped_folder_path
        """
        
        nominated_folder_path_list = []
        dropped_folder_path_list = []
        
        laz_files = os.listdir(self.folder_path)
        for laz_file in laz_files:
            laz_file_path = os.path.join(self.folder_path, laz_file)
            dsObj = DataRemovingAmbigousArea(laz_file_path)
            height, height_indices_stem, height_indices_canopy = dsObj.get_data()
            stats_dict = dsObj.get_stats_plots(height, height_indices_stem, height_indices_canopy, titles_data = ['Stem', 'Canopy'], plot=False)
            nominated_path, dropped_path, nominated_name, dropped_name = dsObj.cmd_executor(las2las_path, stats_dict)
            txt_file_path = dsObj.write_stats(stats_dict)
            points_ratio_info = dsObj.cal_ratio_dropped_points(nominated_path, dropped_path)
            dsObj.append_to_file(txt_file_path, points_ratio_info)
            nominated_folder_path, dropped_folder_path = dsObj.create_folders_moving(nominated_path, dropped_path, nominated_name, dropped_name)
            nominated_folder_path_list.append(nominated_folder_path)
            dropped_folder_path_list.append(dropped_folder_path)
        
        unique_nominated_folder_path = list(set(nominated_folder_path_list))
        unique_dropped_folder_path = list(set(dropped_folder_path_list))
        
        return unique_nominated_folder_path[0], unique_dropped_folder_path[0]
    
    def get_high_stem_laz(self, folder_path):
        """
        Select only the heigh proprotion of stem inside the .laz file

        Args:
            folder_path (str): a path for each .laz file

        Returns:
            str: a path where the seleceted .laz file is stored
        """
        dsObj = DataSelection(folder_path)
        las_laz_list = dsObj.get_las_laz_list()
        file_stem_dict = dsObj.has_label_or_not(las_laz_list)
        higher_stem_proportion_data = dsObj.get_over_median_stem_proportion_data(file_stem_dict)         
        dsObj.create_selected_dir(selected = 'selected', discarded = 'discarded')
        selected_folder_path = dsObj.data_cut_paste(higher_stem_proportion_data)
        
        return selected_folder_path
        
    def data_split(self, lassplit_path, folder_path):
        """
        A plot will be split into 3 parts such as training, validation, and test
        train: 50%
        validation: 25%
        test: 25%

        Args:
            lassplit_path (str): a path where lassplit.exe file exist
            folder_path (_type_): a path where the .laz file exist 
        """
        
        selected_laz_files = os.listdir(folder_path)
        for selected_laz_file in selected_laz_files:
            selected_laz_file_path = os.path.join(folder_path, selected_laz_file)            
            dsobj = DataSplitter(selected_laz_file_path)
            dataset_boundaries = dsobj.get_boundaries()
            subfolder_list = dsobj.cmd_executor(lassplit_path, dataset_boundaries)
            dsobj.create_nonlabelled_test_laz(subfolder_list)
            
        
def main():
    
    folder_path = r'C:\Users\JooHyunAhn\Interpine\DataSets\TreeTools_PlayGroundSet\removing_ambigous_area'
    lassplit_path = r'C:\LAStools\bin\lassplit.exe'
    las2las_path = r'C:\LAStools\bin\las2las.exe'
    selected_folder_path = r'C:\RemoteSensing\MLS\TreeTools\data'
    
    dpobj = DataPrep(folder_path)
    
    # unique_nominated_folder_path, unique_dropped_folder_path = dpobj.removing_ambigous_area(las2las_path)
    # print("Removing ambogous area is completed, and data selection will be started")   
    # selected_folder_path = dpobj.get_high_stem_laz(unique_nominated_folder_path) 
    # print("Data selection is completed, and data splitting will be started")   
    dpobj.data_split(lassplit_path, selected_folder_path)
    print("Data splitting is completed, now training can be started")
    
if __name__ == "__main__":
    main()
    
    