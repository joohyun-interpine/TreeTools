from data_selection import *
from data_splitter import *

class DataPrep:
    """
    This is an integrated data preparation process,
    1. data selection - applied
    2. lid off outliers - yet
    3. over and down sampling - yet
    4. data splitting - applied
    
    """
    def __init__(self):
        pass
    
    def get_high_stem_laz(self, folder_path):
        dsObj = DataSelection(folder_path)
        las_laz_list = dsObj.get_las_laz_list()
        file_stem_dict = dsObj.has_label_or_not(las_laz_list)
        higher_stem_proportion_data = dsObj.get_over_median_stem_proportion_data(file_stem_dict)         
        dsObj.create_selected_dir(selected = 'selected', discarded = 'discarded')
        selected_folder_path = dsObj.data_cut_paste(higher_stem_proportion_data)
        
        return selected_folder_path
        
    def data_split(self, lassplit_path, folder_path):
        
        selected_laz_files = os.listdir(folder_path)
        for selected_laz_file in selected_laz_files:
            selected_laz_file_path = os.path.join(folder_path, selected_laz_file)            
            dsobj = DataSplitter(selected_laz_file_path)
            dataset_boundaries = dsobj.get_boundaries()
            subfolder_list = dsobj.cmd_executor(lassplit_path, dataset_boundaries)
            dsobj.create_nonlabelled_test_laz(subfolder_list)
            
        
def main():
    
    dpobj = DataPrep()
    folder_path = r'C:\Users\JooHyunAhn\Interpine\DataSets\TreeTools_PlayGroundSet\data_splitter'
    lassplit_path = r'C:\LAStools\bin\lassplit.exe'
    
    selected_folder_path = dpobj.get_high_stem_laz(folder_path) 
    print("Data selection is completed, and data splitting will be stated")   
    dpobj.data_split(lassplit_path, selected_folder_path)
    print("Data splitting is completed, now training can be started")   
    
if __name__ == "__main__":
    main()
    
    