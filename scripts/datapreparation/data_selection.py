import laspy
from datetime import datetime, timedelta
import numpy as np
import os
from data_splitter import *


class DataSelection:
    """
    The .laz files will be assessed whether it has high proportion of stem class or not.
    If the stem point proportion is higher than the median proportion. it will be selected and moved to the actual data folder.
    """
    def __init__(self, path):
        self.path = path
        
    def get_las_laz_list(self):
        """
        Get a list that contains the paths of .las or .laz files with the given folder path
        
        Returns:
            list: Full paths of .las or .laz files
        """
        all_files = os.listdir(self.path)
        las_laz_list = []
        for file in all_files:
            if file.endswith('.las') or file.endswith('.laz'):
                file_path = os.path.join(self.path, file)
                las_laz_list.append(file_path)
                
        return las_laz_list
    
    
    def get_stem_proportion(self, laspy_obj):
        """
        Counting the number of points of each class from the 'label' axis which is the last dimesion of the 3d point cloud numpy array.
        Once the number of classes has been counted, then it will be stored in a dictionary with itw own path and proportion of it.
        Args:
            laspy_obj (lsapy obejct): it comes from this line 'laspy.read(las_laz_file)'

        Returns:
            a dictionary: {'path' : proprotion value}
        """
        label_axis_values = laspy_obj.label
        unique_values, counts = np.unique(label_axis_values, return_counts=True)
        counted_class = dict(zip(unique_values, counts))
        stem_class_count = counted_class[4]
        entire_class_count = sum(counted_class.values())
        stem_proportion = stem_class_count / entire_class_count        
            
        return stem_proportion
    
    def outlier_with_1darray(self, filepath, array):
        # Extract values and sort them
        print('filepath', filepath)
        observations = list(array)
        sorted_values = sorted(observations)
        
        # Calculate median which is q2 by quartiles
        q1 = np.percentile(sorted_values, 25)
        q2 = np.percentile(sorted_values, 50)
        q3 = np.percentile(sorted_values, 75)
        # Calculate IQR
        iqr = q3 - q1
        
        # Find no outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = {index: value for index, value in enumerate(array) if value < lower_bound or value > upper_bound}
        non_outliers = {index: value for index, value in enumerate(array) if value >= lower_bound and value <= upper_bound}
        
        # non_outliers = {key: value for key, value in dict.items() if value > lower_bound or value < upper_bound}
        # outliers = {key: value for key, value in dict.items() if value < lower_bound or value > upper_bound}

        return outliers, non_outliers
    
            
    def is_outlier_xyz(self, dict):
        
        x_outlier = []
        y_outlier = []
        z_outlier = []
        
        for paths in dict:
            # las_file = laspy.read(dict[paths])
            las_file = laspy.read(paths)

            # Access specific attributes
            x = las_file.x  # x-coordinate
            x_observations = x.array
            y = las_file.y  # y-coordinate
            y_observations = y.array
            z = las_file.z  # z-coordinate
            z_observations = z.array
            
            
            observations = list(x_observations)
            sorted_values = sorted(observations)
            
            # Calculate median which is q2 by quartiles
            q1 = np.percentile(sorted_values, 25)
            q2 = np.percentile(sorted_values, 50)
            q3 = np.percentile(sorted_values, 75)
            
            # Calculate IQR
            iqr = q3 - q1
            
            # Find no outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            x_outlier.append(paths, lower_bound, upper_bound)
        
        return x_outlier
            
    
    def has_label_or_not(self, las_laz_list):
        """
        checking whether the .las or .laz data has 'label' or not,
        Depends on label containing status, they will sbe stored in each list

        Args:
            las_laz_list (list): must have a full path of each .las or .laz file

        Returns:
            list: .las or .laz objects which have 'label' column
            
        Comment:
            specifical case: 0 count for some classes
        """
        
        label_contained_laz_list = []
        label_uncontained_laz_list = []
        file_stem_dict = {}
        for las_laz_file in las_laz_list:
            laspy_obj = laspy.read(las_laz_file)
            point_format = laspy_obj.point_format
            field_names = list(point_format.dimension_names)
            
            if 'label' in field_names:
                label_contained_laz_list.append(las_laz_file)
            else:
                label_uncontained_laz_list.append(las_laz_file)
                
        
            stem_proportion = self.get_stem_proportion(laspy_obj)      
            file_stem_dict[las_laz_file] = stem_proportion       
        
        print("Total {} does have label info out of {} data".format(len(label_contained_laz_list), len(las_laz_list)))
        print("Total {} does not have label info out of {} data".format(len(label_uncontained_laz_list), len(las_laz_list)))
                        
        return file_stem_dict
    
    def get_over_median_stem_proportion_data(self, dict):
        """
        It calculate the median value of the stem proportion from the given dataset,
        then create a dictionary that contains only the high stem proportions data which are higher than median value that comes from Q2 value of Interquartile range (IQR) 
        The description of IQR -> https://en.wikipedia.org/wiki/Interquartile_range
        
        Args:
            dict (dictionary): {key == .laz or .laz file path: value == a value for stem point's proportion}

        Returns:
            dict: {key == .laz or .laz file path: value == a value for stem point's proportion}
        """
        # Extract values and sort them
        values = list(dict.values())
        sorted_values = sorted(values)
        
        # Calculate median which is q2 by quartiles
        q1 = np.percentile(sorted_values, 25)
        q2_median = np.percentile(sorted_values, 50)
        q3 = np.percentile(sorted_values, 75)
        # Calculate IQR
        iqr = q3 - q1
        
        # Find lower stem proportion data
        stem_proportion_least_bound = q2_median
        upper_bound = q3 + 1.5 * iqr
        
        higher_stem_proportion_data = {key: value for key, value in dict.items() if value > stem_proportion_least_bound and value < upper_bound}
        print('The stem proportion between {} and {} have been selected, which are {} files out of {} files '.format(round(stem_proportion_least_bound, 3),
                                                                                                                     round(upper_bound, 3),
                                                                                                                     len(higher_stem_proportion_data),
                                                                                                                     len(sorted_values)))

        return higher_stem_proportion_data

    def get_no_outliers_by_iqr(self, dict):
        """
        Define the outliers and non-outliers based on IQR

        Args:
            dict (dictionary): {key == .laz or .laz file path: value == a value for stem point's proportion}

        Returns:
            dicts: 
             outliers {key == .laz or .laz file path: value == a value for stem point's proportion}
             non-outliers {key == .laz or .laz file path: value == a value for stem point's proportion}
        """
        
        # Extract values and sort them
        values = list(dict.values())
        sorted_values = sorted(values)
        
        # Calculate median which is q2 by quartiles
        q1 = np.percentile(sorted_values, 25)
        q2 = np.percentile(sorted_values, 50)
        q3 = np.percentile(sorted_values, 75)
        # Calculate IQR
        iqr = q3 - q1
        
        # Find no outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        non_outliers = {key: value for key, value in dict.items() if value > lower_bound or value < upper_bound}
        outliers = {key: value for key, value in dict.items() if value < lower_bound or value > upper_bound}
        
        print('TThe lower bound is {}, upper bound is {}, and median is {}: so the number of outlier is {} and non-outlier is {} '.format(round(lower_bound, 3),
                                                                                                                                          round(upper_bound, 3),
                                                                                                                                          round(q2, 3),
                                                                                                                                          len(non_outliers),
                                                                                                                                          len(outliers)))

        return outliers, non_outliers
    
    
    def create_selected_dir(self, selected = '', discarded = ''):
        """
        Create directories for training, validation and test dataset.

        Args:
            selected (str, optional): Defaults to 'selected'.
            discarded (str, optional): Defaults to 'discarded'.            
        """
        
        selected_folder_name = os.path.join(self.path, selected)        
        discarded_folder_name = os.path.join(self.path, discarded)
        
        if not os.path.exists(selected_folder_name):
            os.mkdir(selected_folder_name)
        else:
            print("{} data folder is already there".format(selected_folder_name))
        if not os.path.exists(discarded_folder_name):
            os.mkdir(discarded_folder_name)
        else:
            print("{} data folder is already there".format(discarded_folder_name))

    
    def data_cut_paste(self, path_list):
        folder_list = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        folder_paths = [os.path.join(self.path, folder) for folder in folder_list]
        
        for selected_lazfile in path_list:
            parent_path = os.path.split(selected_lazfile)[0]
            file_name_extension = os.path.split(selected_lazfile)[-1]
            file_name = os.path.splitext(file_name_extension)[0]
            file_extension = os.path.splitext(file_name_extension)[-1]
            
            selected_cut_path = selected_lazfile
            selected_paste_path = os.path.join(folder_paths[1], file_name_extension)
            selected_cmd = "move " + selected_cut_path + " " + selected_paste_path
            
            os.system(selected_cmd)
            print("{} is moved to {} folder".format(file_name_extension, folder_paths[1]))
        
        all_contents = os.listdir(self.path)
        dicarded_laz_file = [file for file in all_contents if file.endswith('.laz') or file.endswith('.las')]
        dicarded_laz_file_paths = [os.path.join(self.path, file) for file in dicarded_laz_file]
        
        for dicarded_lazfile in dicarded_laz_file_paths:
            file_name_extension = os.path.split(dicarded_lazfile)[-1]
            dicarded_cut_path = dicarded_lazfile
            dicarded_paste_path = os.path.join(folder_paths[0], file_name_extension)
            dicarded_cmd = "move " + dicarded_cut_path + " " + dicarded_paste_path
            
            os.system(dicarded_cmd)
            print("{} is moved to {} folder".format(file_name_extension, folder_paths[0]))

        return folder_paths[1]
    
    
########### If you want to use this script separately from the data_praparation.py then, uncomment the below lines ################

# def main():
#     path = r'C:\Users\JooHyunAhn\Interpine\DataSets\TreeTools_PlayGroundSet\data_selection'
#     dsObj = DataSelection(path)    
#     las_laz_list = dsObj.get_las_laz_list()
#     file_stem_dict = dsObj.has_label_or_not(las_laz_list)
#     higher_stem_proportion_data = dsObj.get_over_median_stem_proportion_data(file_stem_dict)    
#     dsObj.create_dataset_dir(train='train', val='validation', test='test')
#     dsObj.is_outlier_xyz(higher_stem_proportion_data)
    
 
# if __name__ == "__main__":
#     main()