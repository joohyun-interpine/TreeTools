import laspy
from datetime import datetime, timedelta
import numpy as np
import os


class DataSplitter:
    """
    A plot will be splited into three datasets such as training, validation, and testing, the ratio is 50%:25%:25%
    Testing data will be cloned, but the clone does not have label, so it will be used for inference and model assessment along with labelled testing data
    """
    def __init__(self, path):
        self.path = path
        self.parent_path = os.path.split(self.path)[0]

    def create_folders(self):
        """
        Creating the sub folders for data splitting into 3 such as 'train', 'validation', and 'test'
        'test' folder will have two subfolders which are 'labelled', 'nonlabelled'
        """
        
        datasets = ['train', 'val', 'test']
        test_sets = ['labelled', 'nonlabelled']
        sample_dir = 'sample_dir'
        
        for folder in datasets:
            dataset_path = os.path.join(self.parent_path, folder)
            dataset_sample_path = os.path.join(dataset_path, sample_dir)
            
            if folder == 'train' or folder == 'val':
                # Creating for the dataset_path
                if not os.path.exists(dataset_path):
                    os.mkdir(dataset_path)
                else:
                    print("The folder {} is already there".format(dataset_path))
                
                # Creating for the dataset_sample_dir path    
                if not os.path.exists(dataset_sample_path):
                    os.mkdir(dataset_sample_path)
                else:
                    print("The folder {} is already there".format(dataset_sample_path))
                
            else:
                if not os.path.exists(dataset_path):
                    os.mkdir(dataset_path)
                else:
                    print("The folder {} is already there".format(dataset_path))
                    
                for sub_test_folder in test_sets:
                    test_sub_folder_path = os.path.join(dataset_path, sub_test_folder)
                    if not os.path.exists(test_sub_folder_path):
                        os.mkdir(test_sub_folder_path)
                    else:
                        print("The folder {} is already there".format(test_sub_folder_path))
                        
        msg = "Folders {}, {}, {} are all set under the {}".format(datasets[0], datasets[1], datasets[2], self.parent_path)
        print(msg)
        
        sub_folder_paths = [os.path.join(root, folder) for root, dirs, files in os.walk(self.parent_path) for folder in dirs]
        
        return sub_folder_paths
        
    def get_boundaries(self):
        """
        Getting coordinates of each dataset's boundary such as training, validation, and test.
        Assume that the plot can be divided into 4 blocks (2 * 2), 
        then bottom 2 blocks will be used for training that equivalent 50% of area,
        then the top left block will be used for validation,
        and remaining top right block will be used for testing

        Returns:
            dictionay: contains the entire plot boundary, center point coordinate, and each dataset's boundaries
        """
        
        las_file = laspy.read(self.path)
        min_x, min_y, min_z = las_file.header.min
        max_x, max_y, max_z = las_file.header.max

        # Calculate center point coordinates
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        train_xy_coordinate = [min_x, min_y, max_x, center_y]
        val_xy_coordinate = [min_x, center_y, center_x, max_y]
        test_xy_coordinate = [center_x, center_y, max_x, max_y]       
        
        return [train_xy_coordinate, val_xy_coordinate, test_xy_coordinate]
        
    def cmd_executor(self, lassplit_path, dataset_boundaries):
        """
        Create commands for using lassplit.exe through os.system module,
        

        Args:
            lassplit_path (str): file path of lassplit.exe
            dataset_boundaries (list): xmin/ymax value of the training, validation, and test dataset

        Returns:
            list: train_laz_file_path, val_laz_file_path, test_laz_file_path
        """
        sub_folder_paths = self.create_folders()        
        # laz_file_extension = os.path.splitext(os.path.split(self.path)[-1])[-1]
        
        train_list = dataset_boundaries[0]
        train_coor = str(train_list[0]) + " " + str(train_list[1]) + " " + str(train_list[2]) + " " + str(train_list[3])        
        train_laz_file_path = sub_folder_paths[1]
        train_cmd = lassplit_path + " -i " + self.path + " -keep_xy " + train_coor + " -odir " + train_laz_file_path + " -olaz"
        print(train_cmd)
        os.system(train_cmd)        
        print('train data has been splitted')
        
        val_list = dataset_boundaries[1]
        val_coor = str(val_list[0]) + " " + str(val_list[1]) + " " + str(val_list[2]) + " " + str(val_list[3])
        val_laz_file_path = sub_folder_paths[2]
        val_cmd = lassplit_path + " -i " + self.path + " -keep_xy " + val_coor + " -odir " + val_laz_file_path + " -olaz"
        print(val_cmd)
        os.system(val_cmd)
        print('validation data has been splitted')
        
        test_list = dataset_boundaries[2]
        test_coor = str(test_list[0]) + " " + str(test_list[1]) + " " + str(test_list[2]) + " " + str(test_list[3])
        test_laz_file_path = sub_folder_paths[3]
        test_cmd = lassplit_path + " -i " + self.path + " -keep_xy " + test_coor + " -odir " + test_laz_file_path + " -olaz"
        print(test_cmd)
        os.system(test_cmd)
        print('test data has been splitted')
        
        return [train_laz_file_path, val_laz_file_path, test_laz_file_path]
        
    def get_laz_contains_label(self, folder_path):
        """
        Get the laz file' paths with the given folder path

        Args:
            folder_path (str): the folder path where the laz files are sitting

        Returns:
            list: full file path of each laz file
        """
        all_contents = os.listdir(folder_path)
        laz_files = [file for file in all_contents if file.endswith('.laz')]
        laz_file_paths = [os.path.join(folder_path, file) for file in laz_files]

        return laz_file_paths
        
    def create_nonlabelled_test_laz(self, subfolder_list):
        """
        Create label removed test .laz data which will be used for inference when the training is completed

        Args:
            test labelled path (str): the labelled laz file path
        """
        test_laz_file_paths = self.get_laz_contains_label(subfolder_list[-1])
        for laz_file in test_laz_file_paths:
            las_file = laspy.read(laz_file)
            point_format = las_file.point_format
            field_names = list(point_format.dimension_names)
            fields_to_keep = [field_name for field_name in field_names if field_name != 'label']

            header = las_file.header
            header.data_format_id = 1
            header.point_format_id = 3
            header.point_count = len(las_file.points)

            out_las = laspy.LasData(header)
            for field_name in fields_to_keep:
                setattr(out_las, field_name, getattr(las_file, field_name))
            nonlabelled_laz_output_file_path = laz_file.replace("labelled","nonlabelled")
            out_las.write(nonlabelled_laz_output_file_path)
            print("Nonlabelled .laz file created:", nonlabelled_laz_output_file_path)


########### If you want to use this script separately from the data_praparation.py then, uncomment the below lines ################

# def main():
#     lassplit_path = r'C:\LAStools\bin\lassplit.exe'
#     path = r'C:\Users\JooHyunAhn\Interpine\DataSets\TreeTools_PlayGroundSet\data_splitter\LIR42_C2_0_0.laz'
#     dsObj = DataSplitter(path) 
#     dataset_boundaries = dsObj.get_boundaries()
#     subfolder_list = dsObj.cmd_executor(lassplit_path, dataset_boundaries)
#     dsObj.create_nonlabelled_test_laz(subfolder_list)
    
# if __name__ == "__main__":
#     main()


    