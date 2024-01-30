import laspy
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.integrate import quad
import textwrap



class DataRemovingAmbigousArea:
    """
    In the 3d point data plot for a group of trees, there is an ambigous height range which is mixed with stems and canopy,
    this will cause a possibility that the 3d point cloud deep learning model confusing, so to avoid it,
    the ambigous height range need to be removed in the dataset.
    But, depending on demanding, dropped data not going t o be deleted, will be stored in a separated folder.
    """
    def __init__(self, path):
        self.path = path
        self.parent_path = os.path.split(self.path)[0]
        self.filename_extension = os.path.split(self.path)[1]
        self.filename = os.path.splitext(os.path.split(self.path)[1])[0]
        self.fileextension = os.path.splitext(os.path.split(self.path)[1])[-1]
               
        
    def get_data(self, canopy = 2, stem = 4):
        """
        From .laz file, 
        - total z values which are heights values will be extracted as a numpy array
        - another z values that are belong to stem class '4' will be extracted as a numpy array
        - another z values that are belong to canopy class '2' will be extracted as a numpy array        

        Args:
            canopy (int, optional): please check the class number through CloudCompare
            stem (int, optional): please check the class number through CloudCompare

        Returns:
            numpy array: z values which is represented by heights of trees
        """
        # Read LAS file
        laz_file = laspy.read(self.path)

        # Access specific attributes
        z_scaled = laz_file.z  # ScaledArrayView
        height = np.array(z_scaled)
        label = laz_file.label

        # Get indices of points with the target label
        height_indices_stem = height[label == stem]
        height_indices_canopy = height[label == canopy]        
        
        return height, height_indices_stem, height_indices_canopy

    
    # # Calculate the size of the intersection
    # def min_pdf(self, dist_b, dist_1, dist_2):
    #     return min(norm.pdf(dist_b, np.mean(dist_1), np.std(dist_1)),
    #             norm.pdf(dist_b, np.mean(dist_2), np.std(dist_2)))
    
    
    def intersection_point(self, dist_b, dist_1, dist_2):
        """
        Find the intersection point        

        Args:
            dist_b (numpy 1 shape array): 'A' type of values from a spcific criterion
            dist_1 (numpy 1 shape array):  'A' type of values that belong to the dist_1
            dist_2 (numpy 1 shape array): 'A' type of values that belong to the dist_2

        Returns:
            function: intersection point
        """
        return norm.pdf(dist_b, np.mean(dist_1), np.std(dist_1)) - norm.pdf(dist_b, np.mean(dist_2), np.std(dist_2))

    def find_intersection(self, dist_1, dist_2):
        """
        Finding an intersection point between two distributions

        Args:
            dist_1 (numpy 1 shape array): 'A' type of values that belong to the dist_1
            dist_2 (numpy 1 shape array): 'A' type of values that belong to the dist_2

        Returns:
            float: a specific value that is one of the z value that coming from dist_b
        """
        # Initial guess for the intersection point
        initial_guess = 0.5 * (np.mean(dist_1) + np.mean(dist_2))        
        result = lambda dist_b: self.intersection_point(dist_b, dist_1, dist_2)        
        intersection_x = fsolve(result, initial_guess)[0]        
        
        return intersection_x
    
    def create_intersection_mask(self, x_range, dist_1, dist_2):
        """
        Find an intersection area and dataset between dist_1 pdf and dist_2 pdf,

        Args:
            x_range (numpy 1 shape array): the number of divided line of the plot for showing
            dist_1 (_type_): 'A' type of values that belong to the dist_1
            dist_2 (_type_): 'A' type of values that belong to the dist_2

        Returns:
            bool: mask is a boolean array which has True or False, if it is sitting inside of an intersectionm then True
            numpy array: Among the entire data, if the mask says True, it will return
        """
        # Create a mask for the intersection area
        intersection_mask = np.logical_and(dist_1 > 0, dist_2 > 0)

        # Extract the data points corresponding to the intersection area
        intersection_data_range = x_range[intersection_mask]
        
        return intersection_mask, intersection_data_range
        
    def comparison_selection(self, min_max_list):
        """
        Find a lower height and upper height that will be used as values for clipping out from the given .laz data.

        Args:
            min_max_list (list): [Stem IQR75, Canopy IQR25, Intersection IQR25,Intersection IQR75]

        Returns:
            float: lower and upper heights
        """

        stem_iqr75 = min_max_list[0]
        canopy_iqr25 = min_max_list[1]
        intersection_iqr25 = min_max_list[2]
        intersection_iqr75 = min_max_list[3]
        
        if stem_iqr75 < intersection_iqr25:
            min = intersection_iqr25
        else:
            min = stem_iqr75
        
        if canopy_iqr25 < intersection_iqr75:
            max = canopy_iqr25
        else:
            max = intersection_iqr75            
        
        return min, max
        
    def get_stats_plots(self, dist_b, dist_1, dist_2, titles_data = ['', ''], plot=True):
        """
        Calculate the stats from the dist_b, dist_1, dist_2, 
        then gives a comparison plot that has two different distributions.

        Args:
            dist_b (numpy 1 shape array): 'A' type of values from a spcific criterion
            dist_1 (numpy 1 shape array): 'A' type of values that belong to the dist_a
            dist_2 (numpy 1 shape array): 'A' type of values that belong to the dist_b
            titles_data ([str, str]): The string name of each distribution, ex ['Stem', 'Canopy'].
            plot (bool, optional): True or False. If True, then plot will be poped up on the monitor, False, it will be just saved in the folder

        Returns:
            Dictionary: Calculated stats are there
        """
        # Create a histogram with two datasets
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size here
        ax.hist([dist_1, dist_2], bins=30, density=True, color=['brown', 'darkgreen'], edgecolor='black', label=titles_data)

        # Plot the probability density functions
        xmin, xmax = plt.xlim()
        x_range = np.linspace(xmin, xmax, 200)
        pdf_dist_1 = norm.pdf(x_range, np.mean(dist_1), np.std(dist_1))
        pdf_dist_2 = norm.pdf(x_range, np.mean(dist_2), np.std(dist_2))
 
        ax.plot(x_range, pdf_dist_1, color='brown', linestyle='solid', linewidth=3, label="Stem PDF")
        ax.plot(x_range, pdf_dist_2, color='darkgreen', linestyle='solid', linewidth=3, label="Canopy PDF")
        
        intersection_x = self.find_intersection(dist_1, dist_2)
        # min_pdf_callable = lambda dist_b: self.min_pdf(dist_b, dist_1, dist_2)        
        # intersection_size, _ = quad(min_pdf_callable, xmin, xmax)        
        intersection_mask, intersection_height_range = self.create_intersection_mask(x_range, pdf_dist_1, pdf_dist_2)
        # Calculate the IQR from the intersection data
        intersection_iqr_size = np.percentile(intersection_height_range, 75) - np.percentile(intersection_height_range, 25)
        # Calculate and plot the 25% and 75% percentiles for intersection data
        percentile_25 = np.percentile(intersection_height_range, 25)
        percentile_75 = np.percentile(intersection_height_range, 75)
        
        # Calculate and plot the IQR for pdf_data1 (Veg)
        stem_iqr_size = np.percentile(pdf_dist_1, 75) - np.percentile(pdf_dist_1, 25)
        # Calculate and plot the IQR for pdf_data2 (Stem)
        canopy_iqr_size = np.percentile(pdf_dist_2, 75) - np.percentile(pdf_dist_2, 25)
        
        # Stem iqr
        iqr25_stem_pdf = np.percentile(dist_1, 25)
        median_stem_pdf = np.percentile(dist_1, 50)
        iqr75_stem_pdf = np.percentile(dist_1, 75)
        
        # Canopy iqr
        iqr25_canopy_pdf = np.percentile(dist_2, 25)
        median_canopy_pdf = np.percentile(dist_2, 50)
        iqr75_canopy_pdf = np.percentile(dist_2, 75)
        
        # Calculate the proportion below and above a given height range          
        min_max_list = [iqr75_stem_pdf, iqr25_canopy_pdf, percentile_25, percentile_75]
        min_height, max_height = self.comparison_selection(min_max_list)
        # Proportion below the given height range
        # proportion_below = np.mean(intersection_height_range < min_height)
        # Proportion above the given height range
        # proportion_above = np.mean(intersection_height_range > max_height)
        # Proportion between the given height range
        # proportion_between = 1 - (proportion_below + proportion_above)
        # Set the paths for saving plot image.png and stats value in txt
        plot_image_path = os.path.join('r', os.path.join(self.parent_path, self.filename)) + '.png'
        plot_text_path = os.path.join('r', os.path.join(self.parent_path, self.filename)) + '.txt'
        plot_removed_path = os.path.join(os.path.join('r', os.path.join(self.parent_path, self.filename)), self.fileextension)
        min_height_plot = np.min(dist_b)
        max_height_plot = np.max(dist_b)
        
        stats_dict = {"file_name": self.filename_extension,
                      "file_path": self.path,
                      "dist_1": dist_1,
                      "dist_2": dist_2,
                      "min_height_plot": min_height_plot,
                      "max_height_plot": max_height_plot,
                      "x_range": x_range,
                      "pdf_dist_1": pdf_dist_1,
                      "pdf_dist_2": pdf_dist_2,
                      "intersection_x": intersection_x,
                      "intersection_mask": intersection_mask,
                      "intersection_height_range": intersection_height_range,
                    #   "intersection_size": intersection_size,
                      "intersection_iqr_size": intersection_iqr_size,
                      "percentile_25": percentile_25,
                      "percentile_75": percentile_75,
                      "canopy_iqr_size": canopy_iqr_size,
                      "stem_iqr_size": stem_iqr_size,
                      "iqr25_stem_pdf": iqr25_stem_pdf,
                      "median_stem_pdf": median_stem_pdf,
                      "iqr75_stem_pdf": iqr75_stem_pdf,
                      "iqr25_canopy_pdf": iqr25_canopy_pdf,
                      "median_canopy_pdf": median_canopy_pdf,
                      "iqr75_canopy_pdf": iqr75_canopy_pdf,
                      "min_height": min_height,
                      "max_height": max_height,
                    #   "proportion_below": proportion_below,
                    #   "proportion_above": proportion_above,
                    #   "proportion_between": proportion_between,
                    #   "proportion_above": proportion_above,
                      "titles_data": titles_data,
                      "plot_image_path": plot_image_path,
                      "plot_text_path": plot_text_path,
                      "plot_removed_path": plot_removed_path
                      }
        
        # Plot the median lines of the each pdf: Stem and Canopy   
        ax.axvline(stats_dict["median_stem_pdf"], color='brown', linestyle='solid', linewidth=3, label='Stem Median Height: {:.2f}'.format(stats_dict["median_stem_pdf"]))
        ax.axvline(stats_dict["median_canopy_pdf"], color='darkgreen', linestyle='solid', linewidth=3, label='Canopy Median Height: {:.2f}'.format(stats_dict["median_canopy_pdf"]))
        
        # Plot the intersection height
        ax.axvline(stats_dict["intersection_x"], color='red', linestyle='solid', linewidth=3, label='Intersection Height: {:.2f}m'.format(stats_dict["intersection_x"]))
        # ax.scatter(stats_dict["intersection_height_range"], np.zeros_like(stats_dict["intersection_height_range"]), color='lightgreen', label='Intersection Area', alpha=0.6)
        # ax.axvline(stats_dict["percentile_25"], color='red', linestyle='dashed', linewidth=1, label='Intersection IQR25% Height: {:.2f}m'.format(stats_dict["percentile_25"]))
        # ax.axvline(stats_dict["percentile_75"], color='red', linestyle='dashed', linewidth=1, label='Intersection IQR75% Height: {:.2f}m'.format(stats_dict["percentile_75"]))
        # ax.axvline(stats_dict["iqr75_stem_pdf"], color='brown', linestyle='dashed', linewidth=1, label='Stem IQR75% Height: {:.2f}m'.format(stats_dict["iqr75_stem_pdf"]))
        # ax.axvline(stats_dict["iqr25_canopy_pdf"], color='darkgreen', linestyle='dashed', linewidth=1, label='Canopy IQR25% Height: {:.2f}'.format(stats_dict["iqr25_canopy_pdf"]))
        
        # Draw lines for the specified height range
        ax.axvline(stats_dict["min_height"], color='orange', linestyle='solid', linewidth=5, label='Nominated Min Height: {:.2f}'.format(stats_dict["min_height"]))
        ax.axvline(stats_dict["max_height"], color='orange', linestyle='solid', linewidth=5, label='Nominated Max Height: {:.2f}'.format(stats_dict["max_height"]))

        # Customize the plot
        ax.set_title("Ambiguous overlapped area of the '{}' plot".format(self.filename_extension), fontweight='bold', fontsize=16)
        
        ax.set_xlabel('Plot Height')
        ax.set_ylabel('Frequency / PDF')
        ax.legend()
        
        fig.savefig(stats_dict["plot_image_path"])
        
        if plot == True:
            plt.show()
        else:
            pass

        return stats_dict
    
    def cmd_executor(self, las2las_path, stats_dict):
        """
        Create command string that will execute the las2las.exe for clip the 3d point cloud data (.laz) with the given height.        

        Args:
            las2las_path (str): A path where the las2las.exe file exist
            stats_dict (dictionary): all stats are there as a dictionary

        Returns:
            str: nominated_path
            str: dropped_path
            str: nominated_name
            str: dropped_name
        """
        lower_height = stats_dict["min_height"]
        upper_height = stats_dict["max_height"]

        nominated_zone = '_nominated'
        dropped_zone = '_dropped'
        dropped_cmd = las2las_path  + " -i " + self.path + " -drop_z_below " + str(lower_height) + " -drop_z_above " + str(upper_height) + " -odix " + dropped_zone + " -odir " + self.parent_path + " -olaz"
        nominated_cmd = las2las_path  + " -i " + self.path + " -drop_z " + str(lower_height) + " " + str(upper_height) + " -odix " + nominated_zone + " -odir " + self.parent_path + " -olaz"        
        
        os.system(nominated_cmd)
        # Confirmation message
        print(f"Nominated plot has been written")
        
        os.system(dropped_cmd)
        print(f"Dropped plot has been written")
        
        nominated_name = self.filename + nominated_zone + self.fileextension
        nominated_path = os.path.join(self.parent_path, nominated_name)
        dropped_name = self.filename + dropped_zone + self.fileextension
        dropped_path = os.path.join(self.parent_path, dropped_name)
        
        return nominated_path, dropped_path, nominated_name, dropped_name
    
    def abstract(self):
        """
        This will be an intro of the data summary.txt
        """
        
        intro = f"""
        Data Summary: Removing Ambiguous Area

        In the process of preparing data for training a 3D point cloud model capable of accurately segmenting points into distinct classes, 
        it is essential to address areas where canopy and stem features overlap. 
        These overlapping regions, referred to as ambiguous areas, pose a challenge for proper segmentation.
        
        In this process, statistical methodologies have been applied for finding minimum and maximum heights.
        Please refer to Interquartile range (aka. IQR) <https://en.wikipedia.org/wiki/Interquartile_range>

        The following is a summary of the data removal process:
        -------------------------------------------------------------------------------------------------------------------------------------
        """
        
        return intro
    
    def write_stats(self, stats_dict):
        """
        Create txt file that will contain useful info about the data and reason how to clip the ambigous area

        Args:
            stats_dict (dictionary): all stats are there as a dictionary

        Returns:
            str: a path where the txt file will be stored
        """
        # Specify the file path
        txt_file_path = stats_dict["plot_text_path"]

        # Open the file in write mode ('w')
        with open(txt_file_path, 'w') as f:
            abstract_text = textwrap.dedent(self.abstract()).strip()
            f.write(abstract_text + "\n\n")
            
            f.write("- Original plot '{}' comes from: {}\n".format(stats_dict["file_name"], stats_dict["file_path"]))
            # f.write("- Ambigous area removed plot is stored at: {}\n".format(stats_dict["plot_removed_path"]))
            f.write("- An interpretation of the probability density function is stored at: {}\n".format(stats_dict["plot_image_path"]))
            f.write("- This data summary is stored at: {}\n".format(stats_dict["plot_text_path"]))

            f.write("- The height range of this plot is ({:.2f}m ~ {:.2f}m)\n".format(stats_dict["min_height_plot"], stats_dict["max_height_plot"]))
            f.write("- The half of stem data is distributed in ({:.2f}m ~ {:.2f}m)\n".format(stats_dict["iqr25_stem_pdf"], stats_dict["iqr75_stem_pdf"]))
            f.write("- The half of canopy data is distributed in ({:.2f}m ~ {:.2f}m)\n".format(stats_dict["iqr25_canopy_pdf"], stats_dict["iqr75_canopy_pdf"]))
            
            f.write("- The intesection height between stem and canopy is {:.2f}m\n".format(stats_dict["intersection_x"]))
            f.write("- The half of overlapped area is in ({:.2f}m ~ {:.2f}m)\n".format(stats_dict["percentile_25"], stats_dict["percentile_75"]))            
            f.write("")
            f.write("- Considering the distribution of stem, canopy and intersection areathe lowest and highest heights will be {:.2f}m to {:.2f}m\n".format(stats_dict["min_height"], stats_dict["max_height"]))            
            f.write("")
        # Confirmation message
        print(f"Data Summary: Removing Ambiguous Area has been written to {txt_file_path}")
        
        return txt_file_path
    
    def get_num_points(self, las_path):
        """
        Get the total point number of the given .laz file

        Args:
            las_path (str): a path where the .laz file exist

        Returns:
            int: a number of the total point from the given .laz data
        """
        # Read the LAS file
        laz_file = laspy.read(las_path)

        # Get the number of points from the header
        num_points = laz_file.header.point_records_count

        return num_points
    
    def cal_ratio_dropped_points(self, nominated_path, dropped_path):
        """
        Calculate the ratio through the given  .laz data 
        
        Returns:
            list: [int, int, int, float]
        """               
        nominated_num_points = self.get_num_points(nominated_path)
        dropped_num_points = self.get_num_points(dropped_path)
        total_num_points = nominated_num_points + dropped_num_points
        ratio = dropped_num_points / total_num_points
        points_ratio_info = [nominated_num_points, dropped_num_points, total_num_points, ratio]
        
        return points_ratio_info
    
    def append_to_file(self, file_path, points_ratio_info):
        """
        Open the existing data summary.txt file, then add more info on top of that

        Args:
            file_path (str): a path where the existing txt file exist
            points_ratio_info (float): ratio
        """
        
        nominated_num_points, dropped_num_points, total_num_points, ratio = points_ratio_info
        formatted_nominated_num_points = "{:,}".format(nominated_num_points)
        formatted_dropped_num_points = "{:,}".format(dropped_num_points)
        formatted_total_num_points = "{:,}".format(total_num_points)
        # Get the current time
        current_time = datetime.now()
        # Format the time as a string
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
        lines_to_append = [
            f"",
            f"- Number of nominated points: {formatted_nominated_num_points}",
            f"- Number of dropped points: {formatted_dropped_num_points}",
            f"- Number of total points: {formatted_total_num_points}",
            f"- Ratio of dropped points out of total: {ratio:.2%}"
            f"",
            f"So, ambigous area's {formatted_dropped_num_points} is dropped, and {formatted_nominated_num_points} ({100% - ratio:.2%}) is nominated which will be used as training data"
            f"-------------------------------------------------------------------------------------------------------------------------------------"
            f"This data summary has been written on {formatted_time}"
        ]

        
        with open(file_path, 'a') as file:
            for line in lines_to_append:
                file.write(line + '\n')
                
        # Confirmation message
        print(f"Data Summary: Removing Ambiguous Area has been written")

    def create_folders_moving(self, nominated_path, dropped_path, nominated_name, dropped_name):
        """
        Once clipping process is done, then two datasets (dropped, nominated) will be moved to the its own folder for the next step.
        Depending on the usage, it can be used in the future as well

        Args:
            nominated_path (str): a path where the nominated .laz file will be stored
            dropped_path (str): a path where the dropped .laz file will be stored
            nominated_name (str): a string name of the nominated .laz file 
            dropped_name (str): a string name of the dropped .laz file 

        Returns:
            str: nominated_folder_path
            str: dropped_folder_path
        """
        nominated_folder_path = os.path.join(self.parent_path, 'nominated')
        dropped_folder_path = os.path.join(self.parent_path, 'dropped')
        
        # Creating for the nominated_folder_path
        if not os.path.exists(nominated_folder_path):
            os.mkdir(nominated_folder_path)
        else:
            print("The folder {} is already there".format(nominated_folder_path))
        
        
        dest_nominated_path = os.path.join(nominated_folder_path, nominated_name)
        cmd_nominated = "move " + nominated_path + " " + dest_nominated_path
        print(cmd_nominated + " is done")
        os.system(cmd_nominated)
            
        # Creating for the dropped_folder_path
        if not os.path.exists(dropped_folder_path):
            os.mkdir(dropped_folder_path)
        else:
            print("The folder {} is already there".format(dropped_folder_path))
        
        dest_dropped_path = os.path.join(dropped_folder_path, dropped_name)
        cmd_dropped = "move " + dropped_path + " " + dest_dropped_path
        print(cmd_dropped + " is done")
        os.system(cmd_dropped)
        
        return nominated_folder_path, dropped_folder_path

########### If you want to use this script separately from the data_praparation.py then, uncomment the below lines ################
        
# def main():
#     las2las_path = r'C:\LAStools\bin\las2las.exe'
#     path = r'C:\Users\JooHyunAhn\Interpine\DataSets\TreeTools_PlayGroundSet\removing_ambigous_area\mb17t1a_C2_0_0_hnom.laz'
#     dsObj = DataRemovingAmbigousArea(path)
#     height, height_indices_stem, height_indices_canopy = dsObj.get_data()
#     stats_dict = dsObj.get_stats_plots(height, height_indices_stem, height_indices_canopy, titles_data = ['Stem', 'Canopy'], plot=False)
#     nominated_path, dropped_path, nominated_name, dropped_name = dsObj.cmd_executor(las2las_path, stats_dict)
#     txt_file_path = dsObj.write_stats(stats_dict)
#     points_ratio_info = dsObj.cal_ratio_dropped_points(nominated_path, dropped_path)
#     dsObj.append_to_file(txt_file_path, points_ratio_info)
#     nominated_folder_path, dropped_folder_path = dsObj.create_folders_moving(nominated_path, dropped_path, nominated_name, dropped_name)
    
    
    
# if __name__ == "__main__":
#     main()

