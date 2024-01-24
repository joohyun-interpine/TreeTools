import laspy
import os
import sys
import math
import pandas as pd
from tabulate import tabulate
import numpy as np
pd.set_option('display.max_colwidth', None) # this is just to check a dataframe easily

class StraightnessScore:
    """
    Straightness(sweep) of the stem from each tree will be calculated throught the csv file that contains cylinders information,
    The csv files are belong to 'taper' folder which is one of output from TreeTools prediction.
    Thus, to use this script, 'taper' folder that has csv files should be prepared.
    
    This approach has applied geometrical mathmatics theories such as triangle rules, parametric equation etc..
    Mainly, straightness score will be calculated and sitting between 0 and 1, so 0 means perfectly straight and 1 means totally swept.
    Depending on each ratio which is straightness score, stem will classified into a certain class such as 8, 6, L, S, 3, 1
    """
    def __init__(self, parent_path):
        """
        Set the parent directory path

        Args:
            parent_path (str): directory path that contains 'taper' folder
        """
        self.parent_path = parent_path
     
    def read_csv(self, csv_file_path):
        """
        Read a csv file and load as Pandas data frame

        Args:
            csv_file_path (str): hte path of the csv file

        Returns:
            Pandas data frame: Data Frame
        """
        df = pd.read_csv(csv_file_path, sep=',')
        
        return df

    # Function to calculate distance between two points
    def calculate_distance(self, x1, y1, x2, y2):
        """
        Distance between two coordinates can be calcutated,
        this is using Euclidean distance(https://en.wikipedia.org/wiki/Euclidean_distance) 

        Args:
            x1 (float): x coordinate of the first point
            y1 (float): y coordinate of the first point
            x2 (float): x coordinate of the second point
            y2 (float): y coordinate of the second point

        Returns:
            distance (float): Euclidean distance
        """
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def point_on_line_at_height(self, start_coords, end_coords, target_height):
        """
           Finding a 3D coordinate that is sitting on the line at certain given height which is 'z' value between another two 3D points.        
        
           Using the theory of the parametric equation of a line.
           Refer to, (http://www.nabla.hr/PC-ParametricEqu1.htm)

        Args:
            start_coords (tuple): the start 3d coordinate point
            end_coords (tuple): the end 3d coordinate point
            target_height (float): the height 

        Returns:
            coordinate (tuple): coordinate that we want to find
        """
        x1, y1, z1 = start_coords
        x2, y2, z2 = end_coords
        
        # Solve for t when z = target_height
        t = (target_height - z1) / (z2 - z1)

        # Calculate the coordinates of the point on the line
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = target_height
        
        return (x, y, z) 
    
    def data_derivation(self, df):
        """
        In order to calculte the angle and distance, there are some information need to be derivated from the extisting variable values.
        So the location of the center of each cylider is not 
        Args:
            df (pandas dataframe): load a csv file

        Returns:
            pandas dataframe: 4 more variables['Relocate_X', 'Relocate_Y', 'Distance_From_Center', 'Radius'] have been added
        """
        bottom_center_x = 0
        bottom_center_y = 0
        df['Relocate_X'] = df['Bin_X'] - df['Base_X']
        df['Relocate_Y'] = df['Bin_Y'] - df['Base_Y']
        df['Distance_From_Center'] = ((df['Relocate_X'] - bottom_center_x)**2 + (df['Relocate_Y'] - bottom_center_y)**2).apply(math.sqrt)
        
        df['Radius'] = df['MA_Diameter'] / 2
        
        return df

    def get_furthest_cylinders(self, df, file_name):
        """
        Finding the furthest cylinder from the bottom cylider,
        If the top cylinder is the same distance like as the furthest, 
        then the top cyilder will be the furthest cylinder       

        Args:
            df (Pandas DataFrame): DataFrame that has derivated variables

        Returns:
            List: list that contains 
        """
        
        df_sorted = df.sort_values(by='Height_Bin') # Sorting by height of each cylinder
        max_distance_index = df_sorted['Distance_From_Center'].idxmax() # Get the last index number which will be the maximum value contained observation
        max_distance_row = df_sorted.loc[max_distance_index].to_frame().T # Using the index number, get the observation information as row 
        
        bottom_cylinder_row = df.nsmallest(1, 'Height_Bin') # Sorted by 'Height_Bin', then the smallest will be the bottom cylinder
        top_cylinder_row = df.nlargest(1, 'Height_Bin') # Sorted by 'Height_Bin', then the largest will be the top cylinder
        
        # Concatenate DataFrames with ignore_index set to False
        discovered_df = pd.concat([max_distance_row, bottom_cylinder_row, top_cylinder_row], ignore_index=False)
        discovered_sorted_df = discovered_df.sort_values(by='Height_Bin')
        
        bottom_xy_radius = tuple(discovered_sorted_df.iloc[0, df.columns.get_indexer(['Relocate_X', 'Relocate_Y', 'Distance_From_Center', 'Radius'])])
        top_xy_radius = tuple(discovered_sorted_df.iloc[-1, df.columns.get_indexer(['Relocate_X', 'Relocate_Y', 'Distance_From_Center', 'Radius'])])
        
        furthested_sorted_df = discovered_df.sort_values(by='Distance_From_Center')
        furthest_xy_radius = tuple(furthested_sorted_df.iloc[-1, df.columns.get_indexer(['Relocate_X', 'Relocate_Y', 'Distance_From_Center', 'Radius'])])
        
        bottom_top_distance_threshold = 0.3 # 30% of the radius is the threshold that if the any cyliner is far more than 30%, then it is swept
        furthest_top_distance_threshold = 0.1 # 10% of the radius is the threshold that if the any cyliner is far more than 10%, then it is swept
        
        bottom_top_swept_threshold = (bottom_xy_radius[-1] * 2) * bottom_top_distance_threshold
        furthest_top_swept_threshold = (furthest_xy_radius[-1] * 2) * furthest_top_distance_threshold
        
        if abs(bottom_xy_radius[0]) + bottom_top_swept_threshold < abs(top_xy_radius[0]) and abs(bottom_xy_radius[1]) + bottom_top_swept_threshold < abs(top_xy_radius[1]):

            if abs(furthest_xy_radius[0]) + furthest_top_swept_threshold < abs(top_xy_radius[0]) and abs(furthest_xy_radius[1]) + furthest_top_swept_threshold < abs(top_xy_radius[1]):

                if abs(furthest_xy_radius[0]) < abs(top_xy_radius[0]) and abs(furthest_xy_radius[1]) < abs(top_xy_radius[1]) and abs(furthest_xy_radius[-2]) < abs(top_xy_radius[-2]):

                    bottom_furthest_coor_radius = [bottom_xy_radius, furthest_xy_radius]
                    bottom_top_furthest_coor_radius = [bottom_xy_radius, top_xy_radius, furthest_xy_radius]
                    
                    return [discovered_sorted_df, bottom_furthest_coor_radius, bottom_top_furthest_coor_radius]
                else:
  
                    bottom_furthest_coor_radius = [bottom_xy_radius, furthest_xy_radius]
                    bottom_top_furthest_coor_radius = [bottom_xy_radius, top_xy_radius, furthest_xy_radius]
                    
                    return [discovered_sorted_df, bottom_furthest_coor_radius, bottom_top_furthest_coor_radius]
            else:
                print("{} seems prediction error that branch has been detected as stem, thus no straightness score will be calculated".format(file_name))
                return None
        else:
            if abs(furthest_xy_radius[0]) + furthest_top_swept_threshold < abs(top_xy_radius[0]) and abs(furthest_xy_radius[1]) + furthest_top_swept_threshold < abs(top_xy_radius[1]):
                if abs(furthest_xy_radius[0]) < abs(top_xy_radius[0]) and abs(furthest_xy_radius[1]) < abs(top_xy_radius[1]) and abs(furthest_xy_radius[-2]) < abs(top_xy_radius[-2]):
                    bottom_furthest_coor_radius = [bottom_xy_radius, furthest_xy_radius]
                    bottom_top_furthest_coor_radius = [bottom_xy_radius, top_xy_radius, furthest_xy_radius]
                    
                    return [discovered_sorted_df, bottom_furthest_coor_radius, bottom_top_furthest_coor_radius]
                else:
                    bottom_furthest_coor_radius = [bottom_xy_radius, furthest_xy_radius]
                    bottom_top_furthest_coor_radius = [bottom_xy_radius, top_xy_radius, furthest_xy_radius]
                    
                    return [discovered_sorted_df, bottom_furthest_coor_radius, bottom_top_furthest_coor_radius]
            else:
                bottom_furthest_coor_radius = [bottom_xy_radius, furthest_xy_radius]
                bottom_top_furthest_coor_radius = [bottom_xy_radius, top_xy_radius, furthest_xy_radius]
                
                return [discovered_sorted_df, bottom_furthest_coor_radius, bottom_top_furthest_coor_radius]

    def get_length_hypotenuse_xy(self, two_coordinates):
        """
        This is the top view aspect:
        Get the distnaces such as X length, Y length, and hypotenuse between the center of the bottom cylider and the center of the furthest cylinder.
        This is using the Euclidean distance 

        Args:
            two_coordinates (list): that contains 3 tuples that has coordinates of each cylinder

        Returns:
            Float: float values for distances
        """

        hypotenuse = self.calculate_distance(two_coordinates[0][0], two_coordinates[0][1], two_coordinates[1][0], two_coordinates[1][1])
        x_length = abs(two_coordinates[-1][0] - two_coordinates[0][0])
        y_length = abs(two_coordinates[-1][-1] - two_coordinates[0][-1])
        
        return hypotenuse, x_length, y_length    
        
    def get_theta(self, hypotenuse, x_length):
        """
        Using Triangle Rule (https://www.mathwarehouse.com/geometry/triangles/), 
        Apart from the 90 degree among 3 angles in triagle, calcualte the 2 angles such as cosine and sine from the given distances such as hyptenus and x length

        Args:
            hypotenuse (float): float number for distance
            x_length (float): float number for distance

        Returns:
            list: that contains cosine theta/degree, sine theta/degree
        """
        
        y_length = math.sqrt(hypotenuse**2 - x_length**2) # Calculate the Y_length by the given values such as hypotenuse and x_length
        cosine_theta = x_length / hypotenuse # Calculate the theta between the hypotenuse and x_length
        theta_degree = math.degrees(math.acos(cosine_theta)) # Convert to degree from the cosine theta
        
        sine_theta = y_length / hypotenuse # Calculate the theta between the hypotenuse and y_length
        sine_degree = math.degrees(math.acos(sine_theta)) # Convert to degree from the sine theta
        
        return [cosine_theta, theta_degree, sine_theta, sine_degree]
    
    def get_edge_coord(self, theta_cosine, top_cyl_coord, origin_coord):
        """
        Finding the edge point coordinates

        Args:
            theta_cosine (float): cosine theta
            top_cyl_coord (list): coordinate and radius
            origin_coord (list): coordinate and radius

        Returns:
            list: that contains where is the zone and coordinate
        """
        edge_x = theta_cosine * origin_coord[-1]        
        edge_y = math.sqrt((origin_coord[-1] ** 2) - (edge_x ** 2))
        
        if top_cyl_coord[0] < 0 and top_cyl_coord[1] >= 0:
            zone = 'zone 1'
            edge_coor = ((origin_coord[0] - edge_x), (origin_coord[1] + edge_y))            
        elif top_cyl_coord[0] >= 0 and top_cyl_coord[1] >= 0:
            zone = 'zone 2'
            edge_coor = ((origin_coord[0] + edge_x), (origin_coord[1] + edge_y))            
        elif top_cyl_coord[0] < 0 and top_cyl_coord[1] < 0:
            zone = 'zone 3'
            edge_coor = ((origin_coord[0] - edge_x), (origin_coord[1] - edge_y))            
        else:
            zone = 'zone 4'
            edge_coor = ((origin_coord[0] + edge_x), (origin_coord[1] - edge_y))            
        
        
        return [zone, edge_coor]
        
    def get_furthest_circumference_coordinates(self, cylinders_info, theta_phi_list):
        """
        Finding out the measuring starting point' coordinates of top cylinder and bottom finishing point' coordinate,
        however to find out where are them which are sitting some point that above the circumference coordinates, 
        it requires a certain angle that can indicate which direction, so the angle needs to be calcualted.        

        Args:
            cylinders_info (list): contains cyliders coordinate and radius
            theta_phi_list (list): contains cosine and sine theta/degree

        Returns:
            Pandas DataFrame: zone and coordinates
        """
        
        bottom_top_furthest_coordinates = cylinders_info[2] # Get the bottom and furthest cylider's info that contains coordinates and its radius
        
        # bottom_cylider contains coordinates and its radius
        bottom_cyl_coord = bottom_top_furthest_coordinates[0]        
        
        # top_cylider contains coordinates and its radius
        top_cyl_coord = bottom_top_furthest_coordinates[1]
        
        # furthest_cylider contains coordinates and its radius
        furthest_cyl_coord = bottom_top_furthest_coordinates[2]
        
        # furthest_cylider
        theta_cosine = theta_phi_list[0] # !!!!! Potentially has bug, because this only thinking the location of the furthest is at top right of the bottom location
        
        bottom_edge_coor = self.get_edge_coord(theta_cosine, top_cyl_coord, bottom_cyl_coord)
        top_edge_coor = self.get_edge_coord(theta_cosine, top_cyl_coord, top_cyl_coord)
        furthest_edge_coor = self.get_edge_coord(theta_cosine,  top_cyl_coord, furthest_cyl_coord)
        if top_edge_coor == furthest_edge_coor:
            cylinders_info_height_sorted = cylinders_info[0].sort_values(by='Height_Bin')
        
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Zone'] = bottom_edge_coor[0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Cylinder_Edge_X'] = bottom_edge_coor[1][0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Cylinder_Edge_Y'] = bottom_edge_coor[1][1]
            
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[1], 'Zone'] = furthest_edge_coor[0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[1], 'Cylinder_Edge_X'] = furthest_edge_coor[1][0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[1], 'Cylinder_Edge_Y'] = furthest_edge_coor[1][1]
            
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Zone'] = top_edge_coor[0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Cylinder_Edge_X'] = top_edge_coor[1][0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Cylinder_Edge_Y'] = top_edge_coor[1][1]
            
            df = cylinders_info_height_sorted
        else:
                
            cylinders_info_height_sorted = cylinders_info[0].sort_values(by='Height_Bin')
            
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Zone'] = bottom_edge_coor[0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Cylinder_Edge_X'] = bottom_edge_coor[1][0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[0], 'Cylinder_Edge_Y'] = bottom_edge_coor[1][1]
            
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Zone'] = top_edge_coor[0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Cylinder_Edge_X'] = top_edge_coor[1][0]
            cylinders_info_height_sorted.loc[cylinders_info_height_sorted.index[-1], 'Cylinder_Edge_Y'] = top_edge_coor[1][1]
            
            cylinders_info_distance_sorted = cylinders_info_height_sorted.sort_values(by='Distance_From_Center')
            
            cylinders_info_distance_sorted.loc[cylinders_info_distance_sorted.index[-1], 'Zone'] = furthest_edge_coor[0]
            cylinders_info_distance_sorted.loc[cylinders_info_distance_sorted.index[-1], 'Cylinder_Edge_X'] = furthest_edge_coor[1][0]
            cylinders_info_distance_sorted.loc[cylinders_info_distance_sorted.index[-1], 'Cylinder_Edge_Y'] = furthest_edge_coor[1][1]
        
            df = cylinders_info_distance_sorted
        
        return df

    def distances_add_from_edge(self, df):
        """
        Calculate the distance between the bottom edge point and each cylinder's edge point, because eacy cylinder has different radius

        Args:
            df (Pandas DataFrame): has drivated variables and values

        Returns:
            Pandas DataFrame: distances between cylinders will be added as one column
        """
        cylinders_info_height_sorted = df.sort_values(by='Height_Bin')
        bottom_cyl_edge_x = cylinders_info_height_sorted.iloc[0]['Cylinder_Edge_X']
        bottom_cyl_edge_y = cylinders_info_height_sorted.iloc[0]['Cylinder_Edge_Y']
        
        cylinders_info_height_sorted['Distance_From_CenterEdge'] = ((cylinders_info_height_sorted['Cylinder_Edge_X'] - bottom_cyl_edge_x)**2 + (cylinders_info_height_sorted['Cylinder_Edge_Y'] - bottom_cyl_edge_y)**2).apply(math.sqrt)
        
        return cylinders_info_height_sorted

    
    def get_final_furthest_coord(self, furthest_coord, parametric_coord_by_height, distance_length):
        """
        the direction vector of the line and scale it by the desired distance

        Args:
            furthest_coord (_type_): _description_
            parametric_coord_by_height (_type_): _description_
            distance_length (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Convert points to numpy arrays for easier vector operations
        a = np.array(parametric_coord_by_height)
        b = np.array(furthest_coord)

        # Calculate the direction vector
        direction_vector = b - a
        
        # Check if the direction vector is very close to zero
        if np.linalg.norm(direction_vector) < 1e-10:
        # Handle the case when the vector length is very small
        # You can choose an alternative approach or return a default value
            return tuple(a)    

        # Normalize the direction vector
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the new point along the line at the desired distance
        new_point = a + distance_length * normalized_direction

        return tuple(new_point)
    
    def get_length_hypotenuse_3d_coord(self, df):
        """
        Calculate the disstance between two 3D coordinates which are the top and furthest cylinders

        Args:
            dataframe (pandas df): contains bottom, furthest and top information about height, radius, coordinates

        Returns:
            furthest cylider info (dictionary): furthest cylider info
        """
        # coord1 and coord2 are tuples or lists representing 3D coordinates (x, y, z)
        topx1, topy1, topz1 = df.iloc[-1]['Cylinder_Edge_X'], df.iloc[-1]['Cylinder_Edge_Y'], df.iloc[-1]['Height_Bin'] 
        bottomx1, bottomy1, bottomz1 = df.iloc[0]['Cylinder_Edge_X'], df.iloc[0]['Cylinder_Edge_Y'], df.iloc[0]['Height_Bin'] 
        
        cylinders_info_distance_sorted = df.sort_values(by='Distance_From_CenterEdge')
        furthesstx1, furthessty1, furthesstz1 = cylinders_info_distance_sorted.iloc[-1]['Cylinder_Edge_X'], cylinders_info_distance_sorted.iloc[-1]['Cylinder_Edge_Y'], cylinders_info_distance_sorted.iloc[-1]['Height_Bin'] 
        
        if (topx1, topy1, topz1) == (furthesstx1, furthessty1, furthesstz1):
            furthesstx1, furthessty1, furthesstz1 = df.iloc[1]['Cylinder_Edge_X'], df.iloc[1]['Cylinder_Edge_Y'], df.iloc[1]['Height_Bin']
        else:
            pass
        
        # About the top and bottom cylinders
        d3_hypotenuse_between_top_bottom = math.sqrt((topx1 - bottomx1)**2 + (topy1 - bottomy1)**2 + (topz1 - bottomz1)**2)
        length_x_between_edges_top_bottom = df.iloc[-1]['Distance_From_CenterEdge']
        theta_phi_list = self.get_theta(d3_hypotenuse_between_top_bottom, length_x_between_edges_top_bottom)        
        
        # About the top and furthest cyliners        
        height_z_top_to_furthest = topz1 - furthesstz1
        if height_z_top_to_furthest == 0:
            height_z_top_to_furthest = topz1 * (1/10000)
            
        hypotenuse_top_to_furthest_cyl = height_z_top_to_furthest / math.cos(theta_phi_list[-1] * math.pi / 180)
        
        furthest_cyl_x_length = math.sqrt(hypotenuse_top_to_furthest_cyl**2 - height_z_top_to_furthest**2)
        
        furthest_cyl_point_on_line_coord = self.point_on_line_at_height((topx1, topy1, topz1), (bottomx1, bottomy1, bottomz1), furthesstz1)
        final_furthest_coord = self.get_final_furthest_coord((furthesstx1, furthessty1, furthesstz1), furthest_cyl_point_on_line_coord, furthest_cyl_x_length)
  
        furthest_cylinders_info_dict = {"furthest_origin_edge_coord": (furthesstx1, furthessty1, furthesstz1),
                                        "furthest_estimate_on_linecoord": final_furthest_coord                                       
                                        }
        return furthest_cylinders_info_dict
    
    def get_ratio(self, df, dict):
        """
        Calculate the ratio that how much stem is swept,
        it is being calculated by the method for measuring the sweep in forestry industry.
        (https://www.researchgate.net/figure/Technique-for-measuring-sweep_fig1_237332414)

        Args:
            df (pandas dataframe): contains bottom, furthest and top information about height, radius, coordinates
 
            furthest cylider info (dictionary): furthest cylider info

        Returns:
            ratio, height (float): sweep ratio and height of the tree (actually highest of the cylider)
        """
        cylinders_info_height_sorted = df.sort_values(by='Height_Bin')
        sed = cylinders_info_height_sorted.iloc[-1]['MA_Diameter'] # Assume that the top cylinder has 'SED' which is Small End Diameter
        height = cylinders_info_height_sorted.iloc[-1]['Height_Bin']
        distance = math.sqrt((dict['furthest_estimate_on_linecoord'][0] - dict['furthest_origin_edge_coord'][0])**2 + (dict['furthest_estimate_on_linecoord'][1] - dict['furthest_origin_edge_coord'][1])**2)
        ratio = distance / sed
        
        
        return ratio, height
    
    def get_csv(self, taper_path):
        files = os.listdir(taper_path)
        csv_list = []
        for file in files:
            file_path = os.path.join(taper_path, file)
            if file.endswith(".csv"):
                csv_list.append(file_path)
        
        return csv_list
    
    
    def classifier(self, taper_path, file_name, ratio, height):
        """
        Depending on the ratio, the stem will be classified into a certain class such as 8, 6, L, S, 3, 1

        Args:
            taper_path (str): the path where the 'taper' folder is 
            file_name (str): the csv file name and extension
            ratio (float): straightness score
            height (float): the tree height
        """
  
        class_six_height = 6
        class_four_height = 4
        
        class_eight_ratio = 1/8
        class_six_ratio = 1/6
        class_five_ratio = 1/5
        class_three_ratio = 1/3
        class_half_ratio = 1/2
        
        file_no_extension = os.path.splitext(file_name)[0]
        txt_file_path = os.path.join(taper_path, file_no_extension + ".txt")
        
        # Redirect sys.stdout to the output file
        with open(txt_file_path, 'w') as output_file:
            sys.stdout = output_file
        
            print("{}:\n\n==== The straightness score for tree {} is being assessed ==== \n".format(file_name, file_name))
            
            if ratio < class_half_ratio:
                print("{}: Sweep rate '{:.2f}' < {:.2f}".format(file_name, ratio, class_half_ratio))
                if height > class_six_height:
                    print("{}: Height '{}m' > {}m, Sweep rate '{:.2f}' < {:.2f}".format(file_name, height, class_six_height, ratio, class_half_ratio))
                    if ratio < class_eight_ratio:                    
                        print("{}: Class (8) - SED/8, Peeler grade".format(file_name))                    
                    elif class_eight_ratio <= ratio < class_six_ratio:
                        print("{}: Class (6) - SED/6, Under peeler grade, but still straight".format(file_name))
                    elif class_six_ratio <= ratio < class_five_ratio:
                        print("{}: Class (L) - SED/5, Long sawlogs".format(file_name))
                    else:
                        print("exception case for ratio??")
                        
                elif class_four_height < height <= class_six_height:
                    print("{}: {} < Height '{}m' <= {}m, Sweep rate '{:.2f}' < {:.2f}".format(file_name, class_four_height, height, class_six_height, ratio, class_half_ratio))
                    if ratio < class_five_ratio:
                        print("{}: Class (S) - SED/5, Sawlogs grade".format(file_name)) 
                    elif class_five_ratio <= ratio < class_three_ratio:
                        print("{}: Class (3) - SED/3, Rough sawlogs or pulp grade".format(file_name))  
                else:
                    print("{}: Height '{}m' <= {}m, Sweep rate '{:.2f}' < {:.2f}".format(file_name, height, class_four_height, ratio, class_half_ratio))                
                    print("{}: Height '{}m' is too low, it is possibly a regen tree".format(file_name, height))
            else:
                print("{}: Sweep rate '{:.2f}' > {:.2f}".format(file_name, ratio, class_half_ratio))            
                print("{}: Class (1) - SED/2, Sw OK for Pulp grade".format(file_name))
        
        sys.stdout = sys.__stdout__
        print(f"Output saved to {txt_file_path}")
        
    def assess_under_folder(self, ssObj, taper_path):
        """
        using this code for multiple taper csv files that belong to a folder,
        then using this fucntion to calcuate the entire taper csv files,
        it will create result text files that show the class of the stem

        Args:
            ssObj (StraightnessScore Class): as an object
            taper_path (str): the path where the 'taper' folder is 
        """
        # ### Folder entire --------------------------------------  
        csv_list = ssObj.get_csv(taper_path)
        
        for each_tree_taper in csv_list:
            
            file_name = os.path.split(each_tree_taper)[-1]            
            
            df = ssObj.read_csv(each_tree_taper)
            
            # From this, top of 2d view
            derivated_df = ssObj.data_derivation(df)
            cylinders_info = ssObj.get_furthest_cylinders(derivated_df, file_name)
            if cylinders_info != None:
                hypotenuse, x_length, y_length = ssObj.get_length_hypotenuse_xy(cylinders_info[1])        
                theta_phi_list = ssObj.get_theta(hypotenuse, x_length)
                cylinders_edge_coords_df = ssObj.get_furthest_circumference_coordinates(cylinders_info, theta_phi_list)
                cylinders_info_height_sorted_df = ssObj.distances_add_from_edge(cylinders_edge_coords_df)
                
                # From this, side of 3d view
                furthest_cylinders_info_dict = ssObj.get_length_hypotenuse_3d_coord(cylinders_info_height_sorted_df)    
                ratio, height = ssObj.get_ratio(cylinders_info_height_sorted_df, furthest_cylinders_info_dict)
                
                ssObj.classifier(taper_path, file_name, ratio, height)            
                
            else:
                file_no_extension = os.path.splitext(file_name)[0]
                txt_file_path = os.path.join(taper_path, file_no_extension + "_NoScore.txt")      
                          
                with open(txt_file_path, 'a') as file:
                    # Append some text to the file                    
                    file.write("{}:\n\n==== The straightness score for tree {} is being assessed ==== \n".format(file_name, file_name))
                    file.write("\nPotentially, the branch has been detected as a stem, so straightness score has not been calculated.".format(file_name))
                    file.write("\nPlease have a look manually")
                continue

        
    def assess_single_file(self, ssObj, taper_path, csv_file_path):
        """
        using this code for single taper csv files,        
        it will create result text files that show the class of the stem

        Args:
            ssObj (StraightnessScore Class): as an object
            taper_path (str): the path where the 'taper' folder is 
            csv_file_path (str): the file path where the specific taper csvfile
        """                
        # ### Single Data    
        file_name = os.path.split(csv_file_path)[-1]
        df = ssObj.read_csv(csv_file_path)
        
        # From this, top of 2d view
        derivated_df = ssObj.data_derivation(df)
        cylinders_info = ssObj.get_furthest_cylinders(derivated_df, file_name) # cylinders_info[1] has the coordinates of the bottom and furthest cylider
        hypotenuse, x_length, y_length = ssObj.get_length_hypotenuse_xy(cylinders_info[1]) # get the distances between the center of bottom cylinder and the center of furthest cylider from the top view
        theta_phi_list = ssObj.get_theta(hypotenuse, x_length)
        cylinders_edge_coords_df = ssObj.get_furthest_circumference_coordinates(cylinders_info, theta_phi_list)
        cylinders_info_height_sorted_df = ssObj.distances_add_from_edge(cylinders_edge_coords_df)
        
        # From this, side of 3d view
        furthest_cylinders_info_dict = ssObj.get_length_hypotenuse_3d_coord(cylinders_info_height_sorted_df)    
        ratio, height = ssObj.get_ratio(cylinders_info_height_sorted_df, furthest_cylinders_info_dict)
        ssObj.classifier(taper_path, file_name, ratio, height)
        
         
def main():
    
    parent_path = r'C:\Users\JooHyunAhn\Interpine\GitRepos\3d-pointcloud-materials\StraightnessScore\SampleData\Plot95_FT_output'
    taper_path = r'C:\Users\JooHyunAhn\Interpine\GitRepos\3d-pointcloud-materials\StraightnessScore\SampleData\Plot95_FT_output\taper'    
    csv_file_path = r'C:\Users\JooHyunAhn\Interpine\GitRepos\3d-pointcloud-materials\StraightnessScore\SampleData\Plot95_FT_output\taper\Plot95_27.csv'    
    
    ssObj = StraightnessScore(parent_path)    
    
    # # ### Folder entire --------------------------------------  
    ssObj.assess_under_folder(ssObj, taper_path)
    
    # # # ### Single csv file --------------------------------------  
    # ssObj.assess_single_file(ssObj, taper_path, csv_file_path)
    
 
if __name__ == "__main__":
    main()