import os
import laspy

class LazFileWriter:
    def __init__(self, parameters):
        filename = parameters['point_cloud_filename'].replace('\\', '/')
        directory = os.path.dirname(os.path.realpath(filename)).replace('\\', '/') + '/'             
        filename = filename.split('/')[-1]
        filename = filename.split('_')[0]

        parent_path = directory + filename + '_FT_output/'        
        self.parent_path = parent_path
        self.client_folder_path = self.create_subfolder()
        self.laz_file_paths = self.get_laz_contains_label()
        self.label_contained_laz_list = self.is_label_contain_laz()

    def create_subfolder(self):
        sub_folder_name = 'ToClient'
        client_folder_path = os.path.join(self.parent_path, 'ToClient')

        if not os.path.exists(client_folder_path):
            os.mkdir(client_folder_path)
        else:
            print("The folder {} is already there".format(sub_folder_name))

        return client_folder_path

    def get_laz_contains_label(self):
        all_contents = os.listdir(self.parent_path)
        laz_files = [file for file in all_contents if file.endswith('.laz')]
        laz_file_paths = [os.path.join(self.parent_path, file) for file in laz_files]

        return laz_file_paths

    def is_label_contain_laz(self):
        label_contained_laz_list = []

        for each_laz_file in self.laz_file_paths:
            las_file = laspy.read(each_laz_file)
            point_format = las_file.point_format
            field_names = list(point_format.dimension_names)

            if 'label' in field_names:
                label_contained_laz_list.append(each_laz_file)

        return label_contained_laz_list

    def write_laz_file_without_label(self):
        for each_laz_file in self.label_contained_laz_list:
            las_file_with_extension = os.path.basename(each_laz_file)
            las_file_without_extension = os.path.splitext(las_file_with_extension)[0]
            las_file_extension = os.path.splitext(las_file_with_extension)[-1]            
            name_for_client = '' # This is a string for indicating that this is a no label data for someone who needs, you can change it.
            new_laz_file_name = las_file_without_extension + name_for_client + las_file_extension
            new_laz_output_file_path = os.path.join(self.client_folder_path, new_laz_file_name)

            las_file = laspy.read(each_laz_file)
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

            out_las.write(new_laz_output_file_path)
            print("No label contained .laz file created:", new_laz_output_file_path)


# if __name__ == "__main__":
#     main()
