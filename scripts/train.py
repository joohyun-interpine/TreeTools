from tools import load_file, save_file, get_fsct_path
from model import Net, Gpu8gbNet
from train_datasets import TrainingDataset, ValidationDataset
from fsct_exceptions import NoDataFound
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.empty_cache()
from torch_geometric.loader import DataLoader
# torch.backends.cudnn.benchmark = True
from datapreparation.data_preparation import DataPrep
import glob
import random
import threading
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=8192'
import shutil


# Last reviewed by Perry Han, don't change if you don't understand the code.

class TrainModel:
    def __init__(self, parameters): #initialize parameters. The __init__  function is called every time an object is created from a class.
        self.parameters = parameters

        if self.parameters["num_cpu_cores_preprocessing"] == 0:
            print("Pre-processing using default number of CPU cores (all of them).")
            self.parameters["num_cpu_cores_preprocessing"] = os.cpu_count()
            print("Pre-processing using ", self.parameters["num_cpu_cores_preprocessing"], "/", os.cpu_count(), " CPU cores.")

        if self.parameters["preprocess_train_datasets"]:
            self.preprocessing_setup("train")

        if self.parameters["preprocess_validation_datasets"]:
            self.preprocessing_setup("validation")

        self.device = parameters["device"]

    def check_and_fix_data_directory_structure(self, data_sub_directory):
        """
        Creates the data directory and required subdirectories in
        the FSCT directory if they do not already exist.
        """
        fsct_dir = get_fsct_path()
        dir_list = [
            os.path.join(fsct_dir, "data"),
            os.path.join(fsct_dir, "data", data_sub_directory),
            os.path.join(fsct_dir, "data", data_sub_directory, "sample_dir"),
        ]

        for directory in dir_list:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print(directory, "directory created.")

            elif "sample_dir" in directory and self.parameters["clean_sample_directories"]:
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory)
                print(directory, "directory created.")

            else:
                print(directory, "directory found.")

    def preprocessing_setup(self, data_subdirectory): #preprocess point cloud setup (call preprocess_point_cloud function)
        self.check_and_fix_data_directory_structure(data_subdirectory)
        point_cloud_list = glob.glob(get_fsct_path("data") + "/" + data_subdirectory + "/*.laz")
        if len(point_cloud_list) > 0:
            print("Preprocessing point clouds set up")
            for point_cloud_file in point_cloud_list:
                print(point_cloud_file)
                point_cloud, headers = load_file(point_cloud_file, headers_of_interest=["x", "y", "z", "label"])
                self.preprocess_point_cloud(point_cloud, get_fsct_path("data") + "/" + data_subdirectory + "/sample_dir/")

    @staticmethod
    def threaded_boxes(point_cloud, box_size, min_points_per_box, max_points_per_box, path, id_offset, point_divisions): #split point cloud into boxes and save them as numpy arrays
        box_size = np.array(box_size)
        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size
        i = 0
        pds = len(point_divisions)
        while i < pds:
            box = point_cloud
            box = box[
                np.logical_and(
                    np.logical_and(
                        np.logical_and(box[:, 0] >= box_centre_mins[i, 0], box[:, 0] < box_centre_maxes[i, 0]),
                        np.logical_and(box[:, 1] >= box_centre_mins[i, 1], box[:, 1] < box_centre_maxes[i, 1]),
                    ),
                    np.logical_and(box[:, 2] >= box_centre_mins[i, 2], box[:, 2] < box_centre_maxes[i, 2]),
                )
            ]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    indices = list(range(0, box.shape[0]))
                    random.shuffle(indices)
                    random.shuffle(indices)
                    box = box[indices[:max_points_per_box], :]
                    box = np.asarray(box, dtype="float32")
                np.save(path + str(id_offset + i).zfill(7) + ".npy", box)
            i += 1
        return 1

    def global_shift_to_origin(self, point_cloud):  #shift point cloud to origin (0,0,0)
        point_cloud_mins = np.min(point_cloud[:, :3], axis=0)
        point_cloud_maxes = np.max(point_cloud[:, :3], axis=0)
        point_cloud_ranges = point_cloud_maxes - point_cloud_mins
        point_cloud_centre = point_cloud_mins + 0.5 * point_cloud_ranges

        point_cloud[:, :3] = point_cloud[:, :3] - point_cloud_centre
        return point_cloud, point_cloud_centre

    def preprocess_point_cloud(self, point_cloud, sample_dir): #preprocess point cloud and save it as numpy arrays

        # Split the point cloud into boxes and save them as numpy arrays.
        def get_box_centre_list(point_cloud_mins, num_boxes_array):
            box_centre_list = []
            for (dimension_min, dimension_num_boxes, box_size_m, box_overlap) in zip(
                point_cloud_mins,
                num_boxes_array,
                self.parameters["sample_box_size_m"],
                self.parameters["sample_box_overlap"],
            ):
                box_centre_list.append(
                    np.linspace(
                        dimension_min,
                        dimension_min + (int(dimension_num_boxes) * box_size_m),
                        int(int(dimension_num_boxes) / (1 - box_overlap)) + 1,
                    )
                )
            return box_centre_list

        print("Pre-processing point cloud...")

        # Global shift the point cloud to avoid loss of precision during segmentation.
        point_cloud, _ = self.global_shift_to_origin(point_cloud) 

        point_cloud_mins = np.min(point_cloud[:, :3], axis=0) # calculate min and max of point cloud from (0,0,0)
        point_cloud_maxes = np.max(point_cloud[:, :3], axis=0)
        point_cloud_ranges = point_cloud_maxes - point_cloud_mins # calculate range of point cloud
        point_cloud_centre = point_cloud_mins + 0.5 * point_cloud_ranges # calculate centre of point cloud, it should be (0,0,0)

        num_boxes_array = np.ceil(point_cloud_ranges / self.parameters["sample_box_size_m"]) # calculate number of boxes in each dimension
        box_centres = np.vstack(np.meshgrid(*get_box_centre_list(point_cloud_mins, num_boxes_array))).reshape(3, -1).T # calculate centre of each box

        # create list of lists to store points to be assigned to each thread
        point_divisions = [] 
        for thread in range(self.parameters["num_cpu_cores_preprocessing"]):
            point_divisions.append([])

        points_to_assign = box_centres 

        # assign points to each thread
        while points_to_assign.shape[0] > 0: 
            for i in range(self.parameters["num_cpu_cores_preprocessing"]):
                point_divisions[i].append(points_to_assign[0, :])
                points_to_assign = points_to_assign[1:]
                if points_to_assign.shape[0] == 0:
                    break

        threads = []
        id_offset = 0 # set id_offset to 0
        training_data_list = glob.glob(sample_dir + "*.npy") # get list of all files in sample_dir
        if len(training_data_list) > 0: # if there are files in sample_dir, get the last file name and add 1 to it
            id_offset = np.max([int(os.path.basename(i).split(".")[0]) for i in training_data_list]) + 1

        # create threads to process each box in parallel (call threaded_boxes function)
        for thread in range(self.parameters["num_cpu_cores_preprocessing"]):
            for t in range(thread):
                id_offset = id_offset + len(point_divisions[t])
            t = threading.Thread(
                target=self.threaded_boxes,
                args=(
                    point_cloud,
                    self.parameters["sample_box_size_m"],
                    self.parameters["min_points_per_box"],
                    self.parameters["max_points_per_box"],
                    sample_dir,
                    id_offset,
                    point_divisions[thread],
                ),
            )
            threads.append(t)

        for x in threads:
            x.start()

        for x in threads:
            x.join()

    def update_log(self, epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc): # update training history
        self.training_history = np.vstack(
            (self.training_history, np.array([[epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc]])))
        try:
            np.savetxt(os.path.join(get_fsct_path("model"), "training_history.csv"), self.training_history)
        except PermissionError:
            print("training_history not saved this epoch, please close training_history.csv to enable saving.")
            try:
                np.savetxt(
                    os.path.join(get_fsct_path("model"), "training_history_permission_error_backup.csv"),
                    self.training_history,)
            except PermissionError:
                pass

    def run_training(self): # main function to run training
        # Set up print of the device to use for training.
        if self.parameters["device"] == "cpu":
            if self.parameters["num_cpu_cores_deep_learning"] == 0:
                print("Using default number of CPU cores (all of them).")
                self.parameters["num_cpu_cores_deep_learning"] = os.cpu_count()
            print("Running deep learning using ", self.parameters["num_cpu_cores_deep_learning"],"/", os.cpu_count()," CPU cores.")
        else:
            print("Running deep learning using GPU",)

        self.training_history = np.zeros((0, 5))
        
        # Load the training data.
        train_dataset = TrainingDataset(
            root_dir=os.path.join(get_fsct_path("data"), "train/sample_dir/"),
            device=self.device,
            min_sample_points=self.parameters["min_points_per_box"],
            max_sample_points=parameters["max_points_per_box"],
        )
        if len(train_dataset) == 0:
            raise NoDataFound("No training samples found.")

        # Create the data loaders.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=parameters["train_batch_size"],
            shuffle=True,
            drop_last=True,
            # num_workers=self.parameters["num_cpu_cores_deep_learning"],
        )

        # Load the validation data if required.
        if self.parameters["perform_validation_during_training"]:
            # Load the validation data.
            validation_dataset = ValidationDataset(
                root_dir=os.path.join(get_fsct_path("data"), "validation/sample_dir/"),
                device=self.device,
            )
            if len(validation_dataset) == 0:
                raise NoDataFound("No validation samples found.")

            # Create the data loaders.
            self.validation_loader = DataLoader(
                validation_dataset,
                batch_size=parameters["validation_batch_size"],
                shuffle=True,
                drop_last=True,
                # num_workers=self.parameters["num_cpu_cores_deep_learning"],
            )
        
        # Load the model.
        # model = Net(num_classes=4).to(self.device)
        model = Gpu8gbNet(num_classes=4).to(self.device)
        if self.parameters["load_existing_model"]:
            print("Loading existing model...")
            try:
                model.load_state_dict(
                    torch.load(os.path.join(get_fsct_path("model"), self.parameters["model_filename"])),
                    strict=False,
                )

            except FileNotFoundError:
                print("File not found, creating new model...")
                torch.save(
                    model.state_dict(),
                    os.path.join(get_fsct_path("model"), self.parameters["model_filename"]),
                )

            try:
                self.training_history = np.loadtxt(os.path.join(get_fsct_path("model"), "training_history.csv"))
                print("Loaded training history successfully.")
            except OSError:
                pass

        model = model.to(self.device)

        # Set up the optimizer.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.parameters["learning_rate"])

        # Set up the loss function.
        criterion = nn.CrossEntropyLoss()
        val_epoch_loss = 0
        val_epoch_acc = 0

        best_val_acc = 0.0
        patience = 50  # Set your patience value
        no_improvement_count = 0  # Initialize counter for consecutive epochs with no improvement

        # Train the model.
        for epoch in range(self.parameters["num_epochs"]):
            print("=====================================================================")
            print("EPOCH ", epoch)
            
            # TRAINING
            model.train()
            running_loss = 0.0
            running_acc = 0
            i = 0
            running_point_cloud_vis = np.zeros((0, 5))
            for data in self.train_loader:
                data.pos = data.pos.to(self.device)
                data.y = torch.unsqueeze(data.y, 0).to(self.device)
                outputs = model(data)
                loss = criterion(outputs, data.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                running_point_cloud_vis = np.vstack(
                    (running_point_cloud_vis,
                    np.hstack((data.pos.cpu() + np.array([i * 7, 0, 0]), data.y.cpu().T, preds.cpu().T)),
                    )
                )
                if i % 20 == 0:
                    print(
                        "Train sample accuracy: ",
                        np.around(running_acc / (i + 1), 4),
                        ", Loss: ",
                        np.around(running_loss / (i + 1), 4),
                    )

                    if self.parameters["generate_point_cloud_vis"]:
                        save_file(
                            os.path.join(get_fsct_path("data"), "latest_prediction.las"),
                            running_point_cloud_vis,
                            headers_of_interest=["x", "y", "z", "label", "prediction"],
                        )
                i += 1
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = running_acc / len(self.train_loader)
            self.update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
            print("Train epoch accuracy: ", np.around(epoch_acc, 4), ", Loss: ", np.around(epoch_loss, 4), "\n")

            # VALIDATION
            print("Validation")
            if self.parameters["perform_validation_during_training"]:
                model.eval()
                running_loss = 0.0
                running_acc = 0
                i = 0
                for data in self.validation_loader:
                    data.pos = data.pos.to(self.device)
                    data.y = torch.unsqueeze(data.y, 0).to(self.device)

                    outputs = model(data)
                    loss = criterion(outputs, data.y)

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach().item()
                    running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                    if i % 50 == 0:
                        print(
                            "Validation sample accuracy: ",
                            np.around(running_acc / (i + 1), 4),
                            ", Loss: ",
                            np.around(running_loss / (i + 1), 4),
                        )

                    i += 1
                val_epoch_loss = running_loss / len(self.validation_loader)
                val_epoch_acc = running_acc / len(self.validation_loader)
                self.update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
                print("Validation epoch accuracy: ", np.around(val_epoch_acc, 4), ", Loss: ", np.around(val_epoch_loss, 4))
                
                name, extension = os.path.splitext(self.parameters["model_filename"])
                saved_model_name = name + '_Epoch' +str(epoch) + '_ValEpochAcc-' + str(round(val_epoch_acc,2)).replace('.', 'p') + extension
                # Check for improvement in validation accuracy
                if val_epoch_acc > best_val_acc:
                    best_val_acc = val_epoch_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(get_fsct_path("model"), saved_model_name)                        
                    )
                    no_improvement_count = 0  # Reset the counter since there is an improvement
                else:
                    no_improvement_count += 1

                # Early stopping condition
                if no_improvement_count >= patience:
                    print(f"No improvement for {patience} consecutive epochs. Early stopping.")
                    break
                    
def set_params():
    parameters = dict(
    preprocess_train_datasets=1, # If 1, the model will preprocess the training datasets. If 0, the model will use the preprocessed training datasets.
    preprocess_validation_datasets=1, # If 1, the model will preprocess the validation datasets. If 0, the model will use the preprocessed validation datasets.
    clean_sample_directories=0,  # Deletes all samples in the sample directories.
    perform_validation_during_training=1, # If 1, the model will perform validation during training. If 0, the model will only perform validation after training.
    generate_point_cloud_vis=0,  # Useful for visually checking how well the model is learning. Saves a set of samples called "latest_prediction.las" in the "FSCT/data/"" directory. Samples have label and prediction values.
    load_existing_model=0, # If 1, the model will load the model specified in the "model_filename" parameter. If 0, the model will start training from scratch.
    num_epochs=100, # The number of epochs to train the model for.
    learning_rate=0.000025, # don't change if you don't understand how it will affect the model
    input_point_cloud=None, # If you want to use a custom point cloud, set this to the path of the point cloud. If you want to use the FSCT dataset, set this to None.
    model_filename="DecModel.pth", # The model will be saved as this filename in the "FSCT/model/" directory.
    sample_box_size_m=np.array([6, 6, 6]), # The size of the sample boxes in meters. The model will be trained on these boxes.
    sample_box_overlap=[0.5, 0.5, 0.5], # The overlap between sample boxes. The model will be trained on these boxes.
    min_points_per_box=1000, # The minimum number of points that must be in a sample box for it to be used for training.
    max_points_per_box=200000, # The maximum number of points that can be in a sample box for it to be used for training.
    subsample=False, # If True, the model will subsample the point cloud before training. This can speed up training, but can also reduce accuracy.
    subsampling_min_spacing=0.025, # The minimum spacing between points in the subsampled point cloud.
    num_cpu_cores_preprocessing=0,  # 0 Means use all available cores.
    num_cpu_cores_deep_learning=0,  # Setting this higher can cause CUDA issues on Windows.
    train_batch_size=2, # The number of sample boxes that will be used for training at a time.
    validation_batch_size=2, # The number of sample boxes that will be used for validation at a time.
    device="cuda",  # set to "cuda" or "cpu"
    )
    
    return parameters
    
if __name__ == "__main__":
    # dataPreObj = DataPrep()
    # dataPreObj.removing_ambigous_area()
    parameters = set_params()
    run_training = TrainModel(parameters)
    run_training.run_training()