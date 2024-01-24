from abc import ABC
import torch
from torch_geometric.data import Dataset, DataLoader, Data
import numpy as np
import glob
import pandas as pd
from preprocessing import Preprocessing
from model import Net, Gpu8gbNet
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import os
import time
from tools import load_file, save_file
import shutil
import sys
sys.setrecursionlimit(10 ** 8)  # Can be necessary for dealing with large point clouds.


class TestingDataset(Dataset, ABC):
    def __init__(self, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + '*.npy')
        self.device = device
        self.points_per_box = points_per_box

    def len(self):
        return len(self.filenames)

    def get(self, index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:, :3]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos = pos - local_shift
        data = Data(pos=pos, x=None, local_shift=local_shift)
        
        return data


def choose_most_confident_label(point_cloud, original_point_cloud):
    """
    Args:
        original_point_cloud: The original point cloud to be labeled.
        point_cloud: The segmented point cloud (often slightly downsampled from the process).

    Returns:
        The original point cloud with segmentation labels added.
    """

    print("Choosing most confident labels...")
    # with config_context(target_offload="gpu:0"):
    # This is important stage for labeling as a prediction
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', metric='euclidean', radius=0.05, n_jobs=4).fit(point_cloud[:, :3])
    _, indices = neighbours.kneighbors(original_point_cloud[:, :3])

    # print("finding median label value")
    labels = np.zeros((original_point_cloud.shape[0], 5))
    # TODO: to reduce memory requirements calculate the median label separately from the median coordinates
    labels[:, :4] = np.median(point_cloud[indices][:, :, -4:], axis=1)
    labels[:, 4] = np.argmax(labels[:, :4], axis=1)

    # original_point_cloud = np.hstack((original_point_cloud, labels[:, 4:]))
    # return original_point_cloud
    return labels[:,4]

def force_cudnn_initialization():
    # *** Aglika - performs a mock convolution which clears up 
    # PyTorch initializes cuDNN lazily whenever a convolution is executed for the first time. 
    # However, in my case there was not enough GPU memory left to initialize cuDNN because PyTorch 
    # itself already held the entire memory in its internal cache. One can release the cache 
    # manually with "torch.cuda.empty_cache()" right before the first convolution that is executed. 
    # A cleaner solution is to force cuDNN initialization at the beginning by doing a mock convolution:
    #***
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

class SemanticSegmentation:
    def __init__(self, parameters):
        self.sem_seg_start_time = time.time()
        self.parameters = parameters
        if not self.parameters['use_CPU_only']:
            print('Is CUDA available?', torch.cuda.is_available())
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            
        else:
            self.device = torch.device('cpu')

        print("Performing inference on device:", self.device)
        if not torch.cuda.is_available():
            print("Please be aware that inference will be much slower on CPU. An Nvidia GPU is highly recommended.")
        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/'
        self.filename = self.filename.split('/')[-1]
        self.filename = self.filename.split('_')[0]
        self.filename = self.filename.replace('.laz','')
        plotID = self.filename

        self.output_dir = self.directory + self.filename + '_FT_output/'
        self.working_dir = self.directory + self.filename + '_FT_output/working_directory/'

        self.filename = 'working_point_cloud.laz'
        self.directory = self.output_dir
        self.plot_summary = pd.read_csv(self.output_dir + 'plot_summary.csv', index_col=None)
        # self.plot_summary = pd.read_csv(self.output_dir + plotID + '_plot_summary.csv', index_col=None)
        self.plot_centre = [[float(self.plot_summary['Plot Centre X'].item()), float(self.plot_summary['Plot Centre Y'].item())]]

    def inference(self):
        # 1. 0. Data preparation
        # 1. 1. List up the small cubic .npy data
        test_dataset = TestingDataset(root_dir=self.working_dir,
                                      points_per_box=self.parameters['max_points_per_box'],
                                      device=self.device)
        # 1. 2. Tensorising for loading using listed small cubic npy data based on the number of batch size
        test_loader = DataLoader(test_dataset, batch_size=self.parameters['batch_size'], shuffle=False,
                                 num_workers=0)

        # 1. 3. Kill the Torch cache and GPU memory
        force_cudnn_initialization()  # Aglika:  added this to clean the PyTorch cache and free up GPU memory TODO take a course
        
        # 2. 0. Model preparation
        # 2. 1. Set the number of class and GPU usage 
        model = Net(num_classes=4).to(self.device) # This os for normal net
        # model = Gpu8gbNet(num_classes=4).to(self.device) # This is for small net
        
        # 2. 2. Loading the existing model depends on GPU usage
        if self.parameters['use_CPU_only']:
            model.load_state_dict(torch.load('../model/' + self.parameters['model_filename'], map_location=torch.device('cpu')), strict=False)
        else:
            # model.load_state_dict(torch.load('../model/' + self.parameters['model_filename']), strict=False)
            model.load_state_dict(torch.load(self.parameters['model_filename']), strict=False)
        
        # 2. 3. Let the model prepare the evaluation stage
        model.eval()
        
        # 3. 0. Inference against each .npy data        
        num_boxes = test_dataset.len()
        with torch.no_grad():

            self.output_point_cloud = np.zeros((0, 3 + 4)) # xyz and labels' voting
            output_list = []
            for i, data in enumerate(test_loader):
                if (i % 100) == 0 :
                    print('\r', str(i * self.parameters['batch_size']),'/', str(num_boxes))
                data = data.to(self.device) # data will be on GPU computation, this is useless becuase it is already applied under GPU                
                out = model(data) # This is the moment of inference against for each .npy data
                out = out.permute(2, 1, 0).squeeze() # The value of tensor can be shown by cli --> 'out[data.batch.cpu()]'
                batches = np.unique(data.batch.cpu())
                out = torch.softmax(out.cpu().detach(), axis=1)
                pos = data.pos.cpu()
                output = np.hstack((pos, out)) # The value of tensor can be shown by cli --> 'output[data.batch.cpu()]'
                
                for batch in batches:
                    outputb = np.asarray(output[data.batch.cpu() == batch])
                    outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]
                    output_list.append(outputb)

            self.output_point_cloud = np.vstack(output_list) # This is the process that the each output from .npy becomes stacked for merging every output.
            print('\r',str(num_boxes),'/',str(num_boxes),'\r')

        del outputb, out, batches, pos, output  # clean up anything no longer needed to free RAM.

        # From here, original cloud point data will be loaded to prepare the label properly.
        headers_of_interest = ['x', 'y', 'z','classification','intensity','return_number','gps_time','Ring','Range']
        original_point_cloud, headers = load_file(self.directory + self.filename, headers_of_interest=headers_of_interest)
        # the original point cloud was shifted before the semantic segmentation - shift again for choosing the label
        original_point_cloud[:, :2] = original_point_cloud[:, :2] - self.plot_centre

        # Aglika - use NN to find assign segmentation label to the original point cloud
        # self.output_point_cloud is subsampled and labeled and is used to propagate the labels to the original point cloud
        # self.output = choose_most_confident_label(self.output_point_cloud, original_point_cloud[:,:3])  # passing only the important information
                                                                                                        # for memory sake
        # The moment for predicting labels
        labels = choose_most_confident_label(self.output_point_cloud, original_point_cloud[:,:3])
        
        self.output = np.hstack((original_point_cloud[:,:3], np.atleast_2d(labels).T))
        self.output = np.hstack((self.output, original_point_cloud[:, 3:]))  # attach back the remaining attributes 
        headers=['x','y','z','label'] + headers[3:]  # list concatenation - HOW AWESOME

        self.output = np.asarray(self.output, dtype='float64')
        # global shift back, see above
        self.output[:, :2] = self.output[:, :2] + self.plot_centre

        _, indices= np.unique(self.output[:,:3], axis=0, return_index=True) 
        self.output = self.output[indices,:] 

        xyz_offsets = [self.plot_summary['Offset X'][0], self.plot_summary['Offset Y'][0], self.plot_summary['Offset Z'][0]]
        save_file(self.output_dir + 'segmented_raw.laz', self.output, headers_of_interest=headers, offsets=xyz_offsets)

        self.sem_seg_end_time = time.time()
        self.sem_seg_total_time = self.sem_seg_end_time - self.sem_seg_start_time
        self.plot_summary['Semantic Segmentation Time (s)'] = self.sem_seg_total_time
        self.plot_summary.to_csv(self.output_dir + 'plot_summary.csv', index=False)
        print(f"Semantic segmentation took {np.around(self.sem_seg_total_time/60,2)} min")
        print("Semantic segmentation done")
        if self.parameters['delete_working_directory']:
            shutil.rmtree(self.working_dir, ignore_errors=True)