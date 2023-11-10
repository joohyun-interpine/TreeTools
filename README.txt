To setup your environment:

(1) Install Python and Anaconda
(2) Install CUDA:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
(3) Pull repo RemoteSensing from GitHub
(4) Start an Anaconda prompt
(5) cd to C:\RemoteSensing\MLS\TreeTools\InterpineTools
(6a) Use the fsct_env.yml to create the fsctbase environment with the following command:

      conda env create -f fsct_env.yml

(6b) If step 4 does not finish successfuly (or you are using Linux/MacOS) use the portable yml file:

      conda env create -f fsct_env_portable.yml	

This will take a while.

(7) Activate the new environment:

      conda activate fsctbase

(8) Make sure the code will run on the GPU.

> python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 3090'

If cuda is not available you will have to fix its installation. For help go to: 
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

(9) Install wkhtmltopdf using its default settings. (https://wkhtmltopdf.org/downloads.html)

(10) Open scripts\run.py and check the parameters. At the least, edit the coordinates of the plot center, the radius and the plot radius buffer.

(11) Run run.py and follow the prompt. Currently FSCT can work only with las files. Complains can go to laspy.

(12) Check the output now and then. Make sure you have enough disk space - at least 5x the input las files
