# Testing assigning ID by bearing
import numpy as np
import threading
from copy import deepcopy
import pandas as pd
import laspy
# from laspy import lasappender
import traceback
import sys
import subprocess
import os
from scipy import ndimage, misc
import matplotlib.pyplot as plt

## test median filter in 2d
a=misc.__all__
print(a[:])
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = misc.ascent()
# ascent = misc.electrocardiogram()
# ascent = misc.face()
result = ndimage.median_filter(ascent, size=5)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

# test Pandas
workdir = 'L:/APP_hovermap/A212T1A_FT_output/'

# df = pd.read_csv('L:/APP_hovermap/APP_220304/FSCT_work/E204T1A_FT_output/tree_data.csv', index_col='CCI_at_BH')
df = pd.read_csv(workdir + 'tree_data.csv',delimiter=',')
# df= pd.read_csv("lettera.csv", skiprows=1,
newdf = df[df.columns[1:-7]]
newdf.to_csv(workdir + 'temp.csv', index = False)
col_names=df.columns[1:-7]
row_list = df[col_names].values.tolist()    
# print(row_list[1,:])

# exit()


#  Add tree_id attribute to C2_0_0.las 
#
filename2 = workdir + 'C2_0_0.laz' 
las2 = laspy.read(filename2)

# for dim in las2.point_format.dimensions:
#     print(dim.name)
# print(str(len(las2.points)))

mean_gps_time = np.mean(las2.gps_time)           # save this for the purposes of merging
newlas = laspy.LasData(las2.header)
newlas.points = las2.points.copy()
newlas.add_extra_dim(laspy.ExtraBytesParams(name='tree_id', type='u1'))
setattr(newlas, 'tree_id', [1]*len(las2.points))                                        # set the tree_ids to 1  
# newlas.user_data = [1]*len(las2.points)
newlas.write(workdir + 'full.laz')
print('full.las saved')
del newlas

# Add classification 8 for the DTM grid
# 
las = laspy.read(workdir + 'DTM.laz')
las.classification  = [8]*len(las.points)
las.write(workdir + 'DTM_1.laz')
print('DTM_1.las saved')
del las

# Add Ring and Range attributes to the tree_aware point cloud
# 
filename = workdir + 'tree_aware_cropped_point_cloud.laz' 
las = laspy.read(filename)
las_class = las.classification
indices = np.where(las_class == 0 ) # change classification of never classified (buffer) to 'unclassified'
las_class[indices] = 1
# newlas = laspy.LasData(las.header)   # we want to change the header because we are changing the attributes
# newlas = laspy.create(file_version = las.header.version, point_format=las.header.point_format)
newlas = laspy.create()
newlas.header.offsets = [0,0,0]      # Aglika
newlas.header.scales = [0.001, 0.001, 0.001]
newlas.classification = las_class
newlas.x = las.x
newlas.y = las.y
newlas.z = las.z
newlas.gps_time = [mean_gps_time.astype(float)]*len(las)

# Add Ring and Range attributes 
n = len(las.points)
for hdr in ['Ring', 'Range', 'label','height_above_dtm','tree_id']:  
    if hdr == 'Ring' :    
        newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))             
        newlas.Ring = [1]*n
    elif hdr == 'Range' :
        newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))
        newlas.Range =[1.]*n
    else :
        if hdr in ['height_above_dtm'] :
            newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='f4'))
        else:
            newlas.add_extra_dim(laspy.ExtraBytesParams(name=hdr, type='u1'))
        setattr(newlas, hdr, las[hdr])    
newlas.user_data = las['tree_id']
newlas = laspy.convert(newlas, point_format_id=7)
newlas.gps_time=[mean_gps_time]*len(newlas.points)
newlas.write(workdir + 'tree_ids.laz')
print('tree_ids.las saved')

cwd = os.getcwd()
os.chdir(workdir)
merge_command = 'C:/LAStools/bin/lasmerge.exe -i full.las tree_ids.las DTM_1.las -o merged.laz -olaz'
subprocess.call(merge_command)
os.chdir(cwd)

if 0:
    try:
        with laspy.open(workdir + 'full.laz', mode='a') as outlas:
            with laspy.open(workdir + 'tree_ids.laz') as inlas:
                for points in inlas.chunk_iterator(2_000_000):
                    outlas.append_points(points)

    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        print('Error in append las')
        print ("PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError     Info:\n" + str(sys.exc_info()[1]))    


# lasmerge.exe -i full.las tree_aware_cropped_point_cloud.las -change_classification_from_to 0 1 -o merged.laz   # done above
# lasmerge.exe -i full.las tree_ids.las DTM_1.las -o merged.laz

# # arrange csv files in vertical tree listings
# filename = 'L:/APP_hovermap/APP_220304/FSCT_work/E204T1A_test/tree_data.csv' 
# csv = pd.read_csv(filename, index_col=False)
# # csv = pd.read_csv("test.csv", skiprows=1)
# # use skiprows if you want to skip headers
# df_csv = pd.DataFrame(data=csv)
# transposed_csv = df_csv.T
# print(transposed_csv)



# You can basically think of zip() and izip() as transpose operation
# izip() avoids the immediate copying of the data, but will basically do the same.
import csv
# from itertools import izip
# a = izip(*csv.reader(open(filename, "rb")))
# csv.writer(open("temp.csv", "wb")).writerows(a)



if 0:
    ## Multi - Threading test
    # 
    g_count = 0

    def thread_main():
        global g_count
        for i in range(100000):
            g_count += 1

    threads = []

    for i in range(50):
        th = threading.Thread(target=thread_main)
        threads.append(th)

    for th in threads:
        th.start()

    for th in threads:
        th.join()

    print("g_count = {:,}".format(g_count))             