from math import pi
# from mdutils.mdutils import MdUtils
# import markdown
# from mdutils import Html
import pandas as pd
import numpy as np
from tools import load_file, subsample_point_cloud, volume_from_taper, get_output_path
from matplotlib import pyplot as plt
import os
from scipy.spatial import ConvexHull
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import warnings
import sys
import shutil
import laspy
import pdfkit
from copy import deepcopy
from jinja2 import Environment, FileSystemLoader

QC='' # use QC='_QC' to generate plot reports from edited tables. Will look for {plotID}_QC_data.csv file
# QC='_Field'


def make_report(parameters):
    
    workdir, plotID = get_output_path(parameters['point_cloud_filename'])        
    filename = plotID + '.laz'        
    
    plot_summary = pd.read_csv(f'{workdir}/{plotID}_plot_summary.csv', index_col=False)
    total_processing_time = np.around(float(plot_summary['Total Run Time (s)'].item()))

    tree_data, plot_names = create_figures(workdir, plotID, plot_summary)
    plot_names.append(f"{plotID}_canopy.png")

    report_summary = get_report_statistics(plot_summary, tree_data)

    tree_data = tree_data.convert_dtypes()
    cols_to_keep = ['TreeNumber','TreeLocation_X','TreeLocation_Y','TreeLocation_Z','Distance','Bearing','Height','DBH_taper','CCI'] #'Crown_Height','Volume']
    tree_data_report = ['Tree ID','X','Y','Z','Dist','Bea','Ht (m)','DBH (mm)','CCI']   #'Crown_Ht','Volume'
    
    tree_data = tree_data[cols_to_keep]
    cols_dict = {cols_to_keep[i]: tree_data_report[i] for i in range(len(cols_to_keep))}
    tree_data.rename(columns = cols_dict, inplace=True)

    environment = Environment(loader=FileSystemLoader("../html_templates/"))
    template = environment.get_template("MLS_plot_report.html")

    content = template.render(
        plot_id = plotID,
        filename=filename,
        scan_date="",
        processing_time=f'{int(total_processing_time/60.)}:{int(total_processing_time%60.)}',
        report_summary = report_summary,
        tree_data = tree_data,
        canopy_plot = plot_names[2],
        dbh_height_plot = plot_names[1],
        dbh_map_plot = plot_names[0],
        tree_data_columns = tree_data.columns
        )

    with open(f"{workdir}/{plotID}.html", mode="w", encoding="utf-8") as report:
        report.write(content)

    print("HMTL report done") 

    # wkhtnltopdf 
    config = pdfkit.configuration(wkhtmltopdf = "C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")
    kitoptions = {
        "enable-local-file-access": None
    }
    pdfkit.from_file(f"{workdir}/{plotID}.html", f"{workdir}/{plotID}_Plot_Report{QC}.pdf", configuration = config, options = kitoptions)
    print(f'{QC} PDF report done') 
    # os.remove(workdir + 'Plot_Report.html')


def  get_report_statistics(plot_summary, tree_data) :

    if (plot_summary['Plot Radius'].item()) > 0:   
        plot_type = "Circular"
        plot_radius = plot_summary['Plot Radius'].item()
    else : 
        plot_type = "Bounding Box"
        # plot_radius = f'approx: {(maxX-minX)/2}'
        plot_radius = ""
    
    # backward compatibility
    if not 'Plot Centre Z' in plot_summary.columns :
        plot_summary.insert(loc=len(plot_summary.columns), column='Plot Centre Z', value=0.0)   

    if int(plot_summary['Offset X'].item()) < 100 :
        location_ref="centre"
        local_x = plot_summary['Plot Centre X'].item()
        local_y = plot_summary['Plot Centre Y'].item()
        local_z = plot_summary['Plot Centre Z'].item()
        global_x=""
        global_y=""
        global_z=""
    else :
        location_ref="world"
        local_x=""
        local_y=""
        local_z=""
        global_x = plot_summary['Plot Centre X'].item()
        global_y = plot_summary['Plot Centre Y'].item()
        global_z = plot_summary['Plot Centre Z'].item()

    heights = np.around(np.array(tree_data['Height']),2)
    dbhs = np.around(np.array(tree_data['DBH_taper']))
    ccis = np.around(np.array(tree_data['CCI']))


    report_summary = [
        {"attribute":"Population Name", "min":plot_summary["PlotId"].item(), "mean":"", "max":"", "units":""},
        {"attribute":"Stratum ID", "min":1, "mean":"", "max":"", "units":""},
        {"attribute":"Age", "min":"", "mean":"", "max":"", "units":"years"},
        {"attribute":"Plot Type", "min":plot_type, "mean":"", "max":"", "units":""},
        {"attribute":"Plot Radius", "min":plot_radius, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Plot Area", "min":plot_summary["Plot Area"].item(), "mean":"", "max":"", "units":"Ha"},
        {"attribute":"Location Ref", "min":location_ref, "mean":"", "max":"", "units":""},
        {"attribute":"Local X", "min":local_x, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Local Y", "min":local_y, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Local Z", "min":local_z, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Global X", "min":global_x, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Global Y", "min":global_y, "mean":"", "max":"", "units":""},
        {"attribute":"Global Z", "min":global_z, "mean":"", "max":"", "units":"metres"},
        {"attribute":"Trees", "min":plot_summary["Num Trees in Plot"].item(), "mean":"", "max":"", "units":"count"},
        {"attribute":"Stems", "min":plot_summary["Num Trees in Plot"].item(), "mean":"", "max":"", "units":"count"},
        {"attribute":"Stocking SPH", "min":plot_summary["Stems/ha"].item(), "mean":"", "max":"", "units":"Stems per ha"},
        {"attribute":"Merchantable Stems", "min":"", "mean":"", "max":"", "units":"Stems per ha"},
        {"attribute":"Tree Height", "min":np.min(heights), "mean":np.around(np.mean(heights),2), "max":np.max(heights), "units":"metres"},
        {"attribute":"DBH", "min":np.min(dbhs), "mean":np.around(np.mean(dbhs)), "max":np.max(dbhs), "units":"millimetres"},
        {"attribute":"DBH CCI", "min":np.min(ccis), "mean":np.around(np.mean(ccis)), "max":np.max(ccis), "units":"percent"},
        {"attribute":"Diameter Breast Height", "min":plot_summary['DBH_height'].item(), "mean":"", "max":"", "units":"metres"}
    ]

    return report_summary


def create_figures(output_dir, plotID, plot_summary):
    
    DTM, _ = load_file(f'{output_dir}/{plotID}_DTM.laz')

    plot_centre = [float(plot_summary['Plot Centre X'].item()), float(plot_summary['Plot Centre Y'].item())]
    plot_radius = float(plot_summary['Plot Radius'].item())
    plot_radius_buffer = float(plot_summary['Plot Radius Buffer'].item())
    plot_names = []
        
    if os.path.isfile(output_dir + plotID + '_QC_data_fieldID.csv'):
        # tree_data = add_missing_data(output_dir, plotID, plot_centre) # use for filling in missing values
        tree_data = customize_data(output_dir, plotID)
        Volume = []
        DBH_heights = np.array(tree_data['DBH_bin'])
        DBH = np.array(tree_data['DBH_taper'], dtype=int)

    elif QC and os.path.isfile(output_dir + plotID + f'{QC}_data.csv'):
        tree_data = pd.read_csv(output_dir + plotID + f'{QC}_data.csv')
        print(f'Loading file... {output_dir}/{plotID}{QC}_data.csv')
        Volume, Volume_2, DBH_heights = get_QC_volume(output_dir, plotID, tree_data, plot_summary['dbh_height'])
        DBH_heights = np.array(tree_data['DBH_bin'])
        DBH = np.array(tree_data['DBH_taper'], dtype=int)

    else :                       
        tree_data = pd.read_csv(output_dir + plotID + '_tree_data.csv')
        print(f'Loading file... {output_dir}/{plotID}_tree_data.csv')
        Volume = np.array(tree_data['Volume'])
        DBH_heights = np.array(tree_data['DBH_bin'])
        DBH = np.array(tree_data['DBH_taper'], dtype=int)
    
    TreeIds = np.array(tree_data['TreeNumber'], dtype=int)
    TreeLoc_X = np.array(tree_data['TreeLocation_X'])
    TreeLoc_Y = np.array(tree_data['TreeLocation_Y'])
    DBH_z = np.array(tree_data['TreeLocation_Z'])    
    height = np.around(np.array(tree_data['Height']),2)
    CCI = np.array(tree_data['CCI'])
        

    dtmmin = np.min(DTM[:, :2], axis=0)
    dtmmax = np.max(DTM[:, :2], axis=0)
    plot_max_distance = np.max(dtmmax - dtmmin)

    DBH_H = plot_summary['DBH_height'].item()
    
    ## Create Diameter Map Plot
    # 
    if plot_summary['Offset X'].item() > 100 : filename = "C2_E_N"
    else : filename ="C2_0_0"

    with laspy.open(f'{output_dir}/{plotID}_{filename}_hnom.laz') as f: 
        las = f.read()
        points = np.vstack((las.x, las.y, las.z)).T             # get point coordinates
        class_label = np.transpose(getattr(las, 'label'))       # get the 'label' attribute
        heights = np.transpose(getattr(las,'height_above_dtm')) 
        tree_ids = np.transpose(getattr(las,'tree_id')) 

    
    mask = np.abs(heights - DBH_H) < .03
    points = points[mask, :]
    class_label = class_label[mask]

    # fig1 = plt.figure(figsize=(9, 7))
    fig1 = plt.figure(figsize=(16, 16))
    ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.set_title("Plot " + plotID + " - Stem Map - Tree ID (DBH)", fontsize=18)
    ax1.set_title(f"{plotID} Stem Map (a 6mm slice at {DBH_H}m above the ground)", fontsize=18)        
    ax1.set_xlabel("Easting + " + str(np.around(plot_centre[0], 2)) + ' (m)', fontsize=14)
    ax1.set_ylabel("Northing + " + str(np.around(plot_centre[1], 2)) + ' (m)', fontsize=14)
    ax1.axis('equal')

    xmin = (np.min(DTM[:, 0]) - plot_centre[0])
    xmax = (np.max(DTM[:, 0]) - plot_centre[0])
    ymin = (np.min(DTM[:, 1]) - plot_centre[1])
    ymax = (np.max(DTM[:, 1]) - plot_centre[1])

    if plot_radius == 0:
        plot_radius = np.max([(xmax-xmin)/2,(ymax-ymin)/2 ])
    
    ax1.set_facecolor('whitesmoke')
    plot_boundary = plt.Circle(xy=(0,0), radius=plot_radius, facecolor='white', edgecolor='k', zorder=1)
    ax1.add_patch(plot_boundary)
    
    ## remove points outside the plot radius
    mask = np.linalg.norm(points[:,:2] - plot_centre, axis=1) < plot_radius      # find points within the plot boundary
    points = points[mask,:]
    class_label = class_label[mask]


    ax1.scatter(points[class_label==2,0]-plot_centre[0], points[class_label==2, 1]-plot_centre[1], marker=',', s=.1, c='darkseagreen', zorder=3)    # plot Vegetation points
    ax1.scatter(points[class_label==4,0]-plot_centre[0], points[class_label==4, 1]-plot_centre[1], marker=',', s=.1, c='red', zorder=3)             # plot stem points

    
    label_offset = np.array([-0.016, 0.006]) * plot_max_distance   # labels for stems

    for xc,yc,r,id in zip(TreeLoc_X, TreeLoc_Y,DBH/2,TreeIds):
        stem_circle = plt.Circle(xy=(xc-plot_centre[0], yc-plot_centre[1]), radius=r/1000, 
                                    fill=False, edgecolor='b',zorder=4)
        ax1.add_patch(stem_circle)
        ax1.text(xc - plot_centre[0] + label_offset[0],
                    yc - plot_centre[1] + r/1000 + label_offset[1], 
                    str(id) + ' (' + str(round(r*2)) + ')', fontsize=14, zorder=6)

    
    ax1.scatter(plot_centre[0], plot_centre[1], marker='x', s=50, c='blue', zorder=9) # plot center mark

    padding = 0.1
    ax1.set_xlim([xmin + xmin * padding, xmax + xmax * padding])
    ax1.set_ylim([ymin + ymin * padding, ymax + ymax * padding])

    offset_x = xmax - plot_radius_buffer
    offset_y = ymax/2
    # offset_y = (np.max(points[:,1]) + np.min(points[:,1]))/2
    ax1.arrow(offset_x, offset_y, 0, 1.8, width = 0.14 )
    ax1.annotate('North', xy = (offset_x + 0.3, offset_y + 1.2),fontsize=14)
    # ax1.arrow(offset_x, offset_y, 0, 1.2, width = 0.1 )
    # ax1.annotate('North', xy = (offset_x + 0.3, offset_y),fontsize=14)

    # plt.tight_layout()
    handles = [Line2D(range(1), range(1), label='Vegetation Point', color="white", marker='.',
                        markerfacecolor='green'),
                Line2D(range(1), range(1), label='Stem Point', color="white", marker='.',
                        markerfacecolor='red'),                   
                Line2D(range(1), range(1), label='Tree Stem - TreeNumber (DBH)', color="white", marker='o', markerfacecolor='none',
                        markeredgecolor='k')]
    ax1.legend(handles=handles,
                loc='upper right',
                bbox_to_anchor=(1, 1),   
            #    ncol=2,
                facecolor="white",
                borderaxespad=0.,
                fontsize=14,
                markerscale=2
                )
    ##
    # fig1.tight_layout()
    plot_filename = f'{output_dir}/{plotID}_DBH_Map{QC}.png'
    fig1.savefig(plot_filename, dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    plot_names.append(plot_filename)

    fig2 = plt.figure(figsize=(7,3))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title(f'{plotID} Height vs DBH', fontsize=10)
    ax2.set_xlabel("DBH (mm)")              #Aglika - report diameters in mm
    ax2.set_ylabel("Height (m)")
    if DBH.shape[0] > 0:
        bin_width = 15
        bins = np.arange(bin_width, np.ceil(np.max(DBH)) + bin_width, bin_width)
        ax2.hist(DBH,
                    bins=bins,
                    edgecolor='black',
                    facecolor='aliceblue')
        ax2.set_ylabel("Count")
        # ax2.set_ylim(0,10)
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        
        # add a twin to the x axis so it can be shared
        ax3 = ax2.twinx()                             
        ax3.scatter(DBH, height, marker='.', s=12, c='red')
        ax3.set_ylabel("Height")
        ax3.set_ylim(-8, max(height)+5)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
    
    plot_filename = f'{output_dir}/{plotID}_DBH_Height.png'
    fig2.savefig(plot_filename, dpi=600, bbox_inches='tight', pad_inches=0.0)
    # fig2.savefig(output_dir + 'Diameter at Breast Height Distribution.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    plot_names.append(plot_filename)

    return tree_data, plot_names

    
def get_QC_volume(output_dir, plotID, tree_data, dbh_height):

    trees = tree_data['TreeNumber'].values 
    DBH = tree_data['DBH_taper'].values
    h_col = tree_data.columns.get_loc('Height')
    v1 = np.zeros(trees.shape[0])
    v2 = np.zeros(trees.shape[0])
    dbh_heights = np.zeros(trees.shape[0])
    for r, tree_id in enumerate(trees):
        # get the tree_height from the QC data
        tree_height = tree_data.iloc[r,h_col]            
        # get taper output
        # str(plot_summary['PlotId'][0])
        taper_df = pd.read_csv(output_dir + f'/taper_QC/{plotID}_dn_{tree_id}.csv')
        taper = taper_df[['Height_Bin','MA_Diameter']].values
        
        volume1, volume2, bin = volume_from_taper(taper, tree_height, dbh_height)
        volume_QC = (np.pi * ((DBH[r]/2000) ** 2) * (((tree_height - dbh_height)/3) + dbh_height))

        v1[r] = np.around(volume1,2)
        v2[r] = np.around(volume_QC,2)
        dbh_heights[r] = bin

    tree_data.insert(loc=len(tree_data.columns), column='Volume', value=v1)    
    tree_data.insert(loc=len(tree_data.columns), column='Volume_Cone', value=v2)    
    tree_data['DBH_bin'] = np.around(dbh_heights,1)
    tree_data.to_csv(output_dir + plotID + '_tree_data_report.csv', index=None, sep=',')

    return v1, v2, dbh_heights


def add_missing_data(output_dir, plotID, plot_centre):
    # when DBH is measured manually

    if os.path.isfile(os.path.join(output_dir, plotID + '_tree_data_edit.csv')):       
        tree_data = pd.read_csv(os.path.join(output_dir, plotID + '_tree_data_edit.csv'))
        print("Reading tree_data_edit.csv")
    
    tree_data.drop(tree_data.columns[13:], axis=1, inplace=True) # hardcoded
    # tree_data = tree_data.convert_dtypes()
    col_list=list(tree_data.columns)

    DBH = np.array(tree_data['DBH_taper']) 
    CCI = np.array(tree_data['CCI'])  # some edited files have DBH but all edited files do not have CCI
    nodbh_mask = np.logical_or(np.isnan(CCI), np.isnan(DBH))

    # for missing DBH values:
    if np.any(nodbh_mask):
        heights = np.array(tree_data['Height'])
        tree_data.insert(loc=len(tree_data.columns), column='Review', value=nodbh_mask*1)            
        
        real_dbh = DBH[~nodbh_mask] # use calculated height and dbhs to estimate the missing ones
        real_heights = heights[~nodbh_mask]
        for row in np.where(nodbh_mask)[0]:
            margin = .1
            dbhs = np.hstack((real_dbh[real_heights < (heights[row]+margin)], 
                                    real_dbh[real_heights > (heights[row]-margin)]))
            while len(dbhs)==0:
                margin += .1
                dbhs = np.hstack((real_dbh[real_heights < heights[row]+margin], 
                                        real_dbh[real_heights > heights[row]-margin]))                
            DBH[row] = np.floor(np.median(dbhs))
            CCI[row] = .5

        tree_data['DBH_taper'] = DBH
        tree_data['Review'] = nodbh_mask*1
        tree_data['CCI'] = CCI
    else:
        tree_data.insert(loc=len(tree_data.columns), column='Review', value=0)            

    DBH_x = np.array(tree_data['TreeLocation_X']) - plot_centre[0]  #float(plot_summary['Plot Centre X'])
    DBH_y = np.array(tree_data['TreeLocation_Y']) - plot_centre[1]  # float(plot_summary['Plot Centre Y'])

    tree_data['Distance'] = np.around(np.sqrt(DBH_x**2 + DBH_y**2),2)

    # tree_data.droplevel
    bearing = np.round(np.degrees(np.arctan2(DBH_x, DBH_y))) % 360
    tree_data['Bearing'] = np.array(bearing.flatten())
    tree_data = tree_data.sort_values(by='Bearing')
    
    # reset the ids
    tree_data['TreeNumber'] = np.arange(len(tree_data))+1

    # tree_data['TreeLocation_Z'] = DBH_H
    tree_data['PlotId'] = plotID
    tree_data.to_csv(output_dir + plotID + '_tree_data_report.csv', index=None, sep=',')

    return tree_data


def customize_data(output_dir, plotID):
    # for various manually edited data files

    try:
        tree_data = pd.read_csv(os.path.join(output_dir, plotID + '_QC_data_fieldID.csv'))
        print ("Reading ..._fieldID.csv")        
    except Exception as e:
        sys.exit(f'No sush file: {str(e)}')
        
    # tree_data.drop(tree_data.columns[13:], axis=1, inplace=True) # hardcoded
    # tree_data = tree_data.convert_dtypes()
    col_list=list(tree_data.columns)

    fieldID = np.asarray(tree_data['Field_ID']) 
    DBH = np.asarray(tree_data['DBH_taper']) 
    mask = np.logical_and(DBH<110, np.isnan(fieldID))
    
    # tree_data.fieldID.str.contains('-$')
    tree_data.drop(tree_data[mask].index, inplace=True)
    tree_data.reset_index(inplace=True, drop=True)

    tree_data.drop(['TreeNumber'], axis=1, inplace=True) # delete old IDs
    tree_data.rename(columns={"Field_ID":"TreeNumber"},inplace=True)        # rename for later use
    # tree_data.TreeNumber = np.round(tree_data.TreeNumber)
    tree_data.TreeNumber = tree_data.TreeNumber.fillna(0)

    field_x = tree_data.TreeLocation_X  # + 430399.987  #432055.022
    field_y = tree_data.TreeLocation_Y  # + 6430527.852                # 6428973.047
    
    field_df = deepcopy(tree_data)
    field_df.TreeLocation_X = field_x
    field_df.TreeLocation_Y = field_y
    field_df.to_csv(output_dir + plotID + f'_tree_data_report{QC}.csv', index=None, sep=',')
    
    return tree_data