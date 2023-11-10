from math import pi
from mdutils.mdutils import MdUtils
import markdown
from mdutils import Html
import pandas as pd
import numpy as np
from tools import load_file, subsample_point_cloud, volume_from_taper
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

QC='' # use QC='_QC' to generate plot reports from edited tables. Will look for {plotID}_QC_data.csv file
# QC='_Field'


class ReportWriter:
    def __init__(self, parameters):
        self.parameters = parameters
        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        filename = os.path.basename(self.filename)
        if os.path.isdir(filename):
            self.output_dir = filename
            self.plotID = filename.split('_')[0]
        else :
            filename = filename.split('_')[0]
            self.plotID = filename.split('.')[0]
            # filename = filename.replace('.laz','')
            self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + self.plotID + '_FT_output/'
        
        self.filename = self.plotID + '.laz'        
        
        self.plot_summary = pd.read_csv(self.output_dir + self.plotID + '_plot_summary.csv', index_col=False)
        

        self.parameters['plot_centre'] = [float(self.plot_summary['Plot Centre X'].item()),
                                            float(self.plot_summary['Plot Centre Y'].item())]
        
        self.plot_area = float(self.plot_summary['Plot Area'].item())
        self.stems_per_ha = float(self.plot_summary['Stems/ha'].item())
        # self.stems_per_ha = round(self.tree_data.shape[0] / self.plot_area) # done in the Measure class

        self.plot_radius = float(self.plot_summary['Plot Radius'].item())
        self.plot_radius_buffer = float(self.plot_summary['Plot Radius Buffer'].item())
        
        if os.path.isfile(self.output_dir + self.plotID + '_QC_data_fieldID.csv'):
            # self.tree_data = self.add_missing_data() # use for filling in missing values
            self.tree_data = self.customize_data()
            self.Volume = []
            self.DBH_heights = np.array(self.tree_data['DBH_bin'])
            self.DBH = np.array(self.tree_data['DBH'])

        elif QC and os.path.isfile(self.output_dir + self.plotID + f'{QC}_data.csv'):
            self.tree_data = pd.read_csv(self.output_dir + self.plotID + f'{QC}_data.csv')
            print(f'Loading file... {self.output_dir}/{self.plotID}{QC}_data.csv')
            self.Volume, self.Volume_2, self.DBH_heights = self.get_QC_volume()
            self.DBH_heights = np.array(self.tree_data['DBH_bin'])
            self.DBH = np.array(self.tree_data['DBH_taper'])

        else :                       
            self.tree_data = pd.read_csv(self.output_dir + self.plotID + '_tree_data.csv')
            print(f'Loading file... {self.output_dir}/{self.plotID}_tree_data.csv')
            self.Volume = np.array(self.tree_data['Volume'])
            self.DBH_heights = np.array(self.tree_data['DBH_bin'])
            self.DBH = (np.array(self.tree_data['DBH_taper']))
        
        self.DBH = np.int16(self.DBH)

        # if os.path.isfile(os.path.join(self.output_dir, 'tree_data_report.csv')):
        #     self.tree_data = pd.read_csv(os.path.join(self.output_dir, 'tree_data_report.csv'))
        # else:
        #     print('No tree_data_report')
        #     self.tree_data = pd.read_csv(self.output_dir + 'tree_data_sorted.csv')


        self.TreeIds = np.array(self.tree_data['TreeNumber'], dtype=int)
        self.TreeLoc_X = np.array(self.tree_data['TreeLocation_X'])
        self.TreeLoc_Y = np.array(self.tree_data['TreeLocation_Y'])

        self.DBH_z = np.array(self.tree_data['TreeLocation_Z'])    
        self.height = np.around(np.array(self.tree_data['Height']),2)

        self.CCI = np.array(self.tree_data['CCI'])
        # self.Stem2Veg = np.array(self.plot_summary['Num Stem Points']/self.plot_summary['Num Vegetation Points'])
        

    def make_report(self):
        self.plot_outputs()
        self.create_report()


    def create_report(self):
        filename = self.output_dir + 'Plot_Report'
        report_title = f'3D SCAN PLOT REPORT'
        if QC=='_Field':
            report_title = f'Field Data: Plot {self.plotID} Report' 
        elif QC == '_QC':
            report_title = f'TreeTools QC: Plot {self.plotID} Report' 
        mdFile = MdUtils(file_name=filename, title=report_title)
        mdFile.new_header(level=1, title='')  # style is set 'atx' format by default. # Aglika  - to use level 'n' all smaller levels must be used !!
        mdFile.new_header(level=2, title='')  
        mdFile.new_header(level=3, title='')  
        
        level=4
        mdFile.new_header(level=level, title='Plot ID: ' + self.plotID)
        mdFile.new_header(level=level, title='Point Cloud Filename: ' + self.filename)
        mdFile.new_header(level=level,
                          title='Plot Centre (local coords): X: ' + str(np.around(self.parameters['plot_centre'][0], 2)) + ' m, Y: ' +
                                str(np.around(self.parameters['plot_centre'][1], 2)) + ' m')
        if self.plot_radius > 0:
            mdFile.new_header(level=level, title='Plot Radius: ' + str(self.plot_radius) + 'm')
            # self.parameters['plot_radius']) + ' m, ' + ' Plot Radius Buffer: ' + str(self.parameters['plot_radius_buffer']) + ' m, Plot Area: ' + str(np.around(self.plot_area, 3)) + ' ha')
            
        mdFile.new_header(level=level, title='Plot Area: ' + str(np.around(self.plot_area, 3)) + ' ha')

        if self.DBH.shape[0] > 0:
            mdFile.new_header(level=level, title='Stems/ha:  ' + str(np.int16(self.stems_per_ha)))
            mdFile.new_header(level=level, title=f'Mean DBH: {round(np.mean(self.DBH))}mm (min: {np.min(self.DBH)}mm, max: {np.max(self.DBH)}mm)')        
            mdFile.new_header(level=level, title=f'Mean Tree Height: {(np.around(np.mean(self.height), 2))}m (min: {np.min(self.height)}m, max: {np.max(self.height)}m)')           


            # if len(self.Volume)>0:
            #     mdFile.new_line()
            #     mdFile.new_header(level=level, title='Mean Volume: ' + str(np.around(np.mean(self.Volume), 3)) + ' m3')           
            #     mdFile.new_header(level=level, title='minimum Volume: ' + str(np.around(np.min(self.Volume), 3)) + ' m3' +
            #                                 ', maximum Volume: ' + str(np.around(np.max(self.Volume), 3)) + ' m3')        

            #     mdFile.new_header(level=level, title='Total Plot Stem Volume: ' + str(np.around(np.sum(self.Volume), 3)) + ' m3')
            #     #  mdFile.new_header(level=level, title='Total Plot Stem Volume 2: ' + str(np.around(np.sum(self.Volume_2), 3)) + ' m3')

            # mdFile.new_header(level=level+1, title='Stem to Vegetation Point Ratio: ' + str(np.around(self.Stem2Veg[0], 3)))
            # mdFile.new_header(level=level, title='Stem to Vegetation Point Ratio Normalised: ' + str(np.around(self.Stem2Veg[0]/len(self.tree_data), 3)))
            mdFile.new_line()

        else:
            mdFile.new_header(level=level, title='Stems/ha: 0')
            mdFile.new_header(level=level, title='No stems found.')
        
        mdFile.new_line()
        mdFile.write(f'Minimum CCI: {round(np.min(self.CCI))}', bold_italics_code='')
        mdFile.new_header(level=level+1, title='')
        mdFile.new_line()
        total_processing_time = np.around(float(self.plot_summary['Total Run Time (s)'].item()))
        mdFile.write(f'TreeTools Processing Time: {int(total_processing_time/60.)} minutes and {int(total_processing_time%60.)} seconds', bold_italics_code='c')

        mdFile.new_line()
        mdFile.new_header(level=level+1, title='')
        mdFile.new_line('Histograms', bold_italics_code='bc')

        # path = "Diameter at Breast Height Distribution.png"
        # mdFile.new_paragraph(Html.image(path=path, size='300'))
        
        # path = "Tree Height Distribution.png"
        # mdFile.new_paragraph(Html.image(path=path, size='300'))

        path = f'{self.plotID}_DBH_Height.png'
        mdFile.new_paragraph(Html.image(path=path, size='700'))

        path = f"{self.plotID}_canopy.png" 
        mdFile.new_paragraph(Html.image(path=path, size='500'))

        # path = "Stem_Map.png"
        # mdFile.new_paragraph(Html.image(path=path, size='800'))

        path = f'{self.plotID}_DBH_Map{QC}.png'
        mdFile.new_paragraph(Html.image(path=path, size='780'))
        mdFile.new_line()

        # mdFile.new_header(level=2, title=':', style='setext')
        # path = "Tree Volume 1 Distribution.png"
        # mdFile.new_paragraph(Html.image(path=path, size='400'))
        # path = "Tree Volume 2 Distribution.png"
        # mdFile.new_paragraph(Html.image(path=path, size='400'))

        self.tree_data = self.tree_data.convert_dtypes()
        cols_to_read = ['TreeNumber','TreeLocation_X','TreeLocation_Y','DBH_bin','Distance','Bearing','Height','DBH_taper','CCI'] #'Crown_Height','Volume']
        report_cols = ['TreeID','X','Y','Z','Dist','Bea','Ht (m)','DBH (mm)','CCI']   #'Crown_Ht','Volume'

        # data_cols = self.tree_data.columns.astype(str)
        # cols_to_read = [x for x in data_cols if x in cols_to_read] 
        # report_cols = report_cols[: len(cols_to_read)] # big assumption that the missing columns will be at the end 

        plot_data_df = self.tree_data[cols_to_read].round(2)     # get data from appropriate columns (as floats)
        # plot_data_df[['Height']]=np.around(plot_data_df[['Height']],1)
        # plot_data_df[['DBH_taper']]=np.int16(plot_data_df[['DBH_taper']])

        # format as a markdown table
        plot_data_df = plot_data_df.astype(str)                 # convert the data to string values

        string_list = report_cols
        # string_list = ['_'+str(x)+'_' for x in string_list]   # the '_' makes the text italic
        
        for r in np.arange(0, len(plot_data_df)-1) : 
            line = plot_data_df.iloc[r].to_list()     # create a new row for the table -  list of strings
            string_list.extend(line)                 
        
        # DO NOT NEED A SEPARATE MARKDOWN for the TABLE if we don't use markdownFromFile()
        # instead append to the raw html file
        mdFile.new_line()
        mdFile.new_table(columns=plot_data_df.shape[1], rows=(plot_data_df.shape[0]), text=string_list, text_align='right') # a markdown table element
        mdFile.create_md_file()        

        # Tables are not included in the standard Markdown - have to be handled separately
        # 
        # md = markdown.Markdown(extensions = ['tables', 'codehilite','meta'], output_format="html5")  # meta allows the use of Python functions and classes from within Markdown documents and assign results to variables
        md = markdown.Markdown(extensions = ['tables'], output_format="html5")
        with open(filename +'.md', 'r') as f :
            report_html = md.convert(f.read())
        report = ["""<!DOCTYPE html>
                    <html>
                    <head>
                    <meta name="pdfkit-page-size" content="A4"/>
                    <meta name="pdfkit-orientation" content="Portrait"/>
                    <style>
                    h4 {
                        color: black;
                        font-family: Helvetica;
                        font-size: 100%;
                    }
                    p {
                        font-family: Helvetica;
                        font-size: 100%;
                    }
                    table {
                        border-collapse: collapse;
                        width: 80%;
                    }
                    table, tr, td, th, tbody, thead, tfoot {
                        page-break-inside: avoid;
                        font-family: Helvetica;
                    }
                    th, td {
                        text-align: right;
                        padding: 6px;
                    }
                    tr:nth-child(even) {
                        background-color: #EEEEEE;
                    }
                    tr {
                        border-bottom: 1px solid #ddd;
                    }
                    </style>
                    </head>
                    """]
        report.append(report_html)
        report.append( """
                        </body>
                        </html>
                        """ )
        with open(filename + '.html', 'w') as f : f.write(''.join(report))
        print("HMTL report done") 

        # wkhtnltopdf will gives an error if it cannot find the image files
        config = pdfkit.configuration(wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
        kitoptions = {
            "enable-local-file-access": None
        }
        pdfkit.from_file(filename + '.html', self.output_dir + self.plotID + f'_Plot_Report{QC}.pdf', configuration = config, options = kitoptions)
        print(f'{QC} PDF report done') 
        # clean up
        os.remove(self.output_dir + 'Plot_Report.md')
        os.remove(self.output_dir + 'Plot_Report.html')

    def plot_outputs(self):
        self.DTM, _ = load_file(self.output_dir + self.plotID + '_DTM.laz')

        dtmmin = np.min(self.DTM[:, :2], axis=0)
        dtmmax = np.max(self.DTM[:, :2], axis=0)
        plot_max_distance = np.max(dtmmax - dtmmin)
        if self.parameters['plot_centre'] is None:
            self.parameters['plot_centre'] =  (dtmmin + dtmmax) / 2
        else :
            plot_centre = self.parameters['plot_centre']

        if 'DBH_height' in self.plot_summary.columns :
            DBH_H = self.plot_summary['DBH_height'][0]
        else :
            DBH_H = self.parameters['dbh_height']
        
        ## Create Diameter Map Plot
        # 
        # with laspy.open(self.output_dir + self.plotID + 'C2_0_0.laz') as f:      # read segmented cloud
        with laspy.open(self.output_dir + self.plotID + '_C2_0_0_hnom.laz') as f: 
            las = f.read()
            points = np.vstack((las.x, las.y, las.z)).T             # get point coordinates
            class_label = np.transpose(getattr(las, 'label'))       # get the 'label' attribute
            heights = np.transpose(getattr(las,'height_above_dtm')) 
            tree_ids = np.transpose(getattr(las,'tree_id')) 

        # for qc-ed data plot a slice around 1.3m - Always, ow canot see non-detected trees
        # if bool(QC) : 
        if 1:
            mask = np.abs(heights - DBH_H) < .03
            points = points[mask, :]
            class_label = class_label[mask]
        
        # otherwise plot stem points where DBH was measured
        else: 
            # plot vegetation points by height_above_dtm
            # mask = (tree_ids <= 0)
            mask = np.logical_and(class_label==2, np.abs(heights - DBH_H) < .03)  # the vegetation class
            veg_points = deepcopy(points[mask,:])
            label = deepcopy(class_label[mask]) 
        
            # for each tree create a slice by z value so it is parallel to the ground
            # tree_base_z = self.DBH_z - self.DBH_height 
            for i,tree in enumerate(np.unique(tree_ids[tree_ids > 0])):
                tree_mask =  (tree_ids==tree)
                tree_points = points[tree_mask,:]
                tree_label = class_label[tree_mask]
                point_heights = heights[tree_mask]
                # mask = (np.abs(tree_points[:,2] - self.DBH_z[i]) < 0.3)
                mask = (np.abs(point_heights - self.DBH_heights[i]) < 0.1)
                
                veg_points = np.vstack((veg_points, tree_points[mask,:]))
                label = np.hstack((label, tree_label[mask]))
            
            # # put the DBH slice points together
            points = veg_points
            class_label = label
            del heights, las

        # fig1 = plt.figure(figsize=(9, 7))
        fig1 = plt.figure(figsize=(16, 16))
        ax1 = fig1.add_subplot(1, 1, 1)
        # ax1.set_title("Plot " + self.plotID + " - Stem Map - Tree ID (DBH)", fontsize=18)
        ax1.set_title(f"{self.plotID} Stem Map (a 6mm slice at {DBH_H}m above the ground)", fontsize=18)        
        ax1.set_xlabel("Easting + " + str(np.around(plot_centre[0], 2)) + ' (m)', fontsize=14)
        ax1.set_ylabel("Northing + " + str(np.around(plot_centre[1], 2)) + ' (m)', fontsize=14)
        ax1.axis('equal')

        xmin = (np.min(self.DTM[:, 0]) - plot_centre[0])
        xmax = (np.max(self.DTM[:, 0]) - plot_centre[0])
        ymin = (np.min(self.DTM[:, 1]) - plot_centre[1])
        ymax = (np.max(self.DTM[:, 1]) - plot_centre[1])

        if self.plot_radius == 0:
            self.plot_radius = np.max([(xmax-xmin)/2,(ymax-ymin)/2 ])
        
        ax1.set_facecolor('whitesmoke')
        plot_boundary = plt.Circle(xy=(0,0), radius=self.plot_radius, facecolor='white', edgecolor='k', zorder=1)
        ax1.add_patch(plot_boundary)
        
        ## remove points outside the plot radius
        mask = np.linalg.norm(points[:,:2] - plot_centre, axis=1) < self.plot_radius      # find points within the plot boundary
        points = points[mask,:]
        class_label = class_label[mask]

        ax1.scatter(points[class_label==2,0]-plot_centre[0], points[class_label==2, 1]-plot_centre[1], marker=',', s=.1, c='darkseagreen', zorder=3)    # plot Vegetation points
        ax1.scatter(points[class_label==4,0]-plot_centre[0], points[class_label==4, 1]-plot_centre[1], marker=',', s=.1, c='red', zorder=3)             # plot stem points
        
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     subcontours = ax1.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1],
        #                                  self.DTM[:, 2], levels=sub_levels, colors='burlywood', linestyles='dashed',
        #                                  linewidths=2, zorder=3)
        #     contours = ax1.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2],
        #                               levels=levels, colors='darkgreen', linewidths=2, zorder=5)
        # plt.clabel(subcontours, inline=True, fmt='%1.1f', fontsize=6, zorder=4)
        # plt.clabel(contours, inline=True, fmt='%1.0f', fontsize=10, zorder=6)

        
        label_offset = np.array([-0.016, 0.006]) * plot_max_distance   # labels for stems     

        for xc,yc,r,id in zip(self.TreeLoc_X, self.TreeLoc_Y,self.DBH/2,self.TreeIds):
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

        offset_x = xmax - self.plot_radius_buffer
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
        fig1.savefig(f'{self.output_dir}/{self.plotID}_DBH_Map{QC}.png', dpi=1000, bbox_inches='tight', pad_inches=0.0)
        plt.close()


        fig2 = plt.figure(figsize=(7,3))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_title(f'{self.plotID} Height vs DBH', fontsize=10)
        ax2.set_xlabel("DBH (mm)")              #Aglika - report diameters in mm
        ax2.set_ylabel("Height (m)")
        if self.DBH.shape[0] > 0:
            bin_width = 15
            bins = np.arange(bin_width, np.ceil(np.max(self.DBH)) + bin_width, bin_width)
            ax2.hist(self.DBH,
                     bins=bins,
                     edgecolor='black',
                     facecolor='aliceblue')
            ax2.set_ylabel("Count")
            # ax2.set_ylim(0,10)
            ax2.yaxis.set_label_position("left")
            ax2.yaxis.tick_left()
            
            # add a twin to the x axis so it can be shared
            ax3 = ax2.twinx()                             
            ax3.scatter(self.DBH, self.height, marker='.', s=12, c='red')
            ax3.set_ylabel("Height")
            ax3.set_ylim(-8, max(self.height)+5)
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()
            
        fig2.savefig(f'{self.output_dir}/{self.plotID}_DBH_Height.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        # fig2.savefig(self.output_dir + 'Diameter at Breast Height Distribution.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        if 0 :
            fig4 = plt.figure(figsize=(4, 3))
            ax4 = fig4.add_subplot(1, 1, 1)
            ax4.set_title("Tree Volume From Taper Distribution", fontsize=10)
            ax4.set_xlabel("Volume 1 (m^3)")
            ax4.set_ylabel("Count")
            if self.Volume.shape[0] > 0:
                bin_width = 0.1
                bins = np.arange(0, np.ceil(np.max(self.Volume) * 10) / 10 + bin_width, bin_width)

                ax4.hist(self.Volume,
                        bins=bins,
                        range=(0, np.ceil(np.max(self.Volume) * 10) / 10 + bin_width),
                        linewidth=0.5,
                        edgecolor='black',
                        facecolor='green')
            fig4.savefig(self.output_dir + 'Tree Volume 1 Distribution.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            fig4 = plt.figure(figsize=(4, 3))
            ax4 = fig4.add_subplot(1, 1, 1)
            ax4.set_title("Tree Cylindner-Cone Volume Distribution", fontsize=10)
            ax4.set_xlabel("Volume 2 (m^3)")
            ax4.set_ylabel("Count")
            if self.Volume_2.shape[0] > 0:
                bin_width = 0.1
                bins = np.arange(0, np.ceil(np.max(self.Volume_2) * 10) / 10 + bin_width, bin_width)

                ax4.hist(self.Volume_2,
                        bins=bins,
                        range=(0, np.ceil(np.max(self.Volume_2) * 10) / 10 + bin_width),
                        linewidth=0.5,
                        edgecolor='black',
                        facecolor='green')
            fig4.savefig(self.output_dir + 'Tree Volume 2 Distribution.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
            plt.close()
    
    def get_QC_volume(self):

        trees = self.tree_data['TreeNumber'].values 
        DBH = self.tree_data['DBH_taper'].values
        h_col = self.tree_data.columns.get_loc('Height')
        v1 = np.zeros(trees.shape[0])
        v2 = np.zeros(trees.shape[0])
        dbh_heights = np.zeros(trees.shape[0])
        for r, tree_id in enumerate(trees):
            # get the tree_height from the QC data
            tree_height = self.tree_data.iloc[r,h_col]            
            # get taper output
            # str(self.plot_summary['PlotId'][0])
            taper_df = pd.read_csv(self.output_dir + f'/taper_QC/{self.plotID}_dn_{tree_id}.csv')
            taper = taper_df[['Height_Bin','MA_Diameter']].values
            
            volume1, volume2, bin = volume_from_taper(taper, tree_height, self.parameters['dbh_height'])
            volume_QC = (np.pi * ((DBH[r]/2000) ** 2) * (((tree_height - self.parameters['dbh_height'])/3) + self.parameters['dbh_height']))

            v1[r] = np.around(volume1,2)
            v2[r] = np.around(volume_QC,2)
            dbh_heights[r] = bin

        self.tree_data.insert(loc=len(self.tree_data.columns), column='Volume', value=v1)    
        self.tree_data.insert(loc=len(self.tree_data.columns), column='Volume_Cone', value=v2)    
        self.tree_data['DBH_bin'] = np.around(dbh_heights,1)
        self.tree_data.to_csv(self.output_dir + self.plotID + '_tree_data_report.csv', index=None, sep=',')

        return v1, v2, dbh_heights


    def add_missing_data(self):
        # when DBH is measured manually

        if os.path.isfile(os.path.join(self.output_dir, self.plotID + '_tree_data_edit.csv')):       
            tree_data = pd.read_csv(os.path.join(self.output_dir, self.plotID + '_tree_data_edit.csv'))
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

        DBH_x = np.array(tree_data['TreeLocation_X']) - self.plot_centre[0]  #float(self.plot_summary['Plot Centre X'])
        DBH_y = np.array(tree_data['TreeLocation_Y']) - self.plot_centre[1]  # float(self.plot_summary['Plot Centre Y'])

        tree_data['Distance'] = np.around(np.sqrt(DBH_x**2 + DBH_y**2),2)

        # tree_data.droplevel
        bearing = np.round(np.degrees(np.arctan2(DBH_x, DBH_y))) % 360
        tree_data['Bearing'] = np.array(bearing.flatten())
        tree_data = tree_data.sort_values(by='Bearing')
        
        # reset the ids
        tree_data['TreeNumber'] = np.arange(len(tree_data))+1

        # tree_data['TreeLocation_Z'] = DBH_H
        tree_data['PlotId'] = self.plotID
        tree_data.to_csv(self.output_dir + self.plotID + '_tree_data_report.csv', index=None, sep=',')

        # taper_data = pd.read_csv(self.output_dir + self.plotID + '_taper_data.csv')
        # taper_data['TreeNumber'] = np.argsort(bearing.flatten()) + 1  # use the index+1 of the sorted bearing as TreeNumber
        # df = taper_data.sort_values(by = 'TreeNumber')
        # df.to_csv(self.output_dir + self.plotID + '_taper_data_sorted.csv', index=None, sep=',') 

        # if os.path.isfile(self.output_dir + self.plotID + '_taper_data_sorted.csv') :
        #     os.remove(self.output_dir + self.plotID + '_taper_data.csv')

        return tree_data


    def customize_data(self):
        # for various manually edited data files

        try:
            tree_data = pd.read_csv(os.path.join(self.output_dir, self.plotID + '_QC_data_fieldID.csv'))
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
        field_df.to_csv(self.output_dir + self.plotID + f'_tree_data_report{QC}.csv', index=None, sep=',')
        
        return tree_data