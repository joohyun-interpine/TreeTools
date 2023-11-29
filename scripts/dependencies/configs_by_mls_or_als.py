
class Dependencies:
    def __init__(self) -> None:
        pass
    
    def mobile_laser_scanning(self):
        mls =  {"denoise_stem_points":True, 
                "noise_intensity_threshold":10,
                "SOR_filter" : True, # applies Statistical outlier removal on the slices of stem points
                "tree_base_cutoff_height":2.8, # the lowest stem measurement to qualify as a tree (takes care of hanging dead branches)
                "tree_stem_min_height":4., # minimum stem height to qualify for a crop tree (takes care of stumps, undergrowth and non-crop trees)
                "gap_connect":.15, # used in dbscan to connect skeleton points into tree skeleton
                "subsample":0,
                "headers":['x', 'y', 'z', 'intensity','return_number','gps_time','Ring','Range']
                }
        
        return mls
    

    def airborne_laser_scanning(self):
        als =  {"denoise_stem_points":False, # Aerial data has fewer stem points so we are keeping then all
                "tree_base_cutoff_height":13.1, # # to ignore unassigned branches
                "gap_connect":.5, #used in dbscan to connect skeleton points into tree skeleton
                "subsample":1,  # Aerial data has very dense foliage - processing takes ages, therefore subsample
                "headers":['x', 'y', 'z', 'intensity','return_number','number_of_returns','classification', '']
                }
        
        return als