import rasterio
import ast
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte,img_as_float
from skimage import exposure
from skimage.draw import circle
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from collections import Sequence
from itertools import chain, count
import os
import pandas as pd

data_prefix = '../data'
geojson_df = pd.read_csv('geojson_df_full.csv')
meta_df = pd.read_csv('../metadata.csv')

# manual curation of the dataset
# all unpaved roads with 2+ lanes set to paved
geojson_df.loc[(geojson_df.lane_number>2)&(geojson_df.paved == 2),'paved'] = 1

# a function to understand list depth
def depth(seq):
    for level in count():
        if not seq:
            return level
        seq = list(chain.from_iterable(s for s in seq if isinstance(s, Sequence)))

def read_image(preset,path):
    img = imread(path)
    target_channels = np.zeros(shape=(preset['width'],preset['width'],len(preset['channels'])))

    # expand grayscale images to 3 dimensions
    if len(img.shape)<3:
        img = np.expand_dims(img, 2)                

    for i,channel in enumerate(preset['channels']):
        target_channels[:,:,i] = img[:,:,channel-1]

    # target_channels = img_as_ubyte(target_channels)
    # target_channels = exposure.rescale_intensity(target_channels, in_range='uint8')           

    return target_channels
    
def draw_mask(circle_size,
              line_width,
              ls_list,
              mask_size
             ):

    mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    all_points = []
    
    for line in ls_list:

        points_xy = line
        
        all_points.extend(points_xy)
        
        for i,[x,y] in enumerate(points_xy):
            if i-1>-1:
                prev_x = int(float(points_xy[i-1][0]))
                prev_y = int(float(points_xy[i-1][1]))
                mask = cv2.line(mask,(prev_x,prev_y),(int(float(x)),int(float(y))),(150),line_width)
                
        all_points_text = [(str(point[0])+' '+str(point[1])) for point in all_points]
        count_dict =  Counter(all_points_text)

        for key, value in count_dict.items():
            if(value>1):
                x,y = key.split()
                rr, cc = circle(int(float(y)), int(float(x)), circle_size)
                mask[rr.clip(min=0,max=mask_size-1), cc.clip(min=0,max=mask_size-1)] = 255
    return mask

def draw_mask_width(
              ls_list,
              mask_size
             ):

    width_per_lane = 10
    mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    all_points = []
    
    for line in ls_list:

        points_xy = line
        all_points.extend(points_xy)
        
        for i,[x,y,lanes] in enumerate(points_xy):
            if i-1>-1:
                prev_x = int(float(points_xy[i-1][0]))
                prev_y = int(float(points_xy[i-1][1]))
                mask = cv2.line(mask,(prev_x,prev_y),
                                (int(float(x)),int(float(y))),
                                (255),
                                lanes*width_per_lane)
    return mask

def draw_intersections(circle_size,
              ls_list,
              mask_size
             ):

    mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    all_points = []
    
    for line in ls_list:
        points_xy = line
        all_points.extend(points_xy)
        all_points_text = [(str(point[0])+' '+str(point[1])) for point in all_points]
        count_dict =  Counter(all_points_text)
        
        for key, value in count_dict.items():
            if(value>1):
                x,y = key.split()
                rr, cc = circle(int(float(y)), int(float(x)), circle_size)
                mask[rr.clip(min=0,max=mask_size-1), cc.clip(min=0,max=mask_size-1)] = 255
    return mask

def draw_masks(paved = 2,
               road_type = 6,
               lane_number = 6):
    
    random_image = geojson_df[(geojson_df.paved == paved)
                              # &(geojson_df.road_type == road_type)
                              &(geojson_df.lane_number == lane_number)].sample(n=1).img_id.values[0]

    sample_df = geojson_df[(geojson_df.paved == paved)
              # &(geojson_df.road_type == road_type)
              &(geojson_df.lane_number == lane_number)
              &(geojson_df.img_id == random_image)]
    
    rgb_ps_image = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])

    mask2_path = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen_mask',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])[:-3]+'jpg'


    src = rasterio.open(rgb_ps_image)
    ls_list = [ast.literal_eval(ls) for ls in sample_df.linestring.values]
    ls_list_image = []

    for line in ls_list:
        points = []
        for point in line:
            points.append(~src.affine * (point[0],point[1]))
        ls_list_image.append(points)

    img = read_image(preset_dict['rgb_ps'],rgb_ps_image)

    mask = draw_mask(circle_size=15,
                  line_width=15,
                  ls_list=ls_list_image,
                  mask_size=1300
                 )

    mask2 = imread(mask2_path)

    fig=plt.figure(figsize=(20, 6))

    fig.add_subplot(1, 3, 1)
    img += -img.min()
    img *= (1/img.max())
    plt.imshow(img)

    fig.add_subplot(1, 3, 2)
    plt.imshow(mask)

    fig.add_subplot(1, 3, 3)
    plt.imshow(mask2)

    plt.show()     
    
def draw_masks_analyze_road_types():
    road_types = [3,5,6,7]    
    
    # select a random image and the gt
    random_image = geojson_df.sample(n=1).img_id.values[0]
    
    mask2_path = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen_mask',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])[:-3]+'jpg'
    
    mask2 = imread(mask2_path)    
    
    rgb_ps_image = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])
    
    fig=plt.figure(figsize=(25, 10))
    
    img = read_image(preset_dict['rgb_ps'],rgb_ps_image)    
    fig.add_subplot(2, 5, 1)
    img += -img.min()
    img *= (1/img.max())
    plt.imshow(img)
    
    fig.add_subplot(2, 5, 6)
    plt.imshow(mask2)    
    
    for i,road_type in enumerate(road_types):
       
        sample_df = geojson_df[((geojson_df.road_type == road_type)
                                &(geojson_df.img_id == random_image))]

        src = rasterio.open(rgb_ps_image)
        ls_list = [ast.literal_eval(ls) for ls in sample_df.linestring.values]
        ls_list_image = []

        for line in ls_list:
            points = []
            for point in line:
                points.append(~src.affine * (point[0],point[1]))
            ls_list_image.append(points)

        mask = draw_mask(circle_size=15,
                      line_width=15,
                      ls_list=ls_list_image,
                      mask_size=1300
                     )

        fig.add_subplot(2, 5, i+2)
        plt.imshow(mask)

    plt.show()
 
def process_ls(sample_df,lane_numbers):
    if lane_numbers == None:
        ls_list = []
        for ls in sample_df.linestring.values:
            if depth(ast.literal_eval(ls)) == 2:
                ls_list.append(ast.literal_eval(ls))
            elif depth(ast.literal_eval(ls)) == 3:
                for item in ast.literal_eval(ls):
                    ls_list.append(item)
            else:
                raise ValueError('Wrong linestring format')
        return ls_list,None
    else:
        ls_list = []
        lane_numbers_new = []
        
        i = 0
        for ls in sample_df.linestring.values:
            if depth(ast.literal_eval(ls)) == 2:
                ls_list.append(ast.literal_eval(ls))
                lane_numbers_new.append(lane_numbers[i])
            elif depth(ast.literal_eval(ls)) == 3:
                for item in ast.literal_eval(ls):
                    ls_list.append(item)
                    lane_numbers_new.append(lane_numbers[i])
            else:
                raise ValueError('Wrong linestring format')
            i+=1
        return ls_list,lane_numbers_new
        
def draw_masks_new():
    paved_types = [1,2]
    
    # select a random image and the gt
    random_image = geojson_df.sample(n=1).img_id.values[0]
    # random_image = 'AOI_4_Shanghai_img823'
    
    mask2_path = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen_mask',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])[:-3]+'jpg'
    
    mask2 = imread(mask2_path)    
    
    rgb_ps_image = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='RGB-PanSharpen')].img_files.values[0],
                                                    'RGB-PanSharpen',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='RGB-PanSharpen')].img_subfolders.values[0])
    
    fig=plt.figure(figsize=(20, 15))
    
    img = read_image(preset_dict['rgb_ps'],rgb_ps_image)    
    fig.add_subplot(2, 3, 1)
    img += -img.min()
    img *= (1/img.max())
    plt.imshow(img)
    
    fig.add_subplot(2, 3, 5)
    plt.imshow(mask2)    
    
    # draw intersections only
    sample_df = geojson_df[((geojson_df.img_id == random_image))]
    
    src = rasterio.open(rgb_ps_image)
    lane_numbers = None
    ls_list,lane_numbers = process_ls(sample_df,lane_numbers)
    ls_list_image = []
    
    for line in ls_list:
        points = []
        for point in line:
            points.append(~src.affine * (point[0],point[1]))
        ls_list_image.append(points)

    intersections_mask = draw_intersections(circle_size=15,
                                  ls_list=ls_list_image,
                                  mask_size=1300)
    
   
    fig.add_subplot(2, 3, 6)
    plt.imshow(intersections_mask)        
        
    for i,paved_type in enumerate(paved_types):
       
        sample_df = geojson_df[((geojson_df.paved == paved_type)
                                &(geojson_df.img_id == random_image))]
        
        lane_numbers = list(sample_df.lane_number.values)

        src = rasterio.open(rgb_ps_image)
        ls_list,lane_numbers = process_ls(sample_df,lane_numbers)
        ls_list_image = []
      
        for j,line in enumerate(ls_list):
            points = []
            for point in line:
                # create pixel coordinates
                # add lane width to the tuple
                pixel_coordinates = ~src.affine * (point[0],point[1])
                points.append(pixel_coordinates + (lane_numbers[j],))
            ls_list_image.append(points)

        mask = draw_mask_width(
                      ls_list=ls_list_image,
                      mask_size=1300
                     )

        fig.add_subplot(2, 3, i+2)
        plt.imshow(mask)

    plt.show()
    
def create_new_masks(random_image):
    paved_types = [1,2]
    
    rgb_ps_image = os.path.join(data_prefix, meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                     &(meta_df.img_folders=='PAN')].img_files.values[0],
                                                    'PAN',
                                                    meta_df[(meta_df.img_subfolders.str.contains(random_image))
                                                            &(meta_df.img_folders=='PAN')].img_subfolders.values[0])

    # draw intersections only
    sample_df = geojson_df[((geojson_df.img_id == random_image))]
    
    
    src = rasterio.open(rgb_ps_image)
    lane_numbers = None
    ls_list,lane_numbers = process_ls(sample_df,lane_numbers)
    ls_list_image = []

    for line in ls_list:
        points = []
        for point in line:
            points.append(~src.affine * (point[0],point[1]))
        ls_list_image.append(points)
    
    intersections_mask = draw_intersections(circle_size=15,
                                  ls_list=ls_list_image,
                                  mask_size=1300)
    
    road_masks = []
    for i,paved_type in enumerate(paved_types):
       
        sample_df = geojson_df[((geojson_df.paved == paved_type)
                                &(geojson_df.img_id == random_image))]
        
        lane_numbers = list(sample_df.lane_number.values)

        src = rasterio.open(rgb_ps_image)
        ls_list,lane_numbers = process_ls(sample_df,lane_numbers)
        ls_list_image = []
        
        for j,line in enumerate(ls_list):
            points = []
            for point in line:
                # create pixel coordinates
                # add lane width to the tuple
                pixel_coordinates = ~src.affine * (point[0],point[1])
                points.append(pixel_coordinates + (lane_numbers[j],))
            ls_list_image.append(points)

        road_masks.append(draw_mask_width(
                      ls_list=ls_list_image,
                      mask_size=1300
                     ))
    return intersections_mask,road_masks[0],road_masks[1]