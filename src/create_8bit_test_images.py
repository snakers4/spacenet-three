import pandas as pd
from multiprocessing import Pool
import tqdm
import numpy as np
import os
import glob as glob
from skimage.io import imread, imsave
import osmnx as ox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal, ogr, osr
import cv2
import subprocess
import shapely
from shapely.geometry import MultiLineString
from matplotlib.patches import PathPatch
import matplotlib.path

imgs = []

# change this to your data prefix
path_prefix = '../data'

# default variables from the hosts of the challenge
buffer_meters = 2
burnValue = 150

# only test folders
folders = ['AOI_2_Vegas_Roads_Test_Public',
                'AOI_3_Paris_Roads_Test_Public',
                'AOI_4_Shanghai_Roads_Test_Public',
                'AOI_5_Khartoum_Roads_Test_Public']

# image types
prefix_dict = {
    'mul': 'MUL',
    'muls': 'MUL-PanSharpen',
    'pan': 'PAN',
    'rgbps': 'RGB-PanSharpen',    
}

for folder in folders:
    for prefix in prefix_dict.items():
        g = glob.glob(path_prefix+'/{}/{}/*.tif'.format(folder,prefix[1]))
        imgs.extend(g)
        
img_folders = [(img.split('/')[2]) for img in imgs]
img_subfolders = [(img.split('/')[3]) for img in imgs]   
img_files = [(img.split('/')[4]) for img in imgs]   

def create_8bit_test_images(input_data):
    img_path = input_data[0]  
    img_folder = input_data[1]
    img_subfolder = input_data[2]
    img_file = input_data[3]
 
    # create paths for masks and 8bit images
    bit8_folder = os.path.join(path_prefix,img_folder,img_subfolder+'_8bit')
    bit8_path = os.path.join(bit8_folder,img_file)

    if not os.path.exists(bit8_folder):
        os.mkdir(bit8_folder)
    if os.path.isfile(bit8_path):
        os.remove(bit8_path)
        
    try:
        # convert images to 8-bit
        convert_to_8Bit(img_path, 
                        bit8_path,
                        outputPixType='Byte',
                        outputFormat='GTiff',
                        rescale_type='rescale',
                        percentiles=[2,98])

    except BaseException as e:
        print(str(e))
        
    return [bit8_folder,bit8_path,img_path,img_folder,img_subfolder,img_file]

def convert_to_8Bit(inputRaster, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           rescale_type='rescale',
                           percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done strictly between 0 65535 
        if rescale, each band is rescaled to a min and max 
        set by percentiles
    '''

    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', 
           outputFormat]
    
    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()        
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(), 
                                 percentiles[0])
            bmax= np.percentile(band_arr_tmp.flatten(), 
                                percentiles[1])

        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    # print("Conversin command:", cmd)
    subprocess.call(cmd)
    
    return

input_data = zip(imgs,img_folders,img_subfolders,img_files)
input_data = [item for item in input_data]

with Pool(10) as p:
    bit8_data = list(tqdm.tqdm(p.imap(create_8bit_test_images, input_data),
                                   total=len(input_data)))
# transpose the list
bit8_data = list(map(list, zip(*bit8_data)))

bit8_df = pd.DataFrame()

for i,key in enumerate(['bit8_folder','bit8_path','img_path','img_folder','img_subfolder','img_file']):
    bit8_df[key] = bit8_data[i]

bit8_df.to_csv('bit8_test_run.csv')

