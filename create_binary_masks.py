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
path_prefix = 'data'

# default variables from the hosts of the challenge
buffer_meters = 2
burnValue = 150

# only train folders
folders = ['AOI_2_Vegas_Roads_Train',
           'AOI_5_Khartoum_Roads_Train',
           'AOI_3_Paris_Roads_Train',
           'AOI_4_Shanghai_Roads_Train']

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
        
img_folders = [(img.split('/')[1]) for img in imgs]
img_subfolders = [(img.split('/')[2]) for img in imgs]   
img_files = [(img.split('/')[3]) for img in imgs]   

def create_binary_mask(input_data):
    img_path = input_data[0]  
    img_folder = input_data[1]
    img_subfolder = input_data[2]
    img_file = input_data[3]
 
    # create paths for masks and 8bit images
    label_file = os.path.join(path_prefix,img_folder,'geojson/spacenetroads','spacenetroads_AOI'+img_file.split('AOI')[1][0:-3]+'geojson')
    bit8_folder = os.path.join(path_prefix,img_folder,img_subfolder+'_8bit')
    bit8_path = os.path.join(bit8_folder,img_file)
    mask_folder = os.path.join(path_prefix,img_folder,img_subfolder+'_mask')
    mask_path = os.path.join(mask_folder,img_file[:-3])+'png'
    # vis_folder = os.path.join(path_prefix,img_folder,img_subfolder+'_vis')
    # vis_path = os.path.join(vis_folder,img_file[:-3])+'png'
    
    # print(label_file)
    # create the necessary folders and remove the existing files
    
    if not os.path.exists(bit8_folder):
        os.mkdir(bit8_folder)
    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)
    # if not os.path.exists(vis_folder):
    #     os.mkdir(vis_folder)        
    # if os.path.isfile(vis_path):
    #    os.remove(vis_path)
    if os.path.isfile(bit8_path):
        os.remove(bit8_path)        
    if os.path.isfile(mask_path):
        os.remove(mask_path)
    if os.path.isfile(mask_path[:-3]+'jpg'):
        os.remove(mask_path[:-3]+'jpg')               
    
    try:
        # convert images to 8-bit
        convert_to_8Bit(img_path, 
                        bit8_path,
                        outputPixType='Byte',
                        outputFormat='GTiff',
                        rescale_type='rescale',
                        percentiles=[2,98])

        # create masks
        # note that though the output raster file has .png extension
        # in reality I delete this file and save only jpg version later

        mask, gdf_buffer = get_road_buffer(geoJson = label_file,
                                          im_vis_file = bit8_path, 
                                          output_raster = mask_path, 
                                          buffer_meters= buffer_meters, 
                                          burnValue= burnValue, 
                                          bufferRoundness=6, 
                                          plot_file='', # this indicates that no visualization plot is required 
                                          figsize= (6,6),
                                          fontsize=8,
                                          dpi=200,
                                          show_plot=False, 
                                          verbose=False)

        # read the png file, save it as jpeg and 
        mask = imread(mask_path)
        imsave(fname=mask_path[:-3]+'jpg',arr = mask)
        mask_max = np.max(mask) 
        del mask
        # remove the png file, but keep the 8-bit mask
        os.remove(mask_path)
    except BaseException as e:
        print(str(e))
        mask_max = -1
        
    return [label_file,bit8_folder,bit8_path,mask_folder,mask_path[:-3]+'jpg',img_path,img_folder,img_subfolder,img_file,mask_max]

def get_road_buffer(geoJson, im_vis_file, output_raster, 
                              buffer_meters=2, burnValue=1, 
                              bufferRoundness=6, 
                              plot_file='', figsize=(6,6), fontsize=6,
                              dpi=800, show_plot=False, 
                              verbose=False):    
    '''
    Get buffer around roads defined by geojson and image files.
    Calls create_buffer_geopandas() and gdf_to_array().
    Assumes in_vis_file is an 8-bit RGB file.
    Returns geodataframe and ouptut mask.
    '''

    gdf_buffer = create_buffer_geopandas(geoJson,
                                         bufferDistanceMeters=buffer_meters,
                                         bufferRoundness=bufferRoundness, 
                                         projectToUTM=True)    

    
    # create label image
    if len(gdf_buffer) == 0:
        mask_gray = np.zeros(cv2.imread(im_vis_file,0).shape)
        cv2.imwrite(output_raster, mask_gray)        
    else:
        gdf_to_array(gdf_buffer, im_vis_file, output_raster, 
                                          burnValue=burnValue)
    
    # load mask
    mask_gray = cv2.imread(output_raster, 0)
    
    # make plots
    if plot_file:
        # plot all in a line
        if (figsize[0] != figsize[1]):
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4, figsize=figsize)#(13,4))
        # else, plot a 2 x 2 grid
        else:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=figsize)
    
        # road lines
        try:
            gdfRoadLines = gpd.read_file(geoJson)
            gdfRoadLines.plot(ax=ax0, marker='o', color='red')
        except:
            ax0.imshow(mask_gray)
        ax0.axis('off')
        ax0.set_aspect('equal')
        ax0.set_title('Roads from GeoJson', fontsize=fontsize)
                
        # first show raw image
        im_vis = cv2.imread(im_vis_file, 1)
        img_mpl = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_mpl)
        ax1.axis('off')
        ax1.set_title('8-bit RGB Image', fontsize=fontsize)
        
        # plot mask
        ax2.imshow(mask_gray)
        ax2.axis('off')
        ax2.set_title('Roads Mask (' + str(np.round(buffer_meters)) \
                                   + ' meter buffer)', fontsize=fontsize)
     
        # plot combined
        ax3.imshow(img_mpl)    
        # overlay mask
        # set zeros to nan
        z = mask_gray.astype(float)
        z[z==0] = np.nan
        # change palette to orange
        palette = plt.cm.gray
        #palette.set_over('yellow', 0.9)
        palette.set_over('lime', 0.9)
        ax3.imshow(z, cmap=palette, alpha=0.66, 
                norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
        ax3.set_title('8-bit RGB Image + Buffered Roads', fontsize=fontsize) 
        ax3.axis('off')
        
        #plt.axes().set_aspect('equal', 'datalim')

        plt.tight_layout()
        plt.savefig(plot_file, dpi=dpi)
        if not show_plot:
            plt.close()
            
    return mask_gray, gdf_buffer

def create_buffer_geopandas(geoJsonFileName,
                            bufferDistanceMeters=2, 
                            bufferRoundness=1,
                            projectToUTM=True):
    '''
    Create a buffer around the lines of the geojson. 
    Return a geodataframe.
    '''
    
    inGDF = gpd.read_file(geoJsonFileName)
    
    # set a few columns that we will need later
    inGDF['type'] = inGDF['road_type'].values            
    inGDF['class'] = 'highway'  
    inGDF['highway'] = 'highway'  
    
    if len(inGDF) == 0:
        return [], []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = ox.project_gdf(inGDF)
    else:
        tmpGDF = inGDF

    gdf_utm_buffer = tmpGDF

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = tmpGDF.buffer(bufferDistanceMeters,
                                                bufferRoundness)

    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by='class')
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs

    if projectToUTM:
        gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    else:
        gdf_buffer = gdf_utm_dissolve

    return gdf_buffer

def gdf_to_array(gdf, im_file, output_raster, burnValue=150):
    
    '''
    Turn geodataframe to array, save as image file with non-null pixels 
    set to burnValue
    '''

    NoData_value = 0      # -9999

    gdata = gdal.Open(im_file)
    
    # set target info
    target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, 
                                                     gdata.RasterXSize, 
                                                     gdata.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    
    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    
    outdriver=ogr.GetDriverByName('MEMORY')
    outDataSource=outdriver.CreateDataSource('memData')
    tmp=outdriver.Open('memData',1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs, 
                                         geom_type=ogr.wkbMultiPolygon)
    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()
    for geomShape in gdf['geometry'].values:
        
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        outFeature.SetField(burnField, burnValue)
        outLayer.CreateFeature(outFeature)
        outFeature = 0
    
    gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[burnValue])
    outLayer = 0
    outDatSource = 0
    tmp = 0
        
    return 

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
    mask_data = list(tqdm.tqdm(p.imap(create_binary_mask, input_data),
                                   total=len(input_data)))
# transpose the list
mask_data = list(map(list, zip(*mask_data)))

mask_df = pd.DataFrame()

for i,key in enumerate(['label_file','bit8_folder','bit8_path','mask_folder','mask_path','img_path','img_folder','img_subfolder','img_file', 'mask_max']):
    mask_df[key] = mask_data[i]

mask_df.to_csv('mask_df.csv')
