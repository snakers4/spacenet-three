import os
import time
import numpy as np
import pandas as pd
from skimage.io import imread
import torch.utils.data as data

import warnings
warnings.filterwarnings('ignore')

# relative dataset storage path for test dataset
prefix = '../data'
# relative dataset storage path for train dataset
prefix_train = '../'
# metadata and mask_df files
meta_prefix = '../'
meta_data_file = os.path.join(meta_prefix,'metadata.csv')
mask_df_file = os.path.join(meta_prefix,'mask_df.csv')


# high level function that return list of images and cities under presets
def get_test_dataset(preset,
                     preset_dict):
    meta_df = pd.read_csv(meta_data_file)
    
    test_folders = ['AOI_2_Vegas_Roads_Test_Public','AOI_5_Khartoum_Roads_Test_Public',
               'AOI_3_Paris_Roads_Test_Public','AOI_4_Shanghai_Roads_Test_Public']
    
    # select the images
    sample_df = meta_df[(meta_df.img_files.isin(test_folders))
                        &(meta_df.width == preset_dict[preset]['width'])
                        &(meta_df.channels == preset_dict[preset]['channel_count'])
                        &(meta_df.img_folders == preset_dict[preset]['subfolder'])]

    # get the data as lists for simplicity
    or_imgs = list(sample_df[['img_subfolders','img_files','img_folders']]
                   .apply(lambda row: os.path.join(prefix,row['img_files'],row['img_folders']+'_8bit',row['img_subfolders']), axis=1).values)

    le, u = sample_df['img_folders'].factorize()
    sample_df.loc[:,'city_no'] = le
    cty_no = list(sample_df.city_no.values)
    
    city_folders = list(sample_df.img_files.values)
    img_names = list(sample_df.img_subfolders.values)
    
    return or_imgs,city_folders,img_names,cty_no,prefix

# high level function that return list of images and cities under presets
def get_train_dataset(preset,
                     preset_dict):
    mask_df = pd.read_csv(mask_df_file)
    meta_df = pd.read_csv(meta_data_file)
    data_df = mask_df.merge(meta_df[['img_subfolders','width','channels']], how = 'left', left_on = 'img_file', right_on = 'img_subfolders')
    
    # select the images
    sample_df = data_df[(data_df.width == preset_dict[preset]['width'])
                        &(data_df.mask_max > 0)
                        &(data_df.channels == preset_dict[preset]['channel_count'])
                        &(data_df.img_subfolder == preset_dict[preset]['subfolder'])]

    # get the data as lists for simplicity
    bit8_imgs = list(sample_df.bit8_path.values)
    bit8_masks = list(sample_df.mask_path.values)
    bit8_imgs = [(os.path.join(prefix_train,path)) for path in bit8_imgs]
    bit8_masks = [(os.path.join(prefix_train,path)) for path in bit8_masks]
    le, u = sample_df['img_folder'].factorize()
    sample_df.loc[:,'city_no'] = le
    cty_no = list(sample_df.city_no.values)
    
    return bit8_imgs,bit8_masks,cty_no

# dataset class
class SatellitesDataset(data.Dataset):
    def __init__(self,
                 preset,
                 image_paths = [],
                 mask_paths = None,                 
                 transforms = None,
                 ):
        
        self.mask_paths = mask_paths
        self.preset = preset
        self.transforms = transforms
        
        if mask_paths is not None:
            self.image_paths = sorted(image_paths)
            self.mask_paths = sorted(mask_paths)

            if len(self.image_paths) != len(mask_paths):
                raise ValueError('Mask list length <> image list lenth')
            if [path.split('/')[4].split('img')[1].split('.')[0] for path in self.image_paths] != [path.split('/')[4].split('img')[1].split('.')[0] for path in self.mask_paths]:            
                 raise ValueError('Mask list sorting <> image list sorting')
        else:
            self.image_paths = sorted(image_paths)
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.mask_paths is not None: 

            img = imread(self.image_paths[idx])
            target_channels = np.zeros(shape=(self.preset['width'],self.preset['width'],len(self.preset['channels'])))
            
            # expand grayscale images to 3 dimensions
            if len(img.shape)<3:
                img = np.expand_dims(img, 2)                
            
            for i,channel in enumerate(self.preset['channels']):
                target_channels[:,:,i] = img[:,:,channel-1]
            
            target_channels = target_channels.astype('uint8')
            
            mask = imread(self.mask_paths[idx])
            mask = mask.astype('uint8')
            
            if self.transforms is not None:
                 target_channels, mask = self.transforms(target_channels, mask)
            
            return target_channels,mask                    

        else:
            img = imread(self.image_paths[idx])
            target_channels = np.zeros(shape=(self.preset['width'],self.preset['width'],len(self.preset['channels'])))
            
            for i,channel in enumerate(self.preset['channels']):
                target_channels[:,:,i] = img[:,:,channel-1]
            
            target_channels = target_channels.astype('uint8')
            
            if self.transforms is not None:
                 target_channels, _ = self.transforms(target_channels, None)
            return target_channels
