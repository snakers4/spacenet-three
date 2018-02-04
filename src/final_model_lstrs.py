import pandas as pd
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from skimage.io import imread
from tqdm import tqdm
import skimage.transform
import math
from skimage.morphology import skeletonize
import sknw
import cv2
from collections import Counter
from skimage.draw import circle
import os

#globbing
root = '../data'
bit8_folders2 = ['RGB-PanSharpen_8bit']
test_folders1 = ['AOI_2_Vegas_Roads_Test_Public', 
                 'AOI_3_Paris_Roads_Test_Public', 
                 'AOI_4_Shanghai_Roads_Test_Public', 
                 'AOI_5_Khartoum_Roads_Test_Public']
mask_folders2_test_pad = ['norm_ln34_mul_ps_vegetation_aug_dice_predict']

globdf_test = pd.DataFrame()
for test_folder in test_folders1:
    
    for bit8_folder in bit8_folders2:
        local = pd.DataFrame()
        bit8_folder_path = '{}/{}/{}'.format(root, test_folder, bit8_folder)
        local['bit8_img'] = glob.glob('{}/*.tif'.format(bit8_folder_path))
        local['bit8_folder'] = bit8_folder
        local['test_folder'] = test_folder
        globdf_test = pd.concat([globdf_test, local],ignore_index = True)
        del local
        
globdf_masks_test_pad = pd.DataFrame()
for test_folder in test_folders1:
    
    
    for mask_folder_test in mask_folders2_test_pad:
        local = pd.DataFrame()
        mask_folder_test_path = '{}/{}/{}'.format(root, test_folder, mask_folder_test)
        local['mask_img'] = glob.glob('{}/*.jpg'.format(mask_folder_test_path))
        local['mask_folder_test'] = mask_folder_test
        local['test_folder'] = test_folder
        globdf_masks_test_pad = pd.concat([globdf_masks_test_pad, local],ignore_index = True)
        del local

globdf_test['bit8_name'] = globdf_test['bit8_img'].apply(lambda x: x.split('/')[len(x.split('/')) - 1])
globdf_test['img_id'] = globdf_test['bit8_name'].apply(lambda x: x[x.find('AOI'):x.find('.')])

globdf_masks_test_pad['mask_name'] = globdf_masks_test_pad['mask_img'].apply(lambda x: x.split('/')[len(x.split('/')) - 1])
globdf_masks_test_pad['img_id'] = globdf_masks_test_pad['mask_name'].apply(lambda x: x[x.find('AOI'):x.find('.')])

globdf_test_pad = globdf_test.merge(globdf_masks_test_pad, how = 'left', on = ['test_folder', 'img_id'])

# functions
def simplify_edge(ps: np.ndarray, max_distance=3):
    """
    Combine multiple points of graph edges to line segments
    so distance from points to segments <= max_distance
    :param ps: array of points in the edge, including node coordinates
    :param max_distance: maximum distance, if exceeded new segment started
    :return: ndarray of new nodes coordinates
    """
    res_points = []
    cur_idx = 0
    # combine points to the single line while distance from the line to any point < max_distance
    for i in range(1, len(ps) - 1):
        segment = ps[cur_idx:i + 1, :] - ps[cur_idx, :]
        angle = -math.atan2(segment[-1, 1], segment[-1, 0])
        ca = math.cos(angle)
        sa = math.sin(angle)
        # rotate all the points so line is alongside first column coordinate
        # and the second col coordinate means the distance to the line
        segment_rotated = np.array([[ca, -sa], [sa, ca]]).dot(segment.T)
        distance = np.max(np.abs(segment_rotated[1, :]))
        if distance > max_distance:
            res_points.append(ps[cur_idx, :])
            cur_idx = i
    if len(res_points) == 0:
        res_points.append(ps[0, :])
    res_points.append(ps[-1, :])

    return np.array(res_points)

def simplify_graph(graph, max_distance=2):
    """
    :type graph: MultiGraph
    """
    all_segments = []
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            ps = val['pts']
            full_segments = np.row_stack([
                graph.node[s]['o'],
                ps,
                graph.node[e]['o']
            ])

            segments = simplify_edge(full_segments, max_distance=max_distance)
            all_segments.append(segments)

    return all_segments

def segment_to_linestring(segment):
    
    if len(segment) < 2:
        return []
    
    linestring = 'LINESTRING ({})'
    sublinestring = ''
        
    for i, node in enumerate(segment):
        
        if i == 0:
            sublinestring = sublinestring + '{:.1f} {:.1f}'.format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ', {:.1f} {:.1f}'.format(node[1], node[0])
    linestring = linestring.format(sublinestring)
    return linestring

def segmets_to_linestrings(segments):
    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if len(linestrings) == 0:
        linestrings = ['LINESTRING EMPTY']
    return linestrings

def process_masks(mask_paths):
    lnstr_df = pd.DataFrame()
    with tqdm(total=len(mask_paths)) as pbar:
        for msk_pth in mask_paths:
            #print(msk_pth)
            msk = imread(msk_pth)
            msk = msk[6:1306, 6:1306]
            msk_nme = msk_pth.split('/')[3]
            img_id = msk_nme[msk_nme.find('AOI'):msk_nme.find('.')]

            # open and skeletonize
            thresh = 30
            binary = (msk > thresh)*1
            
            ske = skeletonize(binary).astype(np.uint16)

            # build graph from skeleton
            graph = sknw.build_sknw(ske, multi=True)
            segments = simplify_graph(graph)

            linestrings = segmets_to_linestrings(segments)
            local = pd.DataFrame()
            local['WKT_Pix'] = linestrings
            local['ImageId'] = img_id

            lnstr_df = pd.concat([lnstr_df, local], ignore_index = True)
            pbar.update(1)
    return lnstr_df

# calculating result
print('Processing masks into linestrings...')

globdf_test_narrow_vegetation = globdf_test_pad[globdf_test_pad['mask_folder_test'] == 'norm_ln34_mul_ps_vegetation_aug_dice_predict'].copy()
lstrs_test = process_masks(globdf_test_narrow_vegetation.mask_img)

lstrs_test = lstrs_test[['ImageId', 'WKT_Pix']].copy()
lstrs_test = lstrs_test.drop_duplicates()
os.makedirs('../solutions', exist_ok=True)
lstrs_test.to_csv('../solutions/norm_test.csv', index = False)
print('Resulting norm_test.csv file is saved under ../solutions/ directory')