import numpy as np
import math
from skimage.io import imread
import matplotlib.pyplot as plt
import skimage.morphology
import sknw

DILATION_SIZE = 4
MAX_DISTANCE_FROM_LINE_TO_SPLIT = 2
CLIP_THRESHOLD = 0.3


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


def draw_original_graph(graph):
    # draw original points in green/red
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            ps = val['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

            for node in (graph.node[s], graph.node[e]):
                plt.plot(node['o'][1], node['o'][0], 'r.')


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


if __name__ == '__main__':
    # mask_fn = '../data/ln34_wide_masks_mul_ps_vegetation_aug_dice_predict_train/MUL-PanSharpen_AOI_2_Vegas_img1353.jpg'
    mask_fn = '../data/ln34_wide_masks_mul_ps_vegetation_aug_dice_predict_pad/MUL-PanSharpen_AOI_2_Vegas_img1353.jpg'

    img = imread(mask_fn, as_grey=True)
    print(np.min(img), np.max(img))

    img = skimage.morphology.dilation(img, selem=skimage.morphology.disk(DILATION_SIZE))

    img_clip = np.zeros_like(img)
    img_clip[img > 255 * CLIP_THRESHOLD] = 1

    ske = skimage.morphology.skeletonize(img_clip).astype(np.uint16)
    graph = sknw.build_sknw(ske, multi=True)
    all_segments = simplify_graph(graph, max_distance=MAX_DISTANCE_FROM_LINE_TO_SPLIT)

    plt.imshow(img)
    draw_original_graph(graph)
    for segments in all_segments:
        plt.plot(segments[:, 1], segments[:, 0], 'blue', marker='.')
    plt.show()
