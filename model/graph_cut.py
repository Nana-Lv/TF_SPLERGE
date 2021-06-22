#-*-coding:utf-8-*-
import numpy as np
import cv2
import networkx as nx


def graph_cut(prob, weight=0.75):
    G = nx.Graph()
    G.add_node('s')
    G.add_node('t')
    G.add_nodes_from([str(x) for x in range(len(prob))])
    w = []
    for i in range(len(prob)):
        if i < len(prob) - 1:
            w.append((str(i), str(i + 1), weight))
        w.append((str(i), 's', prob[i]))
        w.append((str(i), 't', 1 - prob[i]))
    G.add_weighted_edges_from(w, weight='weight')
    minimum_cut, node_set = nx.minimum_cut(G, 's', 't', capacity='weight')
    node_s, node_t = node_set
    res_row = len(prob) * [0]
    for n in node_s:
        if n != 's':
            res_row[int(n)] = 1
    return res_row


def get_grid_loc(img, space=7, mode=None):

    binary_img = 255 * img.astype(np.uint8)
    grid_loc = [0]
    a, cnts , b = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if mode == 'row':
            loc = int(y + 0.5 * h)
        elif mode == 'col':
            loc = int(x + 0.5 * w)
        grid_loc.append(loc)
    img_space = np.zeros(binary_img.shape, dtype=np.uint8)
    if mode == 'row':
        for loc in grid_loc:
            img_space[max(0, loc - int(space/2)): min(img_space.shape[0], loc + int(space/2) + 1), :] = 1
    elif mode == 'col':
        for loc in grid_loc:
            img_space[:, max(0, loc - int(space/2)): min(img_space.shape[1], loc + int(space/2) + 1)] = 1
    last = binary_img.shape[0] if mode == 'row' else binary_img.shape[1]
    grid_loc.append(last)
    return sorted(grid_loc), img_space


def get_loc(row_region, col_region):
    grid_loc_row, img_space_row = get_grid_loc(row_region, mode='row')
    grid_loc_col, img_space_col = get_grid_loc(col_region, mode='col')
    return grid_loc_row, grid_loc_col, img_space_row, img_space_col


def get_res_and_region(row, col):
    """
    :param row:
    :param col:
    :return: res_row, res_col, row_region, col_region
    res_row and res_col are lists;
    row_region and col_region are images.
    """
    h, w = len(row), len(col)
    np_row = row.numpy()
    np_col = col.numpy()
    res_row = graph_cut(np_row)
    res_col = graph_cut(np_col)
    row_region = np.repeat(np.expand_dims(np.array(res_row), 1), w, axis=1).astype(np.uint8)
    col_region = np.repeat(np.expand_dims(np.array(res_col), 0), h, axis=0).astype(np.uint8)
    return res_row, res_col, row_region, col_region

