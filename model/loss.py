#-*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
import cv2
from model.graph_cut import get_res_and_region, get_loc


def loss_split(inputs, label_split, split, loss_fn=tf.keras.losses.binary_crossentropy):
    label_sr, label_sc = label_split[0]
    label_sr = tf.convert_to_tensor(np.frombuffer(label_sr.numpy(), dtype=np.uint8))
    label_sc = tf.convert_to_tensor(np.frombuffer(label_sc.numpy(), dtype=np.uint8))
    r3, r4, r5, c3, c4, c5 = split(inputs)
    accr, accc = cal_split_acc(label_sr, label_sc, r5, c5)
    loss_split_row = tf.reduce_sum([tf.multiply(0.1, loss_fn(label_sr, r3)),
                                   tf.multiply(0.25, loss_fn(label_sr, r4)),
                                   loss_fn(label_sr, r5)])
    loss_split_col = tf.reduce_sum([tf.multiply(0.1, loss_fn(label_sc, c3)),
                                   tf.multiply(0.25, loss_fn(label_sc, c4)),
                                   loss_fn(label_sc, c5)])
    loss_split_total = tf.add(loss_split_row, loss_split_col)
    return loss_split_total, accr, accc


def cal_split_acc(label_sr, label_sc, r5, c5):
    label_sr, label_sc = list(label_sr.numpy()), list(label_sc.numpy())
    gcr5, gcc5, _, _ = get_res_and_region(r5, c5)
    
    prer = len(set(list(np.where(np.array(label_sr) == 1)[0])).
               intersection(set(list(np.where(np.array(gcr5) == 1)[0])))) / (
                   1e-5 + len(np.where(np.array(gcr5) == 1)[0]))
    recallr = len(set(list(np.where(np.array(label_sr) == 1)[0])).
                  intersection(set(list(np.where(np.array(gcr5) == 1)[0])))) / (
                      1e-5 + len(np.where(np.array(label_sr) == 1)[0]))
    f1r = 2 * prer * recallr / (1e-5 + prer + recallr)
    
    prec = len(set(list(np.where(np.array(label_sc) == 1)[0])).
               intersection(set(list(np.where(np.array(gcc5) == 1)[0])))) / (
                   1e-5 + len(np.where(np.array(gcc5) == 1)[0]))
    recallc = len(set(list(np.where(np.array(label_sc) == 1)[0])).
                  intersection(set(list(np.where(np.array(gcc5) == 1)[0])))) / (
                      1e-5 + len(np.where(np.array(label_sc) == 1)[0]))
    f1c = 2 * prec * recallc / (1e-5 + prec + recallc)
    
    return f1r, f1c


def loss_merge(inputs, grid_loc, mask_char, merge, loss_fn=tf.keras.losses.binary_crossentropy):
    b, h, w, c = inputs.shape
    M, N = len(grid_loc[0]) - 1, len(grid_loc[1]) - 1
    if M <= 1 or N <= 1:
        return False, 0, 0, 0
    matrix_u2, matrix_u3, matrix_d2, matrix_d3, matrix_l2, matrix_l3, matrix_r2, matrix_r3 = merge(inputs, grid_loc)
    D2 = cal_D(matrix_u2, matrix_d2)
    D3 = cal_D(matrix_u3, matrix_d3)
    R2 = cal_R(matrix_l2, matrix_r2)
    R3 = cal_R(matrix_l3, matrix_r3)
    mask_char = np.reshape(np.frombuffer(mask_char[0].numpy(), dtype=np.uint8), (h, w))
    label_D, label_R = gen_merge_label(mask_char, grid_loc)

    accD, accR = cal_merge_acc(label_D, label_R, D3, R3)
    loss_merge_d = tf.reduce_sum([tf.multiply(0.25, loss_fn(label_D, D2)), loss_fn(label_D, D3)])   # reduce_sum
    loss_merge_r = tf.reduce_sum([tf.multiply(0.25, loss_fn(label_R, R2)), loss_fn(label_R, R3)])
    loss_merge_total = tf.add(loss_merge_d, loss_merge_r)
    return True, loss_merge_total, accD, accR


def gen_merge_label(char_mask, grid_loc):
    """
    :param char_mask: shape is (h, w)
    :param grid_loc: [row_loc, col_loc]
    :return:
    """
    row_loc, col_loc = grid_loc
    M, N = len(row_loc) - 1, len(col_loc) - 1
    label_D = np.zeros((M - 1, N))
    label_R = np.zeros((M, N - 1))
    for _idxr in range(1, M - 1):
        for _idxc in range(N):
            su = char_mask[row_loc[_idxr], col_loc[_idxc]: col_loc[_idxc + 1]]
            if max(su) != 0:
                label_D[_idxr - 1, _idxc] = 1
    for _idxc in range(1, N - 1):
        for _idxr in range(M):
            su = char_mask[row_loc[_idxr]: row_loc[_idxr + 1], col_loc[_idxc]]
            if max(su) != 0:
                label_R[_idxr, _idxc - 1] = 1

    return tf.convert_to_tensor(label_D), tf.convert_to_tensor(label_R)


def gen_merge_inputs(split_input, split):
    # split 的 input 是 1, h, w, 3
    b, h, w, c = split_input.shape
    _, _, pred_row, _, _, pred_col = split(split_input)

    pred_row_exp = tf.stack([pred_row for i in range(w)], axis=1)
    pred_col_exp = tf.stack([pred_col for i in range(h)], axis=0)
    pred_row_exp = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(pred_row_exp, dtype=tf.float32), axis=0), axis=-1)
    pred_col_exp = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(pred_col_exp, dtype=tf.float32), axis=0), axis=-1)
    
    res_row, res_col, row_region, col_region = get_res_and_region(pred_row, pred_col)
    grid_loc_row, grid_loc_col, img_space_row, img_space_col = get_loc(row_region, col_region)

    row_region = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(row_region, dtype=tf.float32), axis=0), axis=-1)
    col_region = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(col_region, dtype=tf.float32), axis=0), axis=-1)
    img_space = np.bitwise_or(img_space_row, img_space_col)
    img_space = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(img_space, dtype=tf.float32), axis=0), axis=-1)
    
    grid_loc = [grid_loc_row, grid_loc_col]
    merge_input = tf.concat([split_input, pred_row_exp, pred_col_exp, row_region, col_region, img_space], axis=-1)
    
    return merge_input, grid_loc


def cal_merge_acc(label_D, label_R, D, R):
    
    label_D, label_R = np.array(label_D), np.array(label_R)
    D, R = D.numpy(), R.numpy()
    M, N = len(R), len(D[0])
    D[D >= 0.5] = 1
    D[D < 0.5] = 0
    R[R >= 0.5] = 1
    R[R < 0.5] = 0
    # D
    need_merge, no_need_merge = 0, 0
    pred_merge, no_pred_merge = 0, 0
    pred_merge_right, pred_no_merge_right = 0, 0
    for m in range(M - 1):
        for n in range(N):
            if label_D[m, n] == 1:
                need_merge += 1
            else:
                no_need_merge += 1
            if D[m, n] == 1:
                pred_merge += 1
            else:
                no_pred_merge += 1
            if label_D[m, n] == D[m, n] == 1:
                pred_merge_right += 1
            if label_D[m, n] == D[m, n] == 0:
                pred_no_merge_right += 1
                
    predD = (1e-5 + pred_merge_right) / (1e-5 + pred_merge)
    recallD = (1e-5 + pred_merge_right) / (1e-5 + need_merge)

    # R
    need_merge, no_need_merge = 0, 0
    pred_merge, no_pred_merge = 0, 0
    pred_merge_right, pred_no_merge_right = 0, 0
    for m in range(M):
        for n in range(N - 1):
            if label_R[m, n] == 1:
                need_merge += 1
            else:
                no_need_merge += 1
            if R[m, n] == 1:
                pred_merge += 1
            else:
                no_pred_merge += 1
            if label_R[m, n] == R[m, n] == 1:
                pred_merge_right += 1
            if label_R[m, n] == R[m, n] == 0:
                pred_no_merge_right += 1

    predR = (1e-5 + pred_merge_right) / (1e-5 + pred_merge)
    recallR = (1e-5 + pred_merge_right) / (1e-5 + need_merge)
    
    f1D = 2 * predD * recallD / (predD + recallD)
    f1R = 2 * predR * recallR / (predR + recallR)

    return f1D, f1R


def cal_D(matrix_u, matrix_d):
    # input: M * N; output: M-1 * N
    D = tf.add(tf.multiply(0.5, tf.multiply(matrix_u[0, 1:, :, 0], matrix_d[0, :-1, :, 0])),
               tf.multiply(0.25, tf.add(matrix_u[0, 1:, :, 0], matrix_d[0, :-1, :, 0])))
    return D


def cal_R(matrix_l, matrix_r):
    R = tf.add(tf.multiply(0.5, tf.multiply(matrix_l[0, :, 1:, 0], matrix_r[0, :, :-1, 0])),
               tf.multiply(0.25, tf.add(matrix_l[0, :, 1:, 0], matrix_r[0, :, :-1, 0])))
    return R