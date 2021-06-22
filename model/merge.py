#-*-coding:utf-8-*-
import tensorflow as tf


class SFCN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.avg_pool1 = tf.keras.layers.AvgPool2D(pool_size=(2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.avg_pool2 = tf.keras.layers.AvgPool2D(pool_size=(2, 2), padding='same')
        
    def call(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(c1)
        c2_pool = self.avg_pool1(c2)
        c3 = self.conv3(c2_pool)
        c4 = self.conv4(c3)
        c4_pool = self.avg_pool2(c4)
        return c4_pool


def grid_pool(inputs, structure):
    b, h, w, c = inputs.shape
    row_loc, col_loc = structure
    M, N = len(row_loc) - 1, len(col_loc) - 1   # M 行 N 列
    for _b in range(b):
        whole = []
        for _idxr in range(M):
            hori = []
            for _idxc in range(N):
                y0, y1 = row_loc[_idxr], row_loc[_idxr + 1]
                x0, x1 = col_loc[_idxc], col_loc[_idxc + 1]
                temp_mean = tf.reduce_mean(inputs[:, y0: y1, x0: x1, :], axis=1, keepdims=True)
                temp_mean = tf.reduce_mean(temp_mean, axis=2, keepdims=True)
                temp_mean = tf.image.resize(temp_mean, (y1 - y0, x1 - x0))
                hori.append(temp_mean)
            hori = tf.concat(hori, axis=2)
            whole.append(hori)
        whole = tf.concat(whole, axis=1)
    return whole


def grid_pool_downsize(inputs, structure):
    b, h, w, c = inputs.shape
    row_loc, col_loc = structure
    M, N = len(row_loc) - 1, len(col_loc) - 1  # M 行 N 列
    for _b in range(b):
        whole = []
        for _idxr in range(M):
            hori = []
            for _idxc in range(N):
                y0, y1 = row_loc[_idxr], row_loc[_idxr + 1]
                x0, x1 = col_loc[_idxc], col_loc[_idxc + 1]
                temp_mean = tf.reduce_mean(inputs[:, y0: y1, x0: x1, :], axis=1, keepdims=True)
                temp_mean = tf.reduce_mean(temp_mean, axis=2, keepdims=True)
                hori.append(temp_mean)
            hori = tf.concat(hori, axis=2)
            whole.append(hori)
        whole = tf.concat(whole, axis=1)
    return whole


class Block(tf.keras.Model):
    def __init__(self, block_num, mode=None):
        super().__init__()
        self.block_num = block_num  # start from index 1
        self.mode = mode
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', dilation_rate=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', dilation_rate=3)

        self.branch1 = tf.keras.layers.Conv2D(filters=18, kernel_size=1, padding='same', activation='relu')
        self.branch2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')
        return
    
    def call(self, input, grid_loc):
        c1 = self.conv1(input)
        c2 = self.conv2(input)
        c3 = self.conv3(input)
        c = tf.concat([c1, c2, c3], axis=3)

        branch1 = self.branch1(c)
        branch1 = grid_pool(branch1, grid_loc)

        branch2 = self.branch2(c)
        branch2 = grid_pool(branch2, grid_loc)
        branch2 = tf.keras.activations.sigmoid(branch2)
        
        output = tf.concat([branch1, c, branch2], axis=3)
        return output, branch2
    

class Branch(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.Block1 = Block(1)
        self.Block2 = Block(2)
        self.Block3 = Block(3)
        return
    
    def call(self, input, grid_loc):
        op1, br1 = self.Block1(input, grid_loc)
        op2, br2 = self.Block2(op1, grid_loc)
        op3, br3 = self.Block3(op2, grid_loc)
        br2 = grid_pool_downsize(br2, grid_loc)
        br3 = grid_pool_downsize(br3, grid_loc)
        return br2, br3
    

class Merge(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.branch_u = Branch()
        self.branch_d = Branch()
        self.branch_l = Branch()
        self.branch_r = Branch()
        return

    def call(self, input, grid_loc):
        # 每一个 都是 M × N 矩阵
        matrix_u2, matrix_u3 = self.branch_u(input, grid_loc)
        matrix_d2, matrix_d3 = self.branch_d(input, grid_loc)
        matrix_l2, matrix_l3 = self.branch_l(input, grid_loc)
        matrix_r2, matrix_r3 = self.branch_r(input, grid_loc)
        return matrix_u2, matrix_u3, matrix_d2, matrix_d3, matrix_l2, matrix_l3, matrix_r2, matrix_r3
        

