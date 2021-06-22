#-*-coding:utf-8-*-
import tensorflow as tf


class SFCN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=18, kernel_size=7, padding='same', activation='relu', dilation_rate=2)
    
    def call(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return c3


def proj_row(inputs):
    b, h, w, c = inputs.shape
    avg = tf.math.reduce_mean(inputs, axis=2)
    avg = tf.stack([avg for i in range(w)], axis=2)
    return avg


def proj_col(inputs):
    b, h, w, c = inputs.shape
    avg = tf.math.reduce_mean(inputs, axis=1)
    avg = tf.stack([avg for i in range(h)], axis=1)
    return avg

    
class Block(tf.keras.Model):
    def __init__(self, block_num, mode=None):
        super().__init__()
        self.block_num = block_num # start from index 1
        self.mode = mode
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', dilation_rate=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', dilation_rate=3)
        self.conv3 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation='relu', dilation_rate=4)
        if mode == 'rpn':
            self.mp = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')
        elif mode == 'cpn':
            self.mp = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')
        self.branch1 = tf.keras.layers.Conv2D(filters=18, kernel_size=1, padding='same', activation='relu')
        self.branch2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')#, activation='relu')
        return

    def call(self, inputs):
        c1 = self.conv1(inputs)
        c2 = self.conv2(inputs)
        c3 = self.conv3(inputs)
        c = tf.concat([c1, c2, c3], axis=3)
        if self.block_num <= 3:
            c = self.mp(c)

        branch1 = self.branch1(c)
        if self.mode == 'rpn':
            branch1 = proj_row(branch1)
        elif self.mode == 'cpn':
            branch1 = proj_col(branch1)
        
        branch2 = self.branch2(c)
        if self.mode == 'rpn':
            branch2 = proj_row(branch2)
        elif self.mode == 'cpn':
            branch2 = proj_col(branch2)
        branch2 = tf.keras.activations.sigmoid(branch2)

        output = tf.concat([branch1, c, branch2], axis=3)
        return output, branch2


class RPN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.Block1 = Block(1, mode='rpn')
        self.Block2 = Block(2, mode='rpn')
        self.Block3 = Block(3, mode='rpn')
        self.Block4 = Block(4, mode='rpn')
        self.Block5 = Block(5, mode='rpn')
        return

    def call(self, input):
        op1, r1 = self.Block1(input)
        op2, r2 = self.Block2(op1)
        op3, r3 = self.Block3(op2)
        op4, r4 = self.Block4(op3)
        op5, r5 = self.Block5(op4)
        return r3, r4, r5


class CPN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.Block1 = Block(1, mode='cpn')
        self.Block2 = Block(2, mode='cpn')
        self.Block3 = Block(3, mode='cpn')
        self.Block4 = Block(4, mode='cpn')
        self.Block5 = Block(5, mode='cpn')
        return

    def call(self, inputs):
        op1, c1 = self.Block1(inputs)
        op2, c2 = self.Block2(op1)
        op3, c3 = self.Block3(op2)
        op4, c4 = self.Block4(op3)
        op5, c5 = self.Block5(op4)
        return c3, c4, c5
    

class Split(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sfcn = SFCN()
        self.rpn = RPN()
        self.cpn = CPN()

    def call(self, inputs):
        outputs = self.sfcn(inputs)
        r3, r4, r5 = self.rpn(outputs)
        c3, c4, c5 = self.cpn(outputs)
        return r3[0, :, 0, 0], r4[0, :, 0, 0], r5[0, :, 0, 0], c3[0, 0, :, 0], c4[0, 0, :, 0], c5[0, 0, :, 0]

