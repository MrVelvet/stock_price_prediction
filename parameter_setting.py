import tensorflow as tf 

class param:
    def __init__(self):
        self.targ_set = []
        self.cond = lambda x:1 if x not in self.targ_set else 0
        self.window = 10
        self.data_arange = 1
        #self.data_location = ['/data/x.json', '/data/y.json']
        self.data_location = ['/users/singlestar/desktop/data_st/x_0.2.json', '/users/singlestar/desktop/data_st/y_0.2.json']
        self.data_location_test = ['/users/singlestar/desktop/data_st/x_test.json', '/users/singlestar/desktop/data_st/y_test.json']
        self.proportion = 0.85
        self.num_epoch = 100
        self.batch_size = 128
        self.input_size = -1
        self.hiddensize = 2
        self.layer_num = 1
        self.layer_info = [128, 64, 16, 4]
        self.acti_function = lambda x:tf.maximum(0.01 * x, x )
        #self.acti_function = tf.nn.relu
        self.x_length = -1
        self.keep_prob = 0.5
        