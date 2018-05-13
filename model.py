import tensorflow as tf 
import numpy as np 
from tensorflow.contrib import rnn
from tensorflow.python.training.moving_averages import assign_moving_average


class model_construction:


    def __init__(self, layer_info, acti_function, x_length, y_length, name, batch_size):
        self.layer_info = layer_info
        self.acti_function = acti_function
        self.x_length = x_length
        self.y_length = y_length
        self.input_size = -1
        self.timestep_size = -1
        self.hidden_size = -1
        self.layer_num = -1
        self.name_scope = name
        self.batch_size = batch_size

    def batch_norm(self, x, train, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = np.shape(x)[1]
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, [0], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)
            if train == tf.constant(1):
                train_cond = tf.greater(1, 0)
            else:
                train_cond = tf.greater(0, 1)
            mean, variance = tf.cond(train_cond, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

    def placeholder(self):
        x = tf.placeholder(tf.float32, [None, self.x_length])
        y = tf.placeholder(tf.float32, [None, self.y_length])
        keep_prob = tf.placeholder(tf.float32, [])
        if_train = tf.placeholder(tf.int32, [])
        batch_size = tf.placeholder(tf.int32, [])
        return x, y, keep_prob, if_train, batch_size

    def fc_layer(self, inputs, in_size, out_size, keep_prob, name, if_train, bn_name, activation_function=None):
        with tf.name_scope(name):
            weights = tf.Variable(tf.random_uniform([in_size, out_size]), name='W')
            biases = tf.Variable(tf.constant(0.1, shape=[out_size]), name='b')
        wx_plus_b = tf.matmul(inputs, weights) + biases
        #wx_plus_b = self.batch_norm(x = wx_plus_b, train = self.if_train, name = bn_name)
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        outputs = tf.nn.dropout(outputs, keep_prob)
        return outputs

    def lstm_layer(self, input_size, timestep_size, hidden_size, layer_num, keep_prob):
        x_termlstm = tf.reshape(self.x_term, [-1, timestep_size, input_size])
        lstm_cell = rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
        init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x_termlstm, \
                                            initial_state=init_state, time_major=False)
        lstm_res = outputs[:, -1, :]
        lstm_res = tf.nn.dropout(lstm_res, keep_prob)
        return lstm_res

    def lstm_param_setting(self, input_size, timestep_size, hidden_size, layer_num):
        self.input_size = input_size
        self.timestep_size = timestep_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

    def network_connection(self):
        self.x_term, self.y_term, self.keep_prob, self.if_train, self.batch_size = self.placeholder()
        network = []
        self.lstm_res = self.lstm_layer(self.input_size, self.timestep_size, \
                           self.hidden_size, self.layer_num, self.keep_prob)
        for i in range(len(self.layer_info) + 1):
            if i is 0:
                network.append(self.fc_layer(self.lstm_res, self.hidden_size, self.layer_info[i], \
                                              self.keep_prob, name=self.name_scope + str(i), \
                                              if_train = self.if_train, bn_name = 'train_' + \
                                              str(i), activation_function = self.acti_function))
            elif i is len(self.layer_info):
                network.append(self.fc_layer(network[i - 1], self.layer_info[i - 1], self.y_length, \
                                              self.keep_prob, name=self.name_scope + str(i), \
                                              if_train = self.if_train, bn_name = 'train_' + str(i), \
                                              activation_function = None))
            else:
                network.append(self.fc_layer(network[i - 1], self.layer_info[i - 1], self.layer_info[i], \
                                              self.keep_prob, name=self.name_scope + str(i), \
                                              if_train = self.if_train, bn_name = 'train_' + str(i), \
                                              activation_function = self.acti_function))
        return network

    def estimation(self):
        self.network = self.network_connection()
        self.predictions = tf.argmax(self.lstm_res, 1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                                    logits=self.lstm_res, labels=self.y_term))
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_term, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'))
        self.actuals = tf.argmax(self.y_term, 1)

        self.ones_like_actuals = tf.ones_like(self.actuals)
        self.zeros_like_actuals = tf.zeros_like(self.actuals)
        self.ones_like_predictions = tf.ones_like(self.predictions)
        self.zeros_like_predictions = tf.zeros_like(self.predictions)

        self.tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, self.ones_like_actuals), \
                                                tf.equal(self.predictions, self.ones_like_predictions)), \
                                            "float"))

        self.tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, self.zeros_like_actuals), \
                                                tf.equal(self.predictions, self.zeros_like_predictions)), \
                                            "float"))

        self.fn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, self.zeros_like_actuals), \
                                                tf.equal(self.predictions, self.ones_like_predictions)), \
                                            "float"))

        self.fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, self.ones_like_actuals), \
                                                tf.equal(self.predictions, self.zeros_like_predictions)), \
                                            "float"))
