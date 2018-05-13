import data_clean
import parameter_setting as pa
import json
import tensorflow as tf
import model 
import numpy as np
import tqdm
#import numerical_identify as ni

param = pa.param()
dataset = data_clean.deal_with('/users/singlestar/desktop/data/', param.cond)
if param.data_arange:
    #compare_list = ni.identify('/users/singlestar/desktop/data/', param.cond)
    #compare = compare_list.run()
    compare = [287, 288, 289, 290, 291, 292, 293, 298, 331, 332, 338, 339, 340, 341, 342, 343, 344, 345, 346] 
    #compare = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 21, 22, 23, 24, 287, 288, 289, 290, 291, 292, 293, 298, 331, 332, 338, 339, 340, 341, 342, 343, 344, 345, 346] 
    #compare = [6]
    dataset.data_construct(window = param.window, cond = param.cond, compare = compare)
    print('=============================================')
    print('data setting')
    with open(param.data_location[0], 'w') as file:
        json.dump(dataset.x.tolist(), file)
        file.close()
    with open(param.data_location[1], 'w') as file:
        json.dump(dataset.y.tolist(), file)
        file.close()
    print('=============================================')
    print('data memorizing')
else:
    with open(param.data_location[0], 'r') as file:
        dataset.x = json.load(file)
        file.close()
    with open(param.data_location[1], 'r') as file:
        dataset.y = json.load(file)
        file.close()
    print('=============================================')
    print('data loading')

print(len(dataset.y))
print(sum(np.argmax(dataset.y, axis = 1))/len(dataset.x))
param.x_length = len(dataset.x[0])
param.input_size = param.x_length//(param.window//2)
length_of_data = len(dataset.y)
x_temp, y_temp = dataset.data_shuffle(x = dataset.x, y = dataset.y)
dataset.x_test = x_temp[int(length_of_data*param.proportion):]
dataset.y_test = y_temp[int(length_of_data*param.proportion):]
dataset.x = x_temp[:int(length_of_data*param.proportion)]
dataset.y = y_temp[:int(length_of_data*param.proportion)]
print('=============================================')
print('data dividing')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
with tf.Session(config =config) as sess:
    model_on = model.model_construction(layer_info = param.layer_info, acti_function = param.acti_function, \
                                        x_length = param.x_length, y_length = 2, name = 'global', \
                                        batch_size = param.batch_size)
    model_on.lstm_param_setting(input_size = param.input_size, timestep_size = param.window//2, hidden_size = param.hiddensize, \
                                layer_num = param.layer_num)
    model_on.estimation()
    global_step = tf.Variable(0, trainable = False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(model_on.loss)
    train_step = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
    sess.run(tf.global_variables_initializer())

    for i in range(0, param.num_epoch):
        train_tp = 0
        train_tn = 0
        train_fp = 0
        train_fn = 0
        acc_acc = 0
        j = 0
        dataset.x_buffalo, dataset.y_buffalo = dataset.data_shuffle(dataset.x, dataset.y)
        while (j + 1) * param.batch_size < len(dataset.x):
            x_batch, y_batch = dataset.next_batch(j, param.batch_size)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            feed_dict = {model_on.x_term: x_batch,
                         model_on.y_term: y_batch,
                         model_on.keep_prob: param.keep_prob,
                         model_on.if_train:1,
                         model_on.batch_size:param.batch_size}
            sess.run(train_step, feed_dict = feed_dict)
            feed_dict_ob = {model_on.x_term: x_batch,
                         model_on.y_term: y_batch,
                         model_on.keep_prob: 1.0,
                         model_on.if_train:1,
                         model_on.batch_size:param.batch_size}
            step, losses, acc, t1, t3, t4 = sess.run([global_step, model_on.loss, model_on.accuracy, \
                            model_on.lstm_res, model_on.network[3], model_on.predictions, \
                                                            ],
                                                        feed_dict_ob)
            j += 1
            acc_acc += acc
            #if j %10000 == 1:
            #    print (j)

            #if i > 200:
            #    print('#########')
            #    print(x_batch)
            #    print('############')
            #    print(t3, t4)
            #    print('############')
            #    print(y_batch)
            #    input()


        if i%10 == 0:
            test_tp = 0
            test_tn = 0
            test_fp = 0
            test_fn = 0
            acc_acs = 0
            j_test = 0
            print('{} in train, the loss is {} while the acc is {}'.format(i, \
                    losses, acc_acc/j))
            dataset.x_buffalo, dataset.y_buffalo = dataset.data_shuffle(dataset.x_test, dataset.y_test)
            while (j_test + 1) * param.batch_size < len(dataset.x_test):
                x_batch, y_batch = dataset.next_batch(j_test, param.batch_size)
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                feed_dict_ob = {model_on.x_term: x_batch,
                             model_on.y_term: y_batch,
                             model_on.keep_prob: 1.0,
                             model_on.if_train:0,
                             model_on.batch_size:param.batch_size}
                step, losses, tp, tn, fp, fn = sess.run([global_step, model_on.loss, model_on.tp_op, model_on.tn_op, model_on.fp_op, model_on.fn_op],
                                                           feed_dict_ob)
                j_test += 1
                test_tp += float(tp)
                test_tn += float(tn)
                test_fp += float(fp)
                test_fn += float(fn)
            print('----------------------------------')
            print(test_tp, test_tn, test_fp, test_fn)
            print((test_tp + test_tn)/(test_tp + test_tn + test_fp + test_fn))
            print('----------------------------------')
    print('=============================================')
    print('##############test mode#############')
    with open(param.data_location_test[0], 'r') as file:
        dataset.x_test = json.load(file)
        file.close()
    with open(param.data_location_test[1], 'r') as file:
        dataset.y_test = json.load(file)
        file.close()
    print('---------------------------------')
    print(len(dataset.y))
    print(sum(np.argmax(dataset.y_test, axis = 1))/len(dataset.x_test))
    print('---------------------------------')
    j_test = 0
    res_tp = 0

    dataset.x_buffalo, dataset.y_buffalo = dataset.data_shuffle(dataset.x_test, dataset.y_test)
    while (j_test + 1) * param.batch_size < len(dataset.x_test):
        x_batch, y_batch = dataset.next_batch(j_test, param.batch_size)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        feed_dict_ob = {model_on.x_term: x_batch,
                     model_on.y_term: y_batch,
                     model_on.keep_prob: 1.0,
                     model_on.if_train:0,
                     model_on.batch_size:param.batch_size}
        step, losses, tp, tn, fp, fn = sess.run([global_step, model_on.loss, model_on.tp_op, model_on.tn_op, model_on.fp_op, model_on.fn_op],
                                                   feed_dict_ob)
        j_test += 1
        test_tp += float(tp)
        test_tn += float(tn)
        test_fp += float(fp)
        test_fn += float(fn)
    print('#############test result###############')
    print('----------------------------------')
    print(test_tp, test_tn, test_fp, test_fn)
    print((test_tp + test_tn)/(test_tp + test_tn + test_fp + test_fn))
    print('----------------------------------')

    