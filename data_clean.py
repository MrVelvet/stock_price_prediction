import os
import csv
import numpy as np
from sklearn import preprocessing
class deal_with:


    def __init__(self, path, cond):
        self.path = path
        self.cond = cond
        self.x = []
        self.y = []
        self.x_buffalo = []
        self.y_buffalo = []
        self.x_test = []
        self.y_test = []

    def sentence_judge(self, sentence, compare):
        for index_compare in compare:
            if sentence[index_compare] == 'NA':
                return 0
        return 1

    def data_construct(self, window, cond, compare):
        index = 0
        for item in os.listdir(self.path):
            print(index, item)
            if index > 150:
                break
            error_index = 0
            if cond(item) and (item != '.DS_Store'):
                with open(self.path + item, 'r', encoding = 'utf-8') as file:
                    data = []
                    targ = csv.reader(file)
                    for i, row in enumerate(targ):
                        if i > 0 and len(row) > 1:
                            if self.sentence_judge(row, compare):
                                data.append(row)
                            else:
                                break
                    file.close()
                for i in range(0, len(data) - window):
                    temp_data = []
                    for j in range(window//2):
                        row = data[i + j]
                        for k, value in enumerate(row):
                            if k in compare:
                                temp_data.append(float(value))
                    temp_data = np.array(temp_data).reshape(-1, window//2)
                    temp_data = temp_data.T
                    temp_x = []
                    for r_row in temp_data:
                        try:
                            temp_x.append(preprocessing.scale(r_row))
                        except:
                            error_index = 1
                    temp_x = np.array(temp_x)
                    temp_data = temp_x.T
                    temp_data = temp_data.reshape(1, -1)[0]
                    ##############################
                    price_max = float('-Inf')
                    price_min = float('Inf')
                    for j in range(window//2, window):
                        row = data[i + j]
                        if float(row[6]) > price_max:
                            price_max = float(row[6])
                        if float(row[6]) < price_min:
                            price_min = float(row[6])
                    if price_max > float(data[i + window//2 - 1][6]) * (1 + 0.15) \
                        and price_min < float(data[i + window//2 - 1][6]) * (1 - 0.15):
                        continue
                    if price_max > float(data[i + window//2 - 1][6]) * (1 + 0.15) and error_index == 0:
                        self.x.append(temp_data)
                        self.y.append(np.array([1, 0]))
                    elif price_min < float(data[i + window//2 - 1][6]) * (1 - 0.15) and error_index == 0:
                        self.x.append(temp_data)
                        self.y.append(np.array([0, 1]))
                    #####################################
                    #if float(data[i + window//2 - 1][6]) < float(data[i + window//2][6]) * 0.95:
                    #    self.x.append(temp_data)
                    #    self.y.append(np.array([0, 1]))
                    #elif float(data[i + window//2 - 1][6]) > float(data[i + window//2][6]) * 1.05:
                    #    self.x.append(temp_data)
                    #    self.y.append(np.array([1, 0]))
            index += 1


        self.x = np.array(self.x)
        self.y = np.array(self.y)



    def data_shuffle(self, x, y):
        data = list(zip(x, y))
        np.random.shuffle(data)
        return list(zip(*data))


    def next_batch(self, i, batch_size):
        return self.x_buffalo[(i * batch_size):((i + 1) * batch_size)], \
               self.y_buffalo[(i * batch_size):((i + 1) * batch_size)]

