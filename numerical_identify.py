import os
import csv

class identify:
    def __init__(self, path, cond):
        self.path = path
        self.cond = cond
        self.ret = []

    def numerical_search(self):
        ret = list(range(349))
        for ik, item in enumerate(os.listdir(self.path)):
            print (ik, item)
            if self.cond(item):
                if item == '.DS_Store':
                    continue
                with open(self.path + item, 'r', encoding = 'utf-8') as file:
                    targ = csv.reader(file)
                    for i, row in enumerate(targ):
                        if row[25] == 'NA':
                            break
                        if i > 0 and len(row) > 1:
                            for j, element in enumerate(row):
                                try:
                                    float(element)
                                except:
                                    if j in ret:
                                        ret.remove(j)
                                if element == 'Inf' or element ==  '-Inf':
                                    if j in ret:
                                        ret.remove(j)
                    file.close()

        return ret

    def run(self):
        self.ret = self.numerical_search()
        return self.ret



