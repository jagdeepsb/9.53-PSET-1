import numpy as np

class DataSet():
    def __init__(self, classes1, classes2):
        # read and parse data
        with open('Iris.txt') as f:
            lines = f.readlines()
        self.data = []
        self.num_classes = 0
        self.classes = {}
        for line in lines:
            tokens = line.split(',')
            x = []
            y = None
            y_real = 0
            for i in range(4):
                x.append(float(tokens[i]))
            x = np.array(x)
            curr_class = tokens[4]
            if '\n' in curr_class:
                curr_class = curr_class[:-1]
            if not curr_class in classes1 and not curr_class in classes2:
                continue
            if curr_class in self.classes:
                y = self.classes[curr_class]
            else:
                self.classes[curr_class] = self.num_classes
                y = self.num_classes
                self.num_classes += 1
            y = -1 if curr_class in classes1 else 1
            self.data.append((x, y, self.classes[curr_class]))
        print("CLASSES: ", self.classes)

    # return dataset in randomized order
    def get_randomized(self,):
        data_r = []
        samples = np.random.choice(len(self.data), (len(self.data),), replace=False)
        for sample in samples:
            data_r.append(self.data[sample])
        return data_r