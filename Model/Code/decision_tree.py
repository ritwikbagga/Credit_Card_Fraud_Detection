import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode

from sklearn.model_selection import KFold


# comment1

def f1_score(y_true, y_pred):
    """
    Function for calculating the F1 score

    Params
    ------
    y_true  : the true labels shaped (N, C),
              N is the number of datapoints
              C is the number of classes
    y_pred  : the predicted labels, same shape
              as y_true

    Return
    ------
    score   : the F1 score, shaped (N,)

    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in zip(y_true, y_pred):
        if i[0] == 1:
            if i[1] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if i[1] == 0:
                tn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision = " + str(precision))
    print("recall = " + str(recall))
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1


def gini_index(groups, classes):
    """
    Function for calculating the gini index
        -- The goodness of the split

    Params
    ------
    groups  : A list containing two groups of samples
                resulted by the split
    classes : The classes in the classification problem
                e.g. [0,1]

    Return
    ------
    gini    : the gini index of the split
    """
    split = []
    gini = 1
    for g in groups:
        ones = sum(g)
        split.append(ones / len(g))
    for s in split:
        gini -= s ** 2
    return gini


def get_split(x_train, y_train):
    """
    Function to generate the split which results in
        the best gini_index

    Params
    ------
    x_train : the input data (n_samples, n_features)
    y_train : the input label (n_samples,)

    Return
    ------
    {gini_index, split_index, split_value}
    """
    """
    returns a Node?

    take dim1
    sort it and keep y_train attached to it
    get every index where y_train == 1, store in i
    create an array for each index j in i having all 1s beginning from j (0s for all indexes less than j)
    sum all these arrays to obtain Nl1 vector
    Nl = [1, 2, 3, 4...., S] where S is the number of samples
    we know Nl + Nr = sum(1s in Y)
    create Nr1 using sum(1s in Y) - Nl1
    create Nl0 using Nl - Nl1
    create Nr0 using Nr - Nr1


    G(split) = (Nl0 * Nl1)/ Nl + (Nr0 * Nr1)/Nr
    now find min(array) to get best split index


    """
    ginis = []
    for dim in x_train.T:
        dimsort = dim[np.argsort(dim)]
        ysort = y_train.reshape(-1)[np.argsort(dim)]
        dimy = np.array([dim, y_train.reshape(-1, )]).T

        samples = len(y_train)
        dimy = dimy[np.argsort(dimy[:, 1])]
        ones_indexes = np.argwhere(ysort == 1)[:, 0]
        ones_array = []
        for i in ones_indexes:
            # ones_array.append([0 if j < i else 1 for j in range(samples)])
            temp = [0] * i
            temp.extend([1] * (samples - i - 1))
            ones_array.append(temp)
        ones_array = np.array(ones_array)
        Nl1 = ones_array.sum(axis=0)
        Nl = np.arange(1, samples)
        Nr = np.arange(samples - 1, 0, -1)
        Nr1 = sum(y_train) - Nl1
        Nl0 = Nl - Nl1
        Nr0 = Nr - Nr1
        g_split = (2 / samples) * (((Nl0 * Nl1) / Nl) + ((Nr0 * Nr1) / Nr))
        g_min_val = min(g_split)
        g_min_index = np.where(g_split == min(g_split))[0]
        val_at_min = dimsort[g_min_index][0]

        ginis.append([g_min_val, int(g_min_index[0]), val_at_min])
    ginis = np.array(ginis)
    best_feature_index = np.argmin(ginis[:, 0])  # column number of the best index
    # traning_xy = np.array(x_train,y_train.reshape(-1,))
    sorted_x_train = x_train[np.argsort(x_train[:, best_feature_index])]
    sorted_y_train = y_train[np.argsort(x_train[:, best_feature_index])]
    sample_index = ginis[best_feature_index][1]
    left_data_x = sorted_x_train[0:int(sample_index)]
    left_data_y = sorted_y_train[:int(sample_index)]

    right_data_x = sorted_x_train[int(sample_index):]
    right_data_y = sorted_y_train[int(sample_index):]

    left_node = (left_data_x, left_data_y)
    right_node = (right_data_x, right_data_y)
    split_index = best_feature_index
    split_value = ginis[best_feature_index][2]
    data = (left_node, right_node, split_index, split_value)
    return data


##########################################################
# Alternatively combine gini_index into get_split and
# find the split point using an array instead of a for
# loop, would speed up run time
##########################################################


class DecisionTree(object):
    """
    The Decision Tree classifier
    """

    def __init__(self, max_depth, min_size=5):
        """
        Params
        ------
        max_depth   : the maximum depth of the decision tree
        min_size    : the minimum observation points on a
                        leaf/terminal node
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        self.left = None
        self.right = None
        # self.num_datapoints = None

    def split(self, data, depth):
        """
        Function called recursively to split
            the data in order to build a decision
            tree.

        Params
        ------
        data    : {left_node, right_node, split_index, split_value}
        depth   : the current depth of the node in the decision tree

        Return
        ------
        """
        # base case
        if depth > self.max_depth:
            a = data[0][1].reshape(-1)
            np.append(a, data[1][1].reshape(-1))
            self.root = (None, None, mode(a), None)
            return

        # change state
        # check number of data points in y_train of split

        self.root = get_split(data[0], data[1])
        self.left = DecisionTree(self.max_depth, self.min_size)
        self.right = DecisionTree(self.max_depth, self.min_size)
        self.left.fit(self.root[0][0], self.root[0][1], depth + 1)
        self.right.fit(self.root[1][0], self.root[1][1], depth + 1)

    def fit(self, x_train, y_train, depth=0):
        """
        Fitting the KNN classifier

        Hint: Build the decision tree using
                splits recursively until a leaf
                node is reached

        """
        self.depth = depth
        if (len(y_train) > self.min_size) and (depth < self.max_depth):
            # self.root = get_split(x_train, y_train)
            # self.num_datapoints = len(self.root[0][1]) + len(self.root[1][1])
            self.split((x_train, y_train), depth + 1)
        else:
            try:
                m = mode(y_train.reshape(-1))
            except:
                m = 1

            self.root = (None, None, m, None)
            return

    def predict(self, x_test):
        """
        Predicting the test data

        Hint: Run the test data through the decision tree built
                during training (self.tree)
        """
        prediction = None

        if self.root[-1] is None:
            prediction = self.root[2]

        else:
            if x_test[int(self.root[2])] < self.root[-1]:
                prediction = self.left.predict(x_test)

            else:
                prediction = self.right.predict(x_test)
        if prediction is None:
            print("prediction is None :/")
            raise ValueError
        return prediction


def main():
    # Example running the class DecisionTree

    print("running...")

    # Example running the class KNN
    x_train_path = "../../Data/x_train.csv"
    y_train_path = "../../Data/y_train.csv"

    x_df = np.genfromtxt(x_train_path, delimiter=',')
    y_df = np.genfromtxt(y_train_path, delimiter=',')

    # print(x_df)
    # # input(x_df.values.tolist()[:20])
    # x_train, y_train, x_test, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=2)
    #
    # knn = KNN(n_neighbors=5)
    # knn.fit(x_train.to_numpy().reshape(-1, 1), y_train)
    # y_pred = knn.predict(x_test.to_numpy().reshape(-1, 1))
    # score = f1_score(y_test, y_pred)
    # print(score)
    # input()
    ########################################
    # Simple Guide on KFold
    ########################################
    kf = KFold(n_splits=5)
    kf.get_n_splits(x_df)

    for i, (train_index, test_index) in enumerate(kf.split(x_df)):
        x_train, x_test = x_df[train_index], x_df[test_index]
        y_train, y_test = y_df[train_index], y_df[test_index]
        dt = DecisionTree(max_depth=5)
        print("calling fit...")
        dt.fit(x_train, y_train.reshape(-1, 1))
        y_pred = [dt.predict(x_test_i) for x_test_i in x_test]
        score = f1_score(y_test, y_pred)

        print(str(i) + " f1 score = " + str(score))


if __name__ == "__main__":
    main()




