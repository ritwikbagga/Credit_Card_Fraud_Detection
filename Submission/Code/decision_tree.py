import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


def score(y_true, y_pred):  #return [precision , recall , F1]
    # variables to store count of TP , FP , TN , FN
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in zip(y_true, y_pred):
        if i[0]==1:
            if i[1]==1:
                tp+=1
            else:
                fn+=1
        else: #y_true=0
            if i[1]==0:
                tn+=1
            else:
                fp+=1
    if tp==0:
        precision=1
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    #print("precission is ="+str(precision))
    #print("Recall is = " +str(recall))
    f1 = (2*precision*recall)/(precision+recall)
    #print("F1 score for current fold is="+ str(f1))
    return [precision, recall, f1]






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
    {gini_index, split_index, split_value} #this is a node
    """
    #take each feature , sort and then check indices of y_train
    #every indices of 1 = i (L)
    #calculate gini as gini = (Nl0 * Nl1)/ Nl + (Nr0 * Nr1)/Nr
    #find the min and return

    gini = [] # gini vector to store gini for whole for each feature


    result = {"gini_index":None,
            "split_index":None ,
            "split_value": None } #to return
    features = x_train.T #loop for eqch feature
    best_gini = 10000
    for feature in features:
        y_sorted = y_train.reshape(-1)[np.argsort[feature]] #got the indices of sorted by x1,,
        dy = np.array([feature, y_train.reshape(-1,)]).T
        total_samples = len(y_train)
        find_one = np.argwhere(y_sorted==1)[:,0]
        ones = []

        for index in find_one: #as suggested by TA
            temp = index*[0]
            temp.extend([1]*(total_samples-index-1))
            ones.append(temp)

        L1 = ones.sum(axis=0)
        L  = np.arrange(1, total_samples)
        L0 = L-L1
        R1 = sum(y_train)-L1 #only 2 splits
        R = np.arange(total_samples, -1, 0, -1)
        R0 = R-R1
        cur_gini = (L0*L1)/L + (R0*R1)/R


        # I am really lost here as to how to proceed in code but theoritically I know what is happening


    """
    get the min of gini index(x) and corresponding split index(y), split_value(z) and 
     then get the node {"gini_index":x, "split_index":y , "split_value":z}
    """

    #return result
    pass





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
        self.tree = None

    def terminal(self, node):
        """
        finds the mode of class in node and returns the label
        """

        pass





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
        left = data["left_node"]
        right = data["right_node"]
        if not left or not right: #base case
            data["left_node"] = data["right_node"] = self.terminal(left+right)

        if depth>self.max_depth: #base case
            data["left_node"] , data["right_node"] = self.terminal(left),  self.terminal(right)

        if len(left)<=self.min_size: #base case
            self.terminal(left)

        if len(right)<=self.min_size: #base case
            self.terminal(right)
        else:















    def fit(self, x_train, y_train):
        """
        Hint: Build the decision tree using
                splits recursively until a leaf
                node is reached

        """
        self.tree = get_split(x_train,y_train) #tree={ gini_index, split index, split,vale}
        self.split(self.tree, 1 )
        return self.tree



    def predict(self, x_test):
        """
        Predicting the test data

        Hint: Run the test data through the decision tree built
                during training (self.tree)
        """
        for row in x_test:
            if


def main():
    X_t = np.genfromtxt('../../Data/x_train.csv', delimiter=',')  # shape = (200000, 29)
    y_t = np.genfromtxt('../../Data/y_train.csv', delimiter=',')  # shape = (200000,)
    def k_fold_validation(max_depth=5, train_x=None, train_y=None):
        print("#######################################################")
        print("Decision Tree FOR Max_Depth="+str(max_depth))
        print("#######################################################")
        X = train_x
        y = train_y
        dt = DecisionTree(max_depth=5)
        kf = KFold(n_splits=5)
        kf.get_n_splits(train_x)
        F1_score = 0

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            startt = time.time()
            print("FOLD=" + str(i + 1))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            F1_score += score(y_test, y_pred)[2]
            ent = time.time()
            print("Time for this fold is: "+ str(ent-startt) +" Seconds")

        print("Final F1 for all the folds is " + str(F1_score / 5))
    Max_Depth_List = [3, 6, 9, 12, 15]
    for max_depth in Max_Depth_List:
        start = time.time()
        k_fold_validation(max_depth, X_t, y_t)
        end = time.time()
        print("Total Time for all 5 folds with d= "+str(max_depth)+" is "+str(end-start)+" seconds")





