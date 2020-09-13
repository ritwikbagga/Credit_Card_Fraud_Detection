import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold
from scipy import stats




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

class KNN(object):
    """
    The KNN classifier
    """
    def __init__(self, n_neighbors):
        self.K = n_neighbors
        self.tree = None
        self.y_train = None

    def getKNeighbors(self, x_instance):
        """
        Locating the K nearest neighbors of
        the instance and return
        """

        dist, ind = self.tree.query([x_instance], k= self.K) #dist is distances to k closest neighbors and ind is indices
        return ind[0]  #return indices of the K nearest neighbours


    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier

        Hint:   Build a tree to get neighbors
                faster at test time
        """
        self.tree = BallTree(x_train)
        self.y_train = y_train


    def predict(self, x_test):
        """
        Predicting the test data
        Hint:   Get the K-Neighbors, then generate
                predictions using the labels of the
                neighbors
        """

        y_pred = []
        for x in x_test:
            neighbours = self.getKNeighbors(x)
            labels = self.y_train[neighbours]
            label = stats.mode(labels)
            y_pred.append(label[0][0])

        return y_pred





def main(X, y):

    X_t = X  #X_train
    y_t = y  #y_train
    print("##################################################### KNN ####################################")
    def k_fold_validation(neighbors=5, train_x=None, train_y=None):

        X = train_x
        y = train_y
        knn = KNN(n_neighbors=neighbors)
        kf = KFold(n_splits=5)
        kf.get_n_splits(train_x)
        F1 = 0
        precision = 0
        recall = 0

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("Running for FOLD=" + str(i + 1))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores = score(y_test, y_pred)
            F1+= scores[2]
            precision+=scores[0]
            recall+=scores[1]

        print("Final Precision for all the folds is " + str(precision / 5))
        print("Final Recall for all the folds is " + str(recall / 5))
        print("Final F1 for all the folds is " + str(F1 / 5))
    k_list = [3,5,10,20,25]

    print("############### KNN KFOLD VALIDATION #############")
    print("##################################################")

    for k in k_list:
        print("#######################################################")
        print("Knn for k="+str(k))
        print("#######################################################")
        k_fold_validation(k, X_t, y_t)





    # plt.figure("KNN TIME FOR EACH MODEL")
    # plt.plot(k_list,training_time)
    # plt.xlabel("Value of K")
    # plt.ylabel("Execution time in ms")
    # plt.show()

    # knn2 = KNN(n_neighbors=3)
    # knn2.fit(X_t, y_t)
    # y_pred_Full = knn2.predict(X_t)
    # score_FullSet = score(y_t, y_pred_Full) #[precissiion , recall,F1] for full training set
    # print("#################### FULL TRAINING SET ########## ")
    # print("################################################# ")
    # print("Final Precision for full training set= " + str(score_FullSet[0]))
    # print("Final Recall for full training set = " + str(score_FullSet[1]))
    # print("Final F1 for full training set= " + str(score_FullSet[2]))


















