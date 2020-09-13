import math
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import KFold
from matplotlib.legend_handler import HandlerLine2D

def decision_tree_classsifier(depth,min_samples, training_input, training_output, validation_input):
    clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=5)
    start = time.time()
    clf.fit(training_input, training_output)
    end = time.time()
    return clf.predict(validation_input), (end - start) * 1000




def main(X, y): # X= X_train, y=y_train

    X_t = X  #X_train
    y_t = y  #y_train

    x_test = np.genfromtxt('../../Data/x_test.csv', delimiter=',')  # shape = (50000, 29)
    """
    I found that my best model was a decision tree classifier with max_depth= 6 for the current dataset
    """
    # x_predicted, run_time = decision_tree_classsifier(6, 5, X_t, y_t, x_test)
    # np.savetxt("Best_Predictions_For_x_test.csv", x_predicted, delimiter=",")

    """
    I will plot 2 lines ->test_results and train_results on graph -> y axis will correspond to AUC score
     (for different d) and x axis would be different hyperparameters 
     (I am just trying depth and we can do similar stuff for different
    hyperparameters).  The graph will let us know if my model is over fitting or underfitting. 
    """
    test_results=[]
    train_results= []
    def k_fold_validation(depth=6, train_x=None, train_y=None):

        X = train_x
        y = train_y
        kf = KFold(n_splits=5)
        kf.get_n_splits(train_x)
        F1 = 0
        AUC = 0
        precision = 0
        recall = 0


        for i, (train_index, test_index) in enumerate(kf.split(X)):
            #print(" running for FOLD=" + str(i + 1))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            predicted, run_time = decision_tree_classsifier(depth,10, X_train, y_train, X_test )
            precision+=metrics.precision_score(y_test, predicted)
            recall+= metrics.recall_score(y_test, predicted)
            F1+= metrics.f1_score(y_test,predicted)
            AUC+=metrics.roc_auc_score(y_test,predicted)

        print("Final Precision for all the 5 folds is " + str(precision / 5))
        print("Final Recall for all the 5 folds is " + str(recall / 5))
        print("Final F1 for all the 5 folds is " + str(F1 / 5))
        print("Final AUC for all the 5 folds is " + str(AUC / 5))
        test_results.append(AUC / 5)



    D_list = [3,6,9,12,15]
    training_time=[]  #used to plot graph


    for D in D_list: #hyper parameter tuning for choosing the best Depth
        print("#######################################################")
        print("Decision Tree FOR D="+str(D))
        print("#######################################################")

        print("-----------Avg Validation Set ----------------")
        start = time.time()
        k_fold_validation(D, X_t, y_t) #K FOLD VALIDATION
        end = time.time()
        runtime = (end - start) * 1000  # covert to ms
        training_time.append(runtime)
        print("-----------Full Training Set ----------------")
        predicted_full, run_time_full = decision_tree_classsifier(D, 5, X_t, y_t, X_t)
        precision_full = metrics.precision_score(y_t, predicted_full)
        recall_full = metrics.recall_score(y_t, predicted_full)
        F1_full = metrics.f1_score(y_t, predicted_full)
        AUC_full = metrics.roc_auc_score(y_t, predicted_full)
        print("Final Precision for Full Training Set= " + str(precision_full))
        print("Final Recall for Full Training Set= " + str(recall_full))
        print("Final F1 for Full Training Set= " + str(F1_full))
        print("AUC for Full Training Set= " + str(AUC_full))
        train_results.append(AUC_full)






    plt.figure("DECISION TREE TIME FOR EACH MODEL")
    plt.plot(D_list, training_time)
    plt.xlabel("Value of D")
    plt.ylabel("Execution time in ms")
    plt.show()

    plt.figure("Hyper-parameter tuning for Max-Depth")
    line1, = plt.plot(D_list, train_results, 'b', label ="Train AUC")
    line2, = plt.plot(D_list, test_results, 'r', label ="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("Tree depth")
    plt.show() #we can see how how we start to overfit with increasing depth.  we can choose d=6 for our  best model




