import time
import numpy as np
from knn import main as knn_main
from Best_model import main as Best_main
#from decision_tree import main as decision_tree_main



def main():
    X_train = np.genfromtxt('../../Data/x_train.csv', delimiter=',')  # shape = (200000, 29)
    y_train = np.genfromtxt('../../Data/y_train.csv', delimiter=',')  # shape = (200000,)
    Best_main(X_train, y_train)
    knn_main(X_train, y_train)
    #decison_tree_main(X_train, y_train)


















if __name__ == '__main__':
    main()









