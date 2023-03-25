from function import accuracy
import scipy.io as scio
from sklearn import neighbors, svm, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise


def knn_positioning(rss_train,pos_train,rss_test):
    """KNN positioning"""
    k = 30
    knn = neighbors.KNeighborsRegressor(k, weights='uniform', metric='euclidean')
    knn.fit(rss_train, pos_train)
    pos_knn = knn.predict(rss_test)
    return pos_knn


def knn_crossing_positioning(rss_train,pos_train,rss_test,pos_test):
    """KNN positioning with grid crossing validation to choose optimal k"""
    # Pre-process: Data normalization
    standard_scaler = StandardScaler().fit(rss_train)
    X_train = standard_scaler.transform(rss_train)
    Y_train = pos_train
    X_test = standard_scaler.transform(rss_test)
    Y_test = pos_test

    # Grid crossing validation
    parameters = {'n_neighbors': range(1, 50)}
    knn_reg = neighbors.KNeighborsRegressor(weights='uniform', metric='euclidean')
    clf = GridSearchCV(knn_reg, parameters)  # Crossing validation through grid method
    clf.fit(rss_train, pos_train)
    scores = clf.cv_results_['mean_test_score']
    optimal_k = np.argmax(scores)  # choose the best K value

    # Draw a hyperparameter K-score relationship diagram
    # plt.plot(range(1, scores.shape[0] + 1), scores, '-o', linewidth=2.0)
    # plt.xlabel("k")
    # plt.ylabel("score")
    # plt.grid(True)
    # plt.title("K-score Diagram")
    # plt.show()

    # Ues best k to achieve knn positioning
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors=optimal_k, weights='uniform', metric='euclidean')
    knn_reg.fit(rss_train, pos_train)
    pos_knn = knn_reg.predict(rss_test)
    return pos_knn


def svr_positioning(rss_train,pos_train,rss_test):
    """Support Vector Regression for positioning"""
    clf_x = svm.SVR(kernel='rbf', C=1e3, gamma=0.01)
    clf_y = svm.SVR(kernel='rbf', C=1e3, gamma=0.01)
    clf_x.fit(rss_train, pos_train[:, 0])
    clf_y.fit(rss_train, pos_train[:, 1])
    svr_x = clf_x.predict(rss_test)
    svr_y = clf_y.predict(rss_test)
    pos_svr = np.column_stack((svr_x, svr_y))
    return pos_svr


def rf_positioning(rss_train,pos_train,rss_test):
    """Random Forest for positioning"""
    rf_estimator = RandomForestRegressor(n_estimators=150)
    rf_estimator.fit(rss_train, pos_train)
    pos_rf = rf_estimator.predict(rss_test)
    return pos_rf


def gb_positioning(rss_train,pos_train,rss_test):
    """Gradient Boosting regression for positioning"""
    clf = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=10))
    clf.fit(rss_train, pos_train)
    pos_gb = clf.predict(rss_test)
    return pos_gb


def mlp_positioning(rss_train,pos_train,rss_test):
    """Multi-layer Perceptron (MLP) regressor for positioning"""
    clf = MLPRegressor(hidden_layer_sizes=(100, 100))
    clf.fit(rss_train, pos_train)
    pos_mlp = clf.predict(rss_test)
    return pos_mlp

