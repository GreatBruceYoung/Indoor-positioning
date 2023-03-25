# This is a program that can realize some basic algorithms in indoor positioning field

# Dependent package
import numpy as np
from function import accuracy,positioning
import scipy.io as scio

# Input data
fingerprint_train = scio.loadmat('data/offline_data_random.mat')  # fingerprint database for training
fingerprint_test = scio.loadmat('data/online_data.mat')  # fingerprint data for test
pos_train, rss_train = fingerprint_train['offline_location'], fingerprint_train['offline_rss']
pos_test, rss_test = fingerprint_test['trace'], fingerprint_test['rss']

# Positioning experiments
# KNN regression positioning
pos_knn=positioning.knn_positioning(rss_train,pos_train,rss_test)
rmse_knn = accuracy.rmse_calculator(pos_knn,pos_test)
print('RMSE of KNN:',rmse_knn/100,'m')

# KNN positioning, grid crossing validation
pos_knn=positioning.knn_crossing_positioning(rss_train,pos_train,rss_test,pos_test)
rmse_knn = accuracy.rmse_calculator(pos_knn,pos_test)
print('RMSE of KNN (with optimal k):',rmse_knn/100,'m')

# Support Vector Machine for Regression (SVR) in indoor positioning
pos_svr=positioning.svr_positioning(rss_train,pos_train,rss_test)
rmse_svr=accuracy.rmse_calculator(pos_svr,pos_test)
print('RMSE of SVR:',rmse_svr/100,'m')

# Random forest for positioning
pos_rf=positioning.rf_positioning(rss_train,pos_train,rss_test)
rmse_rf=accuracy.rmse_calculator(pos_rf,pos_test)
print('RMSE of RF:',rmse_rf/100,'m')

# Gradient Boosting regression for positioning
pos_gb=positioning.gb_positioning(rss_train,pos_train,rss_test)
rmse_gb = accuracy.rmse_calculator(pos_gb,pos_test)
print('RMSE of Gradient Boosting:',rmse_gb/100,'m')

# Multi-layer Perceptron (MLP) regressor for positioning
pos_mlp=positioning.mlp_positioning(rss_train,pos_train,rss_test)
rmse_mlp=accuracy.rmse_calculator(pos_mlp,pos_test)
print('RMSE of MLP:',rmse_mlp/100,'m')
