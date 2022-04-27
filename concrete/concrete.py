#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt

import sys
from os.path import join
import random

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    df = pd.read_csv('.//Concrete_Data.csv', index_col=0)
    trainx, testx, trainy, testy = train_test_split(df.iloc[:,0:7].to_numpy(), df.iloc[:,7].to_numpy(), test_size=0.4, shuffle=True)

    # SEEDING THE RNG TO GET REPRODUCABLE RESULTS
    seed(1)
    tf.random.set_seed(2)

    np.set_printoptions(precision=2, suppress=True)

    linear_regression(trainx, trainy, testx, testy)

    # REDUCING THE INPUTS TO THE MOST SIGNIFICANT INPUTS
    # trx = trainx[:,0:5]
    # tex = testx[:,0:5]

    poly_basis(trainx, trainy, testx, testy, 1)
    # poly_basis(trx, trainy, tex, testy, 1)

    GPR(trainx, trainy, testx, testy)
    # GPR(trx, trainy, tex, testy)

    ANN(trainx, trainy, testx, testy)



def linear_regression(trainx, trainy, testx, testy):

    # SIMPLE LIENAR REGRESSION
    model = LinearRegression(fit_intercept=True)
    model.fit(trainx, trainy)
    print(f'Model weights: {model.coef_}')

    y = model.predict(testx)


    MSE = round(np.sum((testy-y)**2)/np.size(testy,0),2)
    x = range(0,np.size(testy))

    plt.title(f'Linear Regression, MSE = {MSE}')
    plt.scatter(testy,y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()

    # SIMPLE LINEAR REGRESSION WITH RIDGE CROSS VERIFICATION REGULARIZATION
    model = RidgeCV(fit_intercept=True)
    model.fit(trainx, trainy)
    print(f'Model weights: {model.coef_}')

    y = model.predict(testx)

    MSE = round(np.sum((testy-y)**2)/np.size(testy,0),2)
    x = range(0,np.size(testy))

    plt.title(f'Linear Regression with Ridge Regularization, MSE = {MSE}')
    plt.scatter(testy, y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()


    # SIMPLE LINEAR REGRESSION WITH LASSO CROSS VERIFICATION REGULARIZATION
    model = LassoCV(fit_intercept=True)
    model.fit(trainx, trainy)
    print(f'Model weights: {model.coef_}')

    y = model.predict(testx)

    MSE = round(np.sum((testy-y)**2)/np.size(testy,0),2)
    x = range(0,np.size(testy))

    plt.title(f'Linear Regression with Lasso Regularization, MSE = {MSE}')
    plt.scatter(testy, y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()


def poly_basis(trainx, trainy, testx, testy, alph):
    
    # LINEAR REGRESSION WITH POLYNOMIAL BASIS FUNCTION
    poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    poly_model.fit(trainx, trainy)

    y = poly_model.predict(testx)
    print(f'Model weights: {poly_model[1].coef_}')

    MSE = round(np.sum((y - testy)**2)/np.size(testy),2)

    plt.title(f'Linear Regression using Polynomial Basis, MSE = {MSE}')
    plt.scatter(testy, y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()

    # LINEAR REGRESSION WITH POLYNOMIAL BASIS FUNCTION AND LASSO REGULARIZATION
    poly_model = make_pipeline(PolynomialFeatures(3), LassoCV())
    poly_model.fit(trainx, trainy)

    y = poly_model.predict(testx)
    print(f'Model weights: {poly_model[1].coef_}')

    MSE = round(np.sum((y - testy)**2)/np.size(testy),2)

    plt.title(f'Lasso Regression using Polynomial Basis, MSE = {MSE}')
    plt.scatter(testy, y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()

    # LINEAR REGRESSION WITH POLYNOMIAL BASIS FUNCTION AND RIDGE REGULARIZAION
    poly_model = make_pipeline(PolynomialFeatures(3), RidgeCV())
    poly_model.fit(trainx, trainy)

    y = poly_model.predict(testx)
    print(f'Model weights: {poly_model[1].coef_}')

    MSE = round(np.sum((y - testy)**2)/np.size(testy),2)

    plt.title(f'Ridge Regression using Polynomial Basis, MSE = {MSE}')
    plt.scatter(testy, y,s=2)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.show()



def GPR(trainx, trainy, testx, testy):
    alph = 1e-6

    # LIST CONTAINING THE THREE KERNELS TO TRY: RBF, RATIONAL QUADRATIC, AND DOT PRODUCT
    kernels = [RBF(), RationalQuadratic(), DotProduct()]

    # ITERATING THROUGH AND TESTING EACH KERNEL
    for i in range(len(kernels)):
        kern = kernels[i]

        # GP REGRESSION WITH THE KERNEL
        gp_regressor = GaussianProcessRegressor(kernel=kern, alpha=alph, normalize_y=False)
        gp_regressor.fit(trainx, trainy)

        y = gp_regressor.predict(testx)

        MSE = round(np.sum((y - testy)**2)/np.size(testy), 2)
        print(f'Mean Squared Error of GPR using Kernel {kern} : {MSE}\n')

        plt.title(f'GPR with Kernel {i+1}, MSE = {MSE}', zorder=1) 
        plt.scatter(testy, y,s=2)
        plt.xlabel('True Output')
        plt.ylabel('Predicted Output')
        plt.show()




def ANN(trainx, trainy, testx, testy):

    # LIST CONTAINING THE 9 ANN MODELS TO TRY
    models = [

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='relu'),
        layers.Dense(1)
    ]), 

    keras.Sequential([
        layers.Dense(10, input_dim=7, activation='relu'),
        layers.Dense(1)
    ]), 

    keras.Sequential([
        layers.Dense(20, input_dim=7, activation='relu'),
        layers.Dense(1)
    ]), 

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ]), 

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='sigmoid'),
        layers.Dense(10, activation='sigmoid'),
        layers.Dense(1)
    ]), 


    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1)
    ]),

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='relu'),
        layers.Dense(10, activation='sigmoid'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1)
    ]),

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='swish'),
        layers.Dense(10, activation='swish'),
        layers.Dense(5, activation='swish'),
        layers.Dense(1)
    ]),

    keras.Sequential([
        layers.Dense(15, input_dim=7, activation='relu'),
        layers.Dense(10, activation='swish'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1)
    ])

    ]

    epochs = 200

    #ITERATING THROUGH AND TESTING EACH MODEL
    for j in range(len(models)):
        model = models[j]
        i = j+1

        # ANN TRAINING WITH THE ANN MODEL
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        model.summary()

        hist = model.fit(trainx, trainy, epochs=epochs, batch_size = 10, verbose=0)
        finalMSE = round(hist.history["loss"][-1], 2)

        y = model.predict(testx)
        MSE = round(np.sum((testy-np.transpose(y))**2)/np.size(testy),2)
        print(f'Model {i} MSE: {MSE}')

        plt.title(f'ANN Output Comparison, MSE = {MSE}')
        plt.scatter(testy, y,s=2)
        plt.xlabel('True Output')
        plt.ylabel('Predicted Output')
        
        plt.show()


if __name__=='__main__':
    main()
