#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

import sys
from os.path import join
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from heatmapplot import *

def main():
    input_path = './casting_data/'

    test = input_path+'test/'

    train = input_path+'train/'

if __name__=='__main__':
    main()
