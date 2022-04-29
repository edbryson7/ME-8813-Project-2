import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

def main():
    # np.set_printoptions(threshold=sys.maxsize)

    train_path = './/casting_data//train//'
    test_path = './/casting_data//test//'
    traindf = pd.read_csv(train_path+'train.csv')
    testdf = pd.read_csv(test_path+'test.csv')

    # trainx, trainy = init_df(traindf, train_path)
    testx, testy = init_df(testdf, test_path)
    print(testy)

def init_df(df, path):
    df = df.sample(frac=1)
    files = df['image'].to_numpy()
    print(files)
    labels = df['label'].to_numpy()
    # labels = [1 for lab in labels if lab == ' def', else 0]

    images = np.empty([300,300,len(files)])

    for i in range(len(files)):
        im = cv2.imread(path+files[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images[:,:,i] = im

    return images, labels

if __name__ == '__main__':
    main()
