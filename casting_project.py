import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('.//log.csv', header=None)
files = df.iloc[:,0]
labels = df.iloc[:,1]

images = np.empty([300,300,len(files)])

for i in range(len(files)):
    # print(files[i], labels[i])

    im = cv2.imread(files[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # print(im)
    # # print(np.shape(im))
    images[:,:,i] = im

for i in range(len(files)):

    plt.imshow(images[:,:,i])
    plt.title(f'{files[i]}, {labels[i]}')
    plt.show()
