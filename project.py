#!/usr/bin/env python3
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

from heatmapplot import *

def main():
    # np.set_printoptions(threshold=sys.maxsize)
    try:
        mode = int(sys.argv[1])
    except:
        mode = 0

    mode = 2

    train_path = './/casting_data//train//'
    test_path = './/casting_data//test//'
    traindf = pd.read_csv(train_path+'train.csv')
    testdf = pd.read_csv(test_path+'test.csv')

    trainx, trainy = init_df(traindf, train_path)
    trainx = np.expand_dims(trainx, axis=-1) # <--- add batch axisp
    testx, testy = init_df(testdf, test_path)
    testx = np.expand_dims(testx, axis=-1) # <--- add batch axisp

    # print(np.shape(trainx))
    # print(np.shape(trainy))

    if mode == 1:
        model = init_cnn(np.shape(trainx[0]))
        train_cnn(model, trainx, trainy, testx, testy)
    elif mode == 2:
        test_cnn(testx, testy)
    else:
        model = init_cnn(np.shape(trainx[0]))
        train_cnn(model, trainx, trainy, testx, testy)
        test_cnn(testx, testy)



def init_cnn(inp_shape):
    print('='*90)
    print(inp_shape)
    model = models.Sequential()

    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=inp_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
    return model

def train_cnn(model, images, labels, test_images, test_labels):
    # model.summary()

    print('='*90)
    history = model.fit(images, labels, epochs=15, batch_size=50, verbose=1,
                    validation_data=(test_images, test_labels))

    # print('Finished training')
    model.save('./model')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.title('CNN Training Accuracy')
    plt.savefig('cnn_acc_history.png',dpi=1200)
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.title('CNN Training Loss')
    plt.savefig('cnn_loss_history.png',dpi=1200)
    plt.show()
    return model

def test_cnn(test_images, test_labels):
    model = tf.keras.models.load_model('./model')
    model.summary()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(f'Validation Loss {test_loss}')
    print(f'Validation Accuracy {test_acc}')

    y = model.predict(test_images)

    # Pulling the categorized label
    y = np.round(y)
    y = y.astype(int)

    assert len(y)==len(test_labels)

    # Creating a heatmap
    heat = np.zeros((2,2))#,dtype=int)
    for i in range(len(y)):
        heat[y[i],test_labels[i]]+=1

    for i in range(2):
        heat[:,i]=heat[:,i]/sum(heat[:,i])

    fig, ax = plt.subplots()
    plt.title(f'CNN Validation Confusion Matrix: {round(test_acc*100,3)}% Accuracy\n0 = Okay, 1 = Deformed\n')
    plt.xlabel('Correct State')
    plt.ylabel('Predicted State')

    im, cbar = heatmap(heat, range(2), range(2), ax=ax, cmap='copper',
            cbarlabel='correct predictions')

    texts = annotate_heatmap(im,heat,textcolors=('white','black'))
    fig.tight_layout()
    plt.savefig('cnn_heatmap.png', dpi=1200)
    plt.show()



def init_df(df, path):
    df = df.sample(frac=1)
    files = df['image'].to_numpy()
    labels = df['label'].to_numpy()

    images = np.empty([len(files),300,300])

    for i in range(len(files)):
        im = cv2.imread(path+files[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images[i,:,:] = im

    return images, labels.flatten()

if __name__ == '__main__':
    main()
