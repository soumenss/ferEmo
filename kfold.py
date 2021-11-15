import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('fer2013.csv') #Reading the FER2013 dataset
data = data.sample(frac= 1)

fold_number = 5

data['KFold'] = range(len(data))
data['KFold'] = data['KFold']%fold_number
data.drop('Usage', inplace= True, axis= 1)

num_classes = 7
width, height = 48, 48
num_epochs = 100
batch_size = 64
num_features = 64

def CRNO(df):
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    #pixel_sequence = df['pixels']
    #df['pixels'] = [int(pixel) for pixel in pixel_sequence.split()]
    data_X = np.array(df['pixels'].tolist(), dtype = 'float32').reshape(-1, width, height, 1)/255.0
    data_Y = to_categorical(df['emotion'], num_classes)
    return data_X, data_Y

data_generator = ImageDataGenerator(featurewise_center = False,
                                    featurewise_std_normalization = False,
                                    rotation_range = 20,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    zoom_range = 0.1,
                                    horizontal_flip = True)

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', restore_best_weights = True)
# es = EarlyStopping(monitor = 'val_accuracy', patience = 30, mode = 'max', restore_best_weights = True)

scorelist = []

for kf in range(fold_number):
    data_train = data[data['KFold'] != kf].copy()
    data_val = data[data['KFold'] == kf].copy()
    
    # print(data_train.shape, data_val.shape)
    
    train_X, train_Y = CRNO(data_train)
    val_X, val_Y = CRNO(data_val)
    
    model = Sequential()

    #module 1
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding = 'same', input_shape=((width, height, 1)), data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #module 2
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #module 3
    model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #flatten
    model.add(Flatten())

    #dense 1
    model.add(Dense(2*2*2*num_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #dense 2
    model.add(Dense(2*2*num_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #dense 3
    model.add(Dense(2*num_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
                  metrics=['accuracy'])

    # model.summary()
    
    history = model.fit(data_generator.flow(train_X, train_Y, batch_size),
                    steps_per_epoch = len(train_X) / batch_size,
                    epochs = num_epochs,
                    verbose = 0, 
                    callbacks = [es],
                    validation_data = (val_X, val_Y))
    
    score = model.evaluate(val_X, val_Y, verbose=1)
    print(score[1])
    scorelist.append(score[1])
    
    # Creating and saving confusion matrix
    pred_Y=model.predict(val_X)
    tesst_Y=np.argmax(val_Y, axis=1)
    pred_Y=np.argmax(pred_Y,axis=1)
    cmatrix=confusion_matrix(tesst_Y, pred_Y)
    cmatrix = cmatrix.astype('float')/cmatrix.sum(axis=1)[:, np.newaxis]
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cmatrix, annot=True, fmt= '.2f', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_'+str(kf)+'_val_acc_'+str(score[1])+'_4_fold.png')

np.savetxt("test_accuracy.csv", scorelist, delimiter =",")