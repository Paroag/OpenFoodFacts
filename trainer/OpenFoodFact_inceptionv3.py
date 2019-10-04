#!/usr/bin/env python
# coding: utf-8

#import os
#mport time

import argparse
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping

"""
Se base sur le guide :
    https://blog.francium.tech/training-deep-learning-models-with-google-cloud-ml-engine-432ae46068d8
"""


def train(job_dir) :
    
    ### GET THE DATA
    
    train_path = "gs://keras-image-dataforgood/"
    
    print("Loading Y_val ...")
    Y_val = pickle.load(file_io.FileIO(train_path + "Y_val.pickle", mode="rb"))
    
    print("Loading X_val ...")
    X_val = pickle.load(file_io.FileIO(train_path + "X_val.pickle", mode="rb"))
    
    print("Loading X_train1 ...")
    X_train1 = pickle.load(file_io.FileIO(train_path + "X_train1.pickle", mode="rb"))
    print("Loading X_train2 ...")
    X_train2 = pickle.load(file_io.FileIO(train_path + "X_train2.pickle", mode="rb"))
    print("Loading X_train3 ...")
    X_train3 = pickle.load(file_io.FileIO(train_path + "X_train3.pickle", mode="rb"))
    print("Loading X_train4 ...")
    X_train4 = pickle.load(file_io.FileIO(train_path + "X_train4.pickle", mode="rb"))

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis=0)
    
    del X_train1, X_train2, X_train3, X_train4
    
    print("Loading Y_train ...")
    Y_train = pickle.load(file_io.FileIO(train_path + "Y_train.pickle", mode="rb"))

    
    """
    print("Loading X_train ...")
    X_train = pickle.load(file_io.FileIO(train_path + "X_train.pickle", mode="rb"))
    print("DONE")
        print("Loading X_train ...")
    X_train = pickle.load(io.BytesIO(file_io.read_file_to_string(train_path + "X_train.pickle", binary_mode=True)))
    print("Loading Y_train ...")
    Y_train = pickle.load(io.BytesIO(file_io.read_file_to_string(train_path + "Y_train.pickle", binary_mode=True)))
    print("Loading X_val ...")
    X_val = pickle.load(io.BytesIO(file_io.read_file_to_string(train_path + "X_val.pickle", binary_mode=True)))
    print("Loading Y_val ...")
    Y_val = pickle.load(io.BytesIO(file_io.read_file_to_string(train_path + "Y_val.pickle", binary_mode=True)))
    # ref : https://stackoverflow.com/questions/44657902/how-to-load-numpy-npz-files-in-google-cloud-ml-jobs-or-from-google-cloud-storage?rq=1
    """
            
   ### KERAS MODEL

    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.20)(x)
    predictions = Dense(4, activation='softmax')(x) ## TODO change static 4
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])
    
    es = EarlyStopping(monitor='val_loss', patience=15)
    model.fit(X_train, Y_train, epochs=100, batch_size= 32, verbose = 1, validation_data = (X_val, Y_val), callbacks = [es])  
    
    model.save("cloud_ml_model.h5")
    
    with file_io.FileIO("cloud_ml_model.h5", mode = "rb") as input_f :
        with file_io.FileIO(job_dir + '/cloud_ml_model.h5', mode = "wb+") as output_f :
            output_f.write(input_f.read())
    
if __name__ == "__main__" :  
    
    ###
    """
    print("--------------------------------------")
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print("--------------------------------------")

   """
    ###

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
          '--job-dir',
          required=True
        )

    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')

    train(job_dir)
