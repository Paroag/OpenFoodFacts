#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pickle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator    

def data_prep(data_dir = "../Data/Scrap_Marques/", target_dir = "./pickle/", duplication_factor = 3, max_rotation_range = 30):
    
    marques = ["Barilla", "Bjorg", "Gerble", "Panzani"]
    
    # dictionnaire d'images sous la forme :
    # {path_image : label}
    
    dic_images = {}
    for index, marque in enumerate(marques) :
        path_images = data_dir + "Scrap_" + marque + "/Images/"
        for file_name in os.listdir(path_images) :
            if not file_name.startswith('.') :
                dic_images[path_images+file_name] = index

    # creation des numpy array X & Y
    X = np.zeros((len(dic_images.keys())*duplication_factor,200,200,3))
    Y = np.zeros((len(dic_images.keys())*duplication_factor, len(marques)))
    
    for index, key in enumerate(dic_images) :
        img = load_img(key, target_size = (200,200))
        img = img_to_array(img)
        samples = np.expand_dims(img, 0)
        datagen = ImageDataGenerator(rotation_range = max_rotation_range)
        it = datagen.flow(samples, batch_size=1)
        for k in range(duplication_factor) :
            batch = it.next()
            img = batch[0]
            X[index*duplication_factor + k, :,:,:] = img/255
            Y[index*duplication_factor + k, dic_images[key]] = 1
            
    # shuffle de X, Y   
    rand = np.arange(len(dic_images.keys())*duplication_factor)
    np.random.shuffle(rand)
    
    X = X[rand,:,:,:]
    Y = Y[rand,:]
    
    # split en train / val (no test yet)
    
    split = 0.8
    
    X_train = X[:round(X.shape[0]*split),:,:,:]
    Y_train = Y[:round(X.shape[0]*split),:]
    
    X_val = X[round(X.shape[0]*split):,:,:,:]
    Y_val = Y[round(X.shape[0]*split):,:]
    
    
    # saving data
    
    split_factor = 4
    
    X_train1 = X_train[:int(X_train.shape[0]/split_factor),:,:,:]
    X_train2 = X_train[int(X_train.shape[0]/split_factor):int(X_train.shape[0]/split_factor*2),:,:,:]
    X_train3 = X_train[int(X_train.shape[0]/split_factor)*2:int(X_train.shape[0]/split_factor*3),:,:,:]
    X_train4 = X_train[int(X_train.shape[0]/split_factor)*3:int(X_train.shape[0]),:,:,:]
    
    pickle.dump(X_train1, open(target_dir + 'X_train1.pickle', 'wb'), protocol=2)
    pickle.dump(X_train2, open(target_dir + 'X_train2.pickle', 'wb'), protocol=2)
    pickle.dump(X_train3, open(target_dir + 'X_train3.pickle', 'wb'), protocol=2)
    pickle.dump(X_train4, open(target_dir + 'X_train4.pickle', 'wb'), protocol=2)
    pickle.dump(Y_train, open(target_dir + 'Y_train.pickle', 'wb'), protocol=2)
    pickle.dump(X_val, open(target_dir + 'X_val.pickle', 'wb'), protocol=2)
    pickle.dump(Y_val, open(target_dir + 'Y_val.pickle', 'wb'), protocol=2)
    
    return

if __name__ == "__main__" :
    
    data_prep(duplication_factor = 3, max_rotation_range = 30)
