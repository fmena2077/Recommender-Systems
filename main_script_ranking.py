# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:39:46 2022

Script implementing a ranking recommender system

@author: Francisco Mena
"""

import os
import pandas as pd
import pprint
import tempfile



from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recsys_ranking import recsys_ranking, RecSysModel

#%%
print( tf.config.list_physical_devices() )

np.random.seed(42)
tf.random.set_seed(42)


#%%

recsys = recsys_ranking()

print("1. Collect data from csv file")

recsys.collect_data(filename = "ratings_clean.csv")



print("2. Train Test Split")

train, test = recsys.split_data()



print("3. Create towers and retrieval model")

ranking_model = RecSysModel(recsys.data)


ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))




""" If possible put the data to cache to improve effiency """
cached_train = train.shuffle(10000).cache()
cached_test = test.cache()



print("4. Train model")


early = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True, monitor = "factorized_top_k/top_50_categorical_accuracy")


ranking_model.fit(cached_train, epochs=100, verbose = 2,
          callbacks = [early])


print("5. Evaluate Model")

ranking_model.evaluate(test, return_dict=True)




print("\n 6. Making predictions for one client")

user_id = "10023"

print(
ranking_model({"user": np.array([user_id]), "item": np.array(["item16130"])} )
)






