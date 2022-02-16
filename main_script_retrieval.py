# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:51:16 2022
Script implementing a retrieval recommender system

@author: Francisco Mena
"""

import os
import pandas as pd
import pprint
import tempfile

# os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\Recommender_systems\\zTest")

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recsys_retrieval import recsys_retrieval, RetrievalModel

#%%

recsys = recsys_retrieval()

print("1. Collect data from csv file")

recsys.collect_data(filename = "ratings_clean.csv")



print("2. Train Test Split")

train, test = recsys.split_data()



print("3. Create towers and retrieval model")

user_model, item_model, task = recsys.create_towers()


model = RetrievalModel(user_model, item_model, task)

model.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1))


""" If possible put the data to cache to improve effiency """
cached_train = train.shuffle(10000).cache()
cached_test = test.cache()



print("4. Train model")


early = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True, monitor = "factorized_top_k/top_50_categorical_accuracy")


model.fit(cached_train, epochs=100, verbose = 2,
          callbacks = [early])


print("5. Evaluate Model")

model.evaluate(test, return_dict=True)




print("\n 6. Making predictions for one client")

user_id = "10023"
index = recsys.make_prediction(model, user_id)


#%% Export model

print("n\7. Exporting the Model and Saving it for later Use")


path = os.path.join(os.getcwd(), "model")

# Save the index.
tf.saved_model.save(index, path)


# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded([user_id])

print(f"Recommendations: {titles[0][:3]}")


#%%












