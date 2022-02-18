# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:59:53 2022

Recommender system - Ranking Model using Tensorflow and based on the info at:
https://www.tensorflow.org/recommenders


@author: Francisco Mena
"""

# os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\Recommender_systems\\zTest")

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

#%%

class recsys_ranking:
    
    def __init__(self):
        self.data = None
    
    def collect_data(self, filename, batch_size = 10000, seed = 42, num_parallel_reads=None):
        
        data = tf.data.experimental.make_csv_dataset(
            filename, batch_size = batch_size, column_names=["user", "item", "rating"], 
            column_defaults=[tf.string, tf.string, tf.int32],
            header=True, num_epochs=1,
            shuffle=True, shuffle_buffer_size=10000, shuffle_seed=seed,
            prefetch_buffer_size=None, num_parallel_reads=num_parallel_reads, sloppy=False,
            num_rows_for_inference=100, compression_type=None, ignore_errors=False
        )

        

        
        self.data = data
    
    
    def split_data(self, n = 4):
        """
        Split data into train and test. Every n-th sample goes into test set.

        Parameters
        ----------
        data : tf.data
            Tensorflow data object with the user,item,rating data.
        n : int, optional
            Every n-th sample goes into the test set. The default is 4.

        Returns
        -------
        train : tf.data
            train set.
        test : tf.data
            test set.

        """
        
        n = int(n)
        
        test = self.data.enumerate() \
                            .filter(lambda x,y: x % n == 0) \
                            .map(lambda x,y: y)

        train = self.data.enumerate() \
                            .filter(lambda x,y: x % n != 0) \
                            .map(lambda x,y: y)

        return train, test
    


#%% Create the class of the Recommender Model

class RankingModel(tf.keras.Model):

  def __init__(self, data, embedding_dimension = 32):
    super().__init__()


    item_names = data.map(lambda x: x["item"])
    user_ids = data.map(lambda x: x["user"])
    
    unique_item_names = np.unique( np.concatenate(list(item_names) ) )
    unique_user_ids = np.unique( np.concatenate(list(user_ids) ) )



    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.item_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_item_names, mask_token=None),
      tf.keras.layers.Embedding(len(unique_item_names) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer: 1= positive review, 0 = neg review
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, item_name = inputs

    user_embedding = self.user_embeddings(user_id)
    item_embedding = self.item_embeddings(item_name)

    return self.ratings(tf.concat([user_embedding, item_embedding], axis=1))


#%% Define Ranking Class



class RecSysModel(tfrs.models.Model):

  def __init__(self, data):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(data)
    #Loss
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user"], features["item"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("rating")

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)




#%%
