"""
Created on Fri Feb 11 11:26:04 2022
Recommender system - Retrieval Model using Tensorflow and based on the info at:
https://www.tensorflow.org/recommenders


@author: Francisco Mena
"""

# os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\Recommender_systems\\zTest")

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

#%%

class recsys_retrieval:
    
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

        
        """ For retrieval only select user and item, no need for ratings """

        data = data.map(lambda x:
                 {"user": x["user"],
                 "item": x["item"]}
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
    
    
    def create_towers(self, embedding_dimension = 32):
        
        item_names = self.data.map(lambda x: x["item"])
        user_ids = self.data.map(lambda x: x["user"])

        unique_item_names = np.unique( np.concatenate(list(item_names) ) )
        unique_user_ids = np.unique( np.concatenate(list(user_ids) ) )


        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_user_ids, mask_token = None),
            tf.keras.layers.Embedding( len(unique_user_ids) + 1, embedding_dimension)
            ])


        item_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_item_names, mask_token = None),
            tf.keras.layers.Embedding( len(unique_item_names) + 1, embedding_dimension)
            ])


        metrics = tfrs.metrics.FactorizedTopK(
            candidates= item_names.map(item_model)
          # candidates= item_names.batch(128).map(item_model)
        )
        
        #Loss
        task = tfrs.tasks.Retrieval(
          metrics=metrics
        )

        
        return user_model, item_model, task


    def make_prediction(self, model, pred_user_id, n = 3):
        
        item_names = self.data.map(lambda x: x["item"])

        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
          tf.data.Dataset.zip((item_names, item_names.map(model.item_model)))
        )

        # Get recommendations for a given user and item.
        _, titles = index(tf.constant([pred_user_id]))
        print(f"Recommendations for user {pred_user_id}: {titles[0, :n]}")

        return index

#%% Define Retrieval Class

class RetrievalModel(tfrs.Model):

    def __init__(self, user_model, item_model, task):
      super().__init__()
      self.user_model: tf.keras.Model = user_model
      self.item_model: tf.keras.Model = item_model
      self.task: tf.keras.layers.Layer = task
      

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        #We pick out the user id and pass them to the user model
        user_embeddings = self.user_model( features["user"] )
        #Then we pick the item names and pass them to the item model
        positive_item_embeddings = self.item_model( features["item"] ) #positive cause item has been used

        return self.task(user_embeddings, positive_item_embeddings)
