# Recommender Systems

This repository shows how to implement recommender systems using
the library scikit-surprise and tensorflow recommenders

Files for scikit-surprise models:

* recsys.py: Class that includes all the relevant methods to create a recommender model


* RecommenderMetrics.py: Performance metrics for recommender systems from the work of Frank Kane (sundog-education.com)


* main_script.py: Example script of how to create a scikit-surprise dataset object, 
  choose different algorithms and train them, compare performance metrics, and generate predictions
  
Files for Tensorflow Retrieval model:

* recsys_retrieval.py: Class that includes all the relevant methods to create a retrieval model with Tensorflow 


* main_script_retrieval.py: Example script of how to read from csv and create a tf.data object, 
  create a retrieval model using tensorflow-recommenders, and recommend items
  
Files for Tensorflow Retrieval model:

* recsys_ranking.py: Class that includes all the relevant methods to create a ranking model 


* main_script_retrieval.py: Example script of how to read from csv and create a tf.data object, 
  create a retrieval model using tensorflow-recommenders, and recommend items
  
