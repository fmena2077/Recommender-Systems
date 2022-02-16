# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:19:27 2022
Recommender Systems
Example script to collect data, create a scikit-surprise dataset, train model, 
generate performance metrics, and predict rankings

@author: Francisco Mena
"""


from datetime import datetime

from recsys import recsys


start = datetime.now()

#%%

recsys = recsys()

""" Process data and create ratings"""
recsys.process_data(filename = "ratings_week.csv", short_test = False)


""" Collect data """
recsys.create_surprise_data(filename = "ratings_clean_short.csv")



""" Create full trainset """
recsys.create_fulltrainset()


print('Number of users: ', recsys.fullTrainSet.n_users, '\n')
print('Number of items: ', recsys.fullTrainSet.n_items, '\n')


#%% 

""" Choose models """

# from surprise import NormalPredictor, SVD, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import NormalPredictor, SVD, KNNWithMeans, KNNBasic

#Recommendation: include NormalPredictor() as a baseline
recsys.AddAlgorithm( NormalPredictor() )
recsys.AddAlgorithm( SVD(random_state = 42) )
recsys.AddAlgorithm( KNNWithMeans(random_state = 42) )

"""User-based KNN"""
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
recsys.AddAlgorithm(UserKNN)

"""Item-based KNN"""
ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
recsys.AddAlgorithm(ItemKNN)

#%% 

""" Evaluate models """

res_crossval = recsys.Evaluate_crossval()
print(res_crossval)
#fit_time & test_time: training time in seconds for each split.


topN = recsys.Evaluate_topN()
print(topN)


evals = recsys.Evaluate_fullSet()
print(evals)

#%% 

""" Make predictions """

model = recsys.algorithm[1]

recsys.SampleTopNRecs(algorithm = model, testSubject=10023, k=10)


#%%

end = datetime.now()
print("Calculation took : " +  str(end-start) )
