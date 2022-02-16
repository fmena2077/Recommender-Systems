# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:47:53 2022
Class that includes relevant methods to train recommender systems, 
based on the work of Frank Kane

@author: Francisco Mena
"""

import os

import pandas as pd
import numpy as np
import csv
from surprise import Dataset
from surprise import Reader

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise.model_selection import cross_validate
from surprise import KNNBaseline
from RecommenderMetrics import RecommenderMetrics

from datetime import datetime

#%%
class recsys:
    
    def __init__(self):
        self.itemID_to_name = {}
        self.name_to_itemID = {}
        self.algorithm = []
    
    def process_data(self, filename = "ratings_week.csv", short_test = False):
        """        
        Function to clean the example data
        Parameters
        ----------
        filename : string, optional
            CSV file with the example data. The default is "ratings_week.csv".
        short_test : boolean, optional
            To do a short calculation and only choose the first 1000 rows. The default is False.

        Returns
        -------
        None, saves clean data as a csv.

        """
                
        filepath = os.path.join(os.getcwd(), filename)
        df = pd.read_csv( filepath )
        
        filesave = "ratings_clean.csv"
        
        if short_test:
            df = df.iloc[:1000]
            filesave = "ratings_clean_short.csv"

        #max num of sessions in a machine
        df["max_sessions"] = df.groupby("player_id")["num_sesiones"].transform('max')

        #tag the items to distinguish raw id from inner id
        df["maquina"] = ["item" + str(x) for x in df["maquina"]]

        #at least 4 sessions to filter clients that play too little
        df = df.loc[ df["max_sessions"]>= 4 ]

        #RATING: Top machine is 10, the rest are scaled down relative to it
        ratings0 = df["num_sesiones"] / df["max_sessions"]
        df["rating"] = np.round( ratings0 * 10, 0 ).astype( int )

        df["rating"].replace({0:1}, inplace = True)

        df.rename(columns = {"player_id": "user", "maquina": "item"}, inplace = True)

        df = df[["user", "item", "rating"]]

        df.to_csv(filesave, index = False)
        
        # A reader is needed but only the rating_scale param is requiered.
        # reader = Reader(rating_scale=(1, 10))
            
        # # The columns must correspond to user id, item id and ratings (in that order).
        # data = Dataset.load_from_df(df, reader)

        # return data
    
    def create_surprise_data(self, filename = "ratings_clean.csv", rating_scale=(1, 10)):
        """        
        Creates a scikit-surprise data objet from a given CSV file. The CSV file must have
        three columns: user , item , rating
        
        Parameters
        ----------
        filename : string, optional
            CSV resulting from process_data function. The default is "ratings_clean.csv".
        rating_scale : tuple, optional
            The rating scale that the user uses to rate the item. The default is (1, 10).

        Returns
        -------
        None, saves results as a csv.

        """
        

        filepath = os.path.join(os.getcwd(), filename)
        self.itemID_to_name = {}
        self.name_to_itemID = {}

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=rating_scale)

        self.data = Dataset.load_from_file(filepath, reader=reader)

        with open(filepath, newline='') as csvfile:
                fileReader = csv.reader(csvfile)
                next(fileReader)  #Skip header line
                for row in fileReader:
                    itemID = int(row[0])                    
                    itemName = row[1]
                    # print(str(itemID) + '-' + str(itemName) )
                    self.itemID_to_name[itemID] = itemName
                    self.name_to_itemID[itemName] = itemID

        

    def getItemName(self, itemID):
        """
        Returns name of the item for a given item inner ID.
        Parameters
        ----------
        itemID : string
            ID of the item.

        Returns
        -------
        string
            Name of the item for a given item inner ID.

        """
        if itemID in self.itemID_to_name:
            return self.itemID_to_name[itemID]
        else:
            return ""
        
    def getItemID(self, itemName):
        """
        Returns the item ID for a given item name.
        Parameters
        ----------
        itemName : string
            Name of the item.

        Returns
        -------
        string
        ID of the item for a given item name.

        """
        if itemName in self.name_to_itemID:
            return self.name_to_itemID[itemName]
        else:
            return 0
        
    
    def GetAntiTestSetForUser(self, testSubject):
        """
        Calculates the anti-test set for a given user
        Parameters
        ----------
        testSubject : string
            ID of the user.

        Returns
        -------
        anti_testset : list
            Anti-test set for the given user.

        """
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject)) #raw_id to inner_id
        user_items = set([j for (j, _) in trainset.ur[u]]) #user items
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset
    
    
    def AddAlgorithm(self, algorithm):
        """
        Adds a model to the algorithm attribute.
        Parameters
        ----------
        algorithm : scikit-surprise model
            scikit-surprise model.

        Returns
        -------
        None.

        """
        self.algorithm.append(algorithm)
        
    
    def create_fulltrainset( self ):
        """
        Creates the full trainset.
        Returns
        -------
        None.

        """
        self.fullTrainSet = self.data.build_full_trainset()
        


    def Evaluate_crossval(self, cv_splits = 3, n_jobs = -1):
        """
        Calculates the cross-validation for the algorithms saved in self
        Parameters
        ----------
        cv_splits : int, optional
            The number of splits to do in cross-validation. The default is 3.
        n_jobs : int, optional
            Number of jobs to use in cross-validation. The default is -1.

        Returns
        -------
        res_crossval : pandas dataframe
            Dataframe with the mean RMSE and MAE of the cross-validation per algorithm.

        """
        benchmark = []
        

        start = datetime.now()

        for algorithm in self.algorithm:
            print("Starting: " ,str(algorithm))
            
            # Perform cross validation
            results = cross_validate(algorithm, self.data, measures=['RMSE','MAE'], cv=cv_splits, verbose=False, n_jobs = n_jobs)
            
            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)
            print("Done: " ,str(algorithm), "\n\n")

        res_crossval = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
        
        end = datetime.now()
        print("Calculation took : " +  str(end-start) )
        
        return res_crossval

    def Evaluate_topN(self, loocv_nsplits = 1, n=10, seed = 42, verbose=True):
        """
        Calculate metrics of Hit Rate, Culumative Hit Rate, Average Reciprocal Hit Rank

        Parameters
        ----------
        loocv_nsplits : int, optional
            Number of splits in the scikit-surprise Leave One Out. The default is 1.
        n : int, optional
            Number of item predictions to make. The default is 10.
        seed : int, optional
            Seed to set the random_state parameter. The default is 42.
        verbose : boolean, optional
            To print info or not. The default is True.

        Returns
        -------
        res_topN : pandas dataframe
            Dataframe with the resulting metrics.

        """

        if (verbose):
            print("Evaluating top-N with leave-one-out...")

        benchmark = []
        
        start = datetime.now()

            

        LOOCV = LeaveOneOut(n_splits=loocv_nsplits, random_state= seed)
        for train, test in LOOCV.split(self.data):
            LOOCVTrain = train
            LOOCVTest = test
            
        LOOCVAntiTestSet = LOOCVTrain.build_anti_testset()



        for algorithm in self.algorithm:
            print("Starting: " ,str(algorithm))
            metrics = {}
            # Evaluate top-10 with Leave One Out testing

            algorithm.fit( LOOCVTrain )
            leftOutPredictions = algorithm.test( LOOCVTest )
            # Build predictions for all ratings not in the training set
            allPredictions = algorithm.test( LOOCVAntiTestSet )
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
    
            name = [str(algorithm).split(' ')[0].split('.')[-1]]
            res = pd.DataFrame(metrics, index = [name])
            benchmark.append(res)
                        
            print("Done: " ,str(algorithm), "\n\n")

    
        res_topN = pd.concat(benchmark)

        #delete LOOCV data, not really necessary but if you have memory problems it might help
        del LOOCV, LOOCVTrain, LOOCVTest, LOOCVAntiTestSet

        end = datetime.now()
        print("Calculation took : " +  str(end-start) )


        return res_topN


    
    def Evaluate_fullSet(self, n= 10, ratingThreshold = 4.0, verbose = True):
        """
        Calculate User Coverage and Diversity
        Parameters
        ----------
        n : int, optional
            Number of predictions to make. The default is 10.
        ratingThreshold : int, optional
            Only consider predictions above the ratingThreshold. The default is 4.0.
        verbose : boolean, optional
            To print info or not. The default is True.

        Returns
        -------
        res_Eval : pandas dataframe
            Datafarme with coverage and diversity per algorithm.

        """
        #Evaluate properties of recommendations on full training set
        benchmark = []
        if (verbose):
            print("Computing recommendations with full data set...")
        
        
        #Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)
        
        
        for algorithm in self.algorithm:
            metrics = {}

            algorithm.fit( self.fullTrainSet )
            fullAntiTestSet = self.fullTrainSet.build_anti_testset()
            allPredictions = algorithm.test( fullAntiTestSet )
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(  topNPredicted, 
                                                                  self.fullTrainSet.n_users, 
                                                                   ratingThreshold= ratingThreshold)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, self.simsAlgo )
            
            # Measure novelty (average popularity rank of recommendations):
            # metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted, 
            #                                                 evaluationData.GetPopularityRankings())
        
            name = [str(algorithm).split(' ')[0].split('.')[-1]]
            res = pd.DataFrame(metrics, index = [name])
            benchmark.append(res)

        
        if (verbose):
            print("Analysis complete.")
        
        
        res_Eval = pd.concat(benchmark)
        
        return res_Eval

    
    def SampleTopNRecs(self, algorithm, testSubject, k=10):
        """
        Make top k recommendations for a given user

        Parameters
        ----------
        algorithm : scikit-surprise model
            trained algorithm chosen to use for the recommendations.
        testSubject : string, optional
            User ID
        k : int, optional
            The number of top recommendations to return. The default is 10.

        Returns
        -------
        None, prints recommendations and their predicted ratings.

        """
        
        
        print("\nUsing recommender ", str(algorithm) )
        
        print("\nBuilding recommendation model...")
        trainSet = self.fullTrainSet
        algorithm.fit(trainSet)
        
        print("Computing recommendations...")
        testSet = self.GetAntiTestSetForUser(testSubject)
    
        predictions = algorithm.test(testSet)
        
        recommendations = []
        
        print ("\nFor client " + str(testSubject)  + " we recommend:")
        for userID, ItemID, actualRating, estimatedRating, _ in predictions:

            recommendations.append((ItemID, estimatedRating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        for ratings in recommendations[:k]:
            
            print((ratings[0], ratings[1]))
            

#%%
""" Other References:
    https://colab.research.google.com/github/singhsidhukuldeep/Recommendation-System/blob/master/Building_Recommender_System_with_Surprise.ipynb#scrollTo=mBTeRS7GDC-D
    https://stackoverflow.com/questions/65282827/how-to-make-predictions-with-scikits-surprise

"""