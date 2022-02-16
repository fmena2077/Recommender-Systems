""" Recommender Metrics from the work of Frank Kane (sundog-education.com)"""

import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        """
        Calculates Mean Absolute Error from scikit-surprise

        Parameters
        ----------
        predictions : recommender predictions

        Returns
        -------
        float
            MAE

        """
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        """
        Calculates Root Mean Squared Error from scikit-surprise

        Parameters
        ----------
        predictions : recommender predictions

        Returns
        -------
        float
            MAE

        """
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=4.0):
        """
        Get top n predictions

        Parameters
        ----------
        predictions : list
            List with scikit-surprise predictions.
        n : int, optional
            number of predictions. The default is 10.
        minimumRating : int, optional
            minimum rating to consider in the predictions. The default is 4.0.

        Returns
        -------
        topN : list
            top N recommender predictions.

        """
        topN = defaultdict(list)


        for userID, ItemID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append( (ItemID, estimatedRating) )

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        """
        Calculate the Hit Rate

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        leftOutPredictions : 
            List with items from scikit-surprise leave-one-out.

        Returns
        -------
        float
            Hit Rate.

        """
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutItemID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for ItemID, predictedRating in topNPredicted[int(userID)]:
                if (leftOutItemID == ItemID):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        """
        Calculate the Cumulative Hit Rate

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        leftOutPredictions : 
            List with items from scikit-surprise leave-one-out.
        ratingCutoff : int, optional
            minimum rating to consider. The default is 0.

        Returns
        -------
        float
            Cumululative Hit Rate.

        """
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for ItemID, predictedRating in topNPredicted[int(userID)]:
                    if ( leftOutItemID == ItemID ):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        """
        Calculates the Rating Hit Rate

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        leftOutPredictions : 
            List with items from scikit-surprise leave-one-out.

        Returns
        -------
        None, prints the rating hit rate.

        """
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for ItemID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutItemID) == ItemID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        """
        Calculates Average Reciprocal Hit Rank

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        leftOutPredictions : 
            List with items from scikit-surprise leave-one-out.

        Returns
        -------
        float
            Average Reciprocal Hit Rank.

        """
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for ItemID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if ( leftOutItemID == ItemID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        """
        Calculate User Coverage

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        numUsers : 
            Number of users.
        ratingThreshold : int, optional
            Minimum rating to consider. The default is 0.

        Returns
        -------
        float
            User Coverage.

        """
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for ItemID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        """
        Calculates diversity

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        simsAlgo : 
            KNNBaseline trained with the full Train set.

        Returns
        -------
        float
            Diversity.

        """
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                Item1 = pair[0][0]
                Item2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(Item1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(Item2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        """
        Calculate Novelty

        Parameters
        ----------
        topNPredicted : defaultdict
            Dictionary with scikit-surprise predictions.
        rankings : 
            rankings.

        Returns
        -------
        float
            Novelty.

        """
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                ItemID = rating[0]
                rank = rankings[ItemID]
                total += rank
                n += 1
        return total / n

