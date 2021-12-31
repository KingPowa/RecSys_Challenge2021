#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender



class ItemKNNSimilarityHybridRecommenderNormal(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, verbose = True):
        super(ItemKNNSimilarityHybridRecommenderNormal, self).__init__(URM_train, verbose = verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')


    def fit(self, alpha = 0.5):

        self.alpha = alpha

        W_sparse = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        #self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = W_sparse
        self.W_sparse = check_matrix(self.W_sparse, format='csr')