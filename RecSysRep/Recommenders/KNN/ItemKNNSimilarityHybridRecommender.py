#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from numpy import linalg as LA



class ItemKNNSimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, verbose = True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__(URM_train, verbose = verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')


    def fit(self, topK=None, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W_sparse = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        if (topK is not None):
            self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(W_sparse, format='csr')


class ItemKNNSimilarityHybridRecommender_L2(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, verbose = True):
        super(ItemKNNSimilarityHybridRecommender_L2, self).__init__(URM_train, verbose = verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        l2_1 = LA.norm(Similarity_1, 2)
        l2_1_sim = Similarity_1 / l2_1

        l2_2 = LA.norm(Similarity_2, 2)
        l2_2_sim = Similarity_2 / l2_2

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(l2_1_sim.copy(), 'csr')
        self.Similarity_2 = check_matrix(l2_2_sim.copy(), 'csr')


    def fit(self, topK=None, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W_sparse = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        if (topK is not None):
            self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')