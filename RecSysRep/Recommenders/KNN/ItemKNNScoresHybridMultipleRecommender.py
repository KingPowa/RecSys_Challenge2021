#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: GitMasters
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class ItemKNNScoresHybridMultipleRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*beta + R3*(1-alpha-beta)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridMultipleRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, verbose = True):
        super(ItemKNNScoresHybridMultipleRecommender, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        
        
    def fit(self, alpha = 0.5, beta = 0.5, gamma = 0.5):

        sump = alpha + beta + gamma

        self.alpha = alpha/sump
        self.beta = beta/sump
        self.gamma = gamma/sump

        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {self.beta}")
        print(f"{self.Recommender_3.RECOMMENDER_NAME} with weight gamma: {self.gamma}")


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1*self.alpha + item_weights_2*self.beta + item_weights_3*self.gamma

        return item_weights