#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/20

@author: GitMasters
"""

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from numpy import linalg as LA
import numpy as np


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

        l2_1 = LA.norm(item_weights_1, 2)
        l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        l2_2_scores = item_weights_2 / l2_2

        l2_3 = LA.norm(item_weights_3, 2)
        l2_3_scores = item_weights_3 / l2_3

        item_weights = item_weights_1*self.alpha + item_weights_2*self.beta + item_weights_3*self.gamma

        return item_weights

class ItemKNNScoresHybridTwoRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridTwoRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridTwoRecommender, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array = None, items_to_compute = None):
        
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        l2_1 = LA.norm(item_weights_1, 2)
        l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        l2_2_scores = item_weights_2 / l2_2
        

        item_weights = l2_1_scores*self.alpha + l2_2_scores*(1 - self.alpha)

        return item_weights

class ItemKNNScoresHybridOfHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridOfHybridRecommender"


    def __init__(self, URM_train, Recommender_Hybrid, Recommender_2, verbose = True):
        super(ItemKNNScoresHybridOfHybridRecommender, self).__init__(URM_train, verbose = verbose)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_Hybrid
        self.Recommender_2 = Recommender_2

        n_users = URM_train.shape[0]
        print(n_users)

        item_weights_1 = self.Recommender_1._compute_item_score(np.arange(n_users))
        item_weights_2 = self.Recommender_2._compute_item_score(np.arange(n_users))

        l2_1 = LA.norm(item_weights_1, 2)
        self.l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        self.l2_2_scores = item_weights_2 / l2_2
        
        
    def fit(self, alpha = 0.5):

        self.alpha = alpha

        '''
        print(f"CURRENT CONFIGURATION:\n{self.Recommender_1.RECOMMENDER_NAME} with weight alpha: {self.alpha}")
        print(f"{self.Recommender_2.RECOMMENDER_NAME} with weight beta: {1 - self.alpha}")
        '''

    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        l2_1 = LA.norm(item_weights_1, 2)
        l2_1_scores = item_weights_1 / l2_1

        l2_2 = LA.norm(item_weights_2, 2)
        l2_2_scores = item_weights_2 / l2_2

        item_weights = l2_1_scores*self.alpha + l2_2_scores*(1 - self.alpha)

        return item_weights