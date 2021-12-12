#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys

sys.path.append('../RecSysRep/')

import Basics.Load as ld
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
import numpy as np
from scipy.sparse import *
import os
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

if __name__ == '__main__':

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)
    ICM_length_all_5bal = ld.getICMlength('5bal')
    ICM_length_all_3bal = ld.getICMlength('3bal')
    ICM_length_all_5km = ld.getICMlength('5km')
    ICM_length_all_3km = ld.getICMlength('5km')

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    # evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    output_folder_path = "../result_experiments/S_SLIM_ElasticNet_genre_subgenre_channel_5bal"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 60
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # In[10]:

    hyperparameters_range_dictionary = {
        "l1_ratio": Real(low=1e-2, high=1e-1, prior='log-uniform'),
        "topK": Integer(500, 2000),
        "alpha": Real(low=1e-4, high=1e-1, prior='uniform'),
        "workers": Categorical([4]),
        "mw": Integer(1, 20),
    }

    # In[11]:

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = MultiThreadSLIM_SLIM_S_ElasticNetRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation)

    # In[12]:

    ICM_stacked = hstack((ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_length_all_5bal))

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_stacked],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
    )

    # In[13]:

    hyperparameterSearch.search(recommender_input_args,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                resume_from_saved=True,
                                )
