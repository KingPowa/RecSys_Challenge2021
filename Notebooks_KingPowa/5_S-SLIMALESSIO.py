#!/usr/bin/env python
# coding: utf-8

# In[5]:



import sys
sys.path.append('../RecSysRep/')
import Basics.Load as ld
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import numpy as np
from scipy.sparse import *
import os
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
 
if __name__ == '__main__':


    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


    # In[7]:



    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    # evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])



    output_folder_path = "../result_experiments/S_SLIM_ElasticNet"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    n_cases = 30  # using 10 as an example
    n_random_starts = int(n_cases*0.3)
    metric_to_optimize = "MAP"   
    cutoff_to_optimize = 10


    # In[10]:


    hyperparameters_range_dictionary = {
        "l1_ratio": Real(low = 1e-2, high = 1.0, prior = 'log-uniform'),
        "topK": Integer(200, 2000),
        "alpha": Real(low = 1e-3, high = 2.0, prior = 'uniform'),
        "workers": Categorical([8]),
        "mw": Real(low=1, high=50, prior = 'log-uniform'),
    }


    # In[11]:


    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = MultiThreadSLIM_SLIM_S_ElasticNetRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)


    # In[12]:


    
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_genre_all],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {},
    )


    # In[13]:


    hyperparameterSearch.search(recommender_input_args,
        hyperparameter_search_space = hyperparameters_range_dictionary,
        n_cases = n_cases,
        n_random_starts = n_random_starts,
        output_folder_path = output_folder_path,
        output_file_name_root = recommender_class.RECOMMENDER_NAME,
        metric_to_optimize = metric_to_optimize,
        cutoff_to_optimize = cutoff_to_optimize,
    )
    

    #{'l1_ratio': 0.00015498813204475819, 'topK': 1000, 'alpha': 0.001, 'workers': 8, 'mw': 10} 0.468

    from Recommenders.DataIO import DataIO

    data_loader = DataIO(folder_path = output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    search_metadata.keys()

    hyp = search_metadata["hyperparameters_best"]
    print(hyp)

    recommender = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_all.tocsr(), ICM_genre_all)
    K = 10

    recommender.fit(l1_ratio=hyp["l1_ratio"], alpha=hyp["alpha"], workers=hyp["workers"], mw=hyp["mw"], topK=hyp["topK"])

    import pandas as pd

    user_test_path = '../data/data_target_users_test.csv'
    user_test_dataframe = pd.read_csv(filepath_or_buffer=user_test_path,
                                    sep=",",
                                    dtype={0:int})

    subm_set = user_test_dataframe.to_numpy().T[0]


    subm_res = {"user_id":[], "item_list":[]}

    for user_id in subm_set:
        subm_res["user_id"].append(user_id)
        res = recommender.recommend(user_id, K)
        res = ' '.join(map(str, res))
        if user_id < 3:
            print(user_id)
            print(res)
        subm_res["item_list"].append(res)


    # print(subm_res)

    submission = pd.DataFrame.from_dict(subm_res)
    # submission

    from datetime import datetime
    now = datetime.now() # current date and time


    submission.to_csv('../subs/submission {:%Y_%m_%d %H_%M_%S}SSLIMELASTICNET.csv'.format(now), index=False)




