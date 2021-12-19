#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
sys.path.append('../RecSysRep/')


# In[7]:

if __name__ == "__main__":

    import Basics.Load as ld
    import numpy as np
    import scipy.sparse as sps

    np.random.seed(1234)

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


    # # SLIM Model

    # In[8]:


    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


    # In[9]:


    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender


    # In[10]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


    # In[12]:


    import os

    output_folder_path = "result_experiments/5_ASLRl"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    n_cases = 50  # using 10 as an example
    n_random_starts = int(n_cases*0.3)
    metric_to_optimize = "MAP"   
    cutoff_to_optimize = 10


    # In[13]:


    from skopt.space import Real, Integer, Categorical


    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]
            
    hyperparameters_range_dictionary = {
                "num_factors": Integer(20, 700),
                "epochs": Categorical([600]),
                "confidence_scaling": Categorical(["linear", "log"]),
                "alpha": Real(low = 1e-3, high = 50.0, prior = 'log-uniform'),
                "epsilon": Real(low = 1e-3, high = 10.0, prior = 'log-uniform'),
                "reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            }

    earlystopping_keywargs = {"validation_every_n": 5,
                            "stop_on_validation": True,
                            "evaluator_object": evaluator_validation,
                            "lower_validations_allowed": 5,
                            "validation_metric": metric_to_optimize,
                            }


    # In[14]:


    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = IALSRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                            evaluator_validation=evaluator_validation)


    # In[15]:


    from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = earlystopping_keywargs
    )


    # In[ ]:


    hyperparameterSearch.search(recommender_input_args,
                        hyperparameter_search_space = hyperparameters_range_dictionary,
                        n_cases = n_cases,
                        n_random_starts = n_random_starts,
                        output_folder_path = output_folder_path, # Where to save the results
                        output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                        metric_to_optimize = metric_to_optimize,
                        cutoff_to_optimize = cutoff_to_optimize
                        )


    # In[12]:


    from Recommenders.DataIO import DataIO

    data_loader = DataIO(folder_path = output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    search_metadata.keys()


    # In[16]:


    hyp = search_metadata["hyperparameters_best"]
    hyp


    recommender = IALSRecommender(URM_all)

    recommender.fit(**hyp)


    # In[ ]:


    import pandas as pd
    K=10

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


    submission.to_csv('../subs/submission {:%Y_%m_%d %H_%M_%S}.csv'.format(now), index=False)

