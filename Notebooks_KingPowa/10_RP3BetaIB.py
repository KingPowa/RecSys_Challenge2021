#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


# In[17]:


if __name__ == "__main__":
    import Basics.Load as ld

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    ICM_length_all = ld.getICMlength()
    ICM_all = sps.hstack([ICM_genre_all, ICM_channel_all, ICM_length_all])
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


    # In[18]:


    import os

    output_folder_path = "../result_experiments/RP3Beta_Item_BasedOpt/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    # # SLIM Model

    # In[19]:


    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    #URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


    # In[20]:


    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommenderICM

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    class TrainUserBased(object):
        def __init__(self, URM_train, ICM_all, evaluator):
            # Hold this implementation specific arguments as the fields of the class.
            self.URM_train = URM_train
            self.evaluator = evaluator

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            
            search_args = { "alpha": trial.suggest_uniform('alpha', 0, 2),
                            "alpha": trial.suggest_uniform('beta', 0, 2),  
                            "topK": trial.suggest_int('topK', 1000, 8000), 
                            "normalize_similarity": trial.suggest_categorical("normalize_similarity", [True, False]),
                            "implicit": True}
            
            #omega = trial.suggest_uniform('omega', 0.1, 0.9)

            recommender = RP3betaRecommenderICM(URM_train, ICM_all)
            recommender.fit(**search_args)
            result_dict, _ = self.evaluator.evaluateRecommender(recommender)

            map_v = result_dict.loc[cutoff]["MAP"]
            return -map_v

    import optuna

    study = optuna.create_study(direction='minimize')
    study.optimize(TrainUserBased(URM_train, ICM_all, evaluator_validation), n_trials=500)