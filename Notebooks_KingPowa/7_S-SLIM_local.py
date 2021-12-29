#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


# In[17]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[18]:


import os

ICM_all = ld.getICMselected('7')

output_folder_path = "result_experiments/S-SLIM_BPR_weightedopt/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


# # SLIM Model

# In[19]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

#URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=1224)


# In[20]:


evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
#evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[28]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "epochs": Categorical([3000]),
    "lambda_i": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "lambda_j": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "learning_rate": Real(low = 4e-4, high = 1e-1, prior = 'log-uniform'),
    "w"
    "topK": Integer(800, 8000),
    "random_seed":Categorical([1224]),
    "sgd_mode": Categorical(["sgd", "adagrad", "adam"])
}

earlystopping_keywargs = {"validation_every_n": 18,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 15,
                          "validation_metric": "MAP",
                          }


# In[29]:


from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython_HybridW

class Objd(object):
        def __init__(self, URM_train, ICM_all, evaluator):
            # Hold this implementation specific arguments as the fields of the class.
            self.URM_train = URM_train
            self.evaluator = evaluator

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            
            search_args = {"epochs": 3000, 
                        "lambda_i": trial.suggest_loguniform('lambda_i', 1e-5, 1e-2), 
                        "lambda_j": trial.suggest_loguniform('lambda_j', 1e-5, 1e-2), 
                        "learning_rate": trial.suggest_uniform('learning_rate', 4e-4, 1e-1), 
                        "topK": trial.suggest_int('topK', 2000, 8000), 
                        "random_seed": 1234,
                        "mw": trial.suggest_uniform("mw", 0, 1), 
                        "sgd_mode": "sgd"}

            earlystopping_keywargs = {"validation_every_n": 18,
                        "stop_on_validation": True,
                        "evaluator_object": self.evaluator,
                        "lower_validations_allowed": 12,
                        "validation_metric": "MAP"
                        }

            recommender = SLIM_BPR_Cython_HybridW(self.URM_train, self.ICM_all)
            recommender.fit(**search_args, **earlystopping_keywargs)
            result_dict, _ = self.evaluator.evaluateRecommender(recommender)

            map_v = result_dict.loc[10]["MAP"]
            return -map_v

import optuna

study = optuna.create_study(direction='minimize')
study.optimize(Objd(URM_train, ICM_all, evaluator_validation), n_trials=500)