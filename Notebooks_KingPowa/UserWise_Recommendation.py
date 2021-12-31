#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import pandas as pd
from scipy.sparse import *

sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
ICM_all = ld.getICMselected()


# In[5]:


import numpy as np
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

#URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=1234)


# In[4]:


from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.NonPersonalizedRecommender import GlobalEffects
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython_Hybrid
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommenderICM
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender_Hybrid
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython_Hybrid

# Define dictionary for all models
collaborative_recommender_class = { "P3alpha": P3alphaRecommender,
                                    "RP3beta": RP3betaRecommender,
                                    "SLIMER" : MultiThreadSLIM_SLIMElasticNetRecommender,
                                    "TopPop": TopPop,
                                    "SLIMBPR": SLIM_BPR_Cython,
                                    "IALS": IALSRecommender
                                  }

content_recommender_class = { "ItemKNNCBF": ItemKNNCBFRecommender   
                            }

hybrid_recommender_class = {"IALSHyb": IALSRecommender_Hybrid,
                            "SLIMgensub" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                            "SLIM_BPR_Hyb" : SLIM_BPR_Cython_Hybrid,
                            "MF_Hyb" : MatrixFactorization_BPR_Cython_Hybrid,
                            "RP3ICM" : RP3betaRecommenderICM
                            }


# In[9]:


profile_length = np.ediff1d(csr_matrix(URM_train).indptr)
sorted_users = np.argsort(profile_length)

n_groups = 10
block_size = int(len(profile_length)*(n_groups/100))
cutoff = 10

evaluators = []

for group_id in range(0, n_groups):
    
    start_pos = group_id*block_size
    end_pos = min((group_id+1)*block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id, 
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))


    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluators.append(EvaluatorHoldout(URM_validation, cutoff_list=[cutoff], ignore_users=users_not_in_group))


# In[24]:


class TrainUserBased(object):
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
                        "sgd_mode": "sgd"}
        
        earlystopping_keywargs = {"validation_every_n": 18,
                        "stop_on_validation": True,
                        "evaluator_object": self.evaluator,
                        "lower_validations_allowed": 12,
                        "validation_metric": "MAP"
                        }
        
        #omega = trial.suggest_uniform('omega', 0.1, 0.9)

        recommender = SLIM_BPR_Cython_Hybrid(URM_train, ICM_all)
        recommender.fit(**search_args, **earlystopping_keywargs)
        result_dict, _ = self.evaluator.evaluateRecommender(recommender)

        map_v = result_dict.loc[cutoff]["MAP"]
        return -map_v


# In[25]:


import optuna

study = optuna.create_study(direction='minimize')
study.optimize(TrainUserBased(URM_train, ICM_all, evaluators[0]), n_trials=500)


# In[16]:





# In[ ]:




