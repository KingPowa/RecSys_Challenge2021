#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[3]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[4]:


import os

output_folder_path = "../result_experiments/3_FunkSVD_py"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 50  # using 10 as an example
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# In[5]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "epochs": Categorical([500]), # because I want one specific value which is the max
    "num_factors": Integer(10, 200),
    "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
    "batch_size": Categorical([2, 8, 16, 32, 64, 128, 256, 512, 1024]),
    "item_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "user_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
}


# In[6]:


earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 5,
                          "validation_metric": metric_to_optimize,
                          }


# In[7]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs     # Additiona hyperparameters for the fit function
)


# In[10]:


from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = MatrixFactorization_FunkSVD_Cython

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[ ]:


hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "last",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


# In[ ]:




