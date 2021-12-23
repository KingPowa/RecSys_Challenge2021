#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()


# In[3]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[4]:


import os

output_folder_path = "../result_experiments/GlobalEffects/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 200
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# In[5]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "lambda_item": Real(0, 5),
    "lambda_user": Real(0, 5),
}



from Recommenders.NonPersonalizedRecommender import GlobalEffects
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = GlobalEffects

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[7]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {} 
)


# In[ ]:


hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


# In[ ]:




