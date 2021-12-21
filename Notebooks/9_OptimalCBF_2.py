#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)
ICM_length_all_7km = ld.getICMlength('7km')
ICM_length_all_9km = ld.getICMlength('9km')

# In[3]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[4]:


import os
    
n_cases = 100
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10



from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "shrink": Integer(0, 2000),
    "topK": Integer(10, 3000),
    "feature_weighting": Categorical(["BM25", "TF-IDF", "none"]),
    "normalize": Categorical([True, False]),
}

# In[10]:


from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = ItemKNNCBFRecommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)



from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_length_all_7km],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}     # Additiona hyperparameters for the fit function
)

output_folder_path = "../result_experiments/9_ItemKNN_length_7km/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_length_all_9km],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}     # Additiona hyperparameters for the fit function
)

output_folder_path = "../result_experiments/9_ItemKNN_length_9km/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )




