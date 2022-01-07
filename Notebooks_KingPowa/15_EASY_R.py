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

URM_all, ICM1, ICM2, ICM3, _ = ld.getCOOs()
ICML = ld.getICMlength('5km')
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[18]:


import os

ICM_all = sps.hstack([ICM1, ICM2, ICM3, ICML])

output_folder_path = "result_experiments/EASE_R/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 200  # using 10 as an example
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# # SLIM Model

# In[19]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

#URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=6666)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


# In[28]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
        "topK": Integer(2000, 9000),
        "normalize_matrix": Categorical([False]),
        "l2_norm": Real(low = 1e3, high = 1e5, prior = 'log-uniform'),
}



# In[29]:


from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = EASE_R_Recommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[30]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}     # Additiona hyperparameters for the fit function
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


# In[ ]:


from Recommenders.DataIO import DataIO

data_loader = DataIO(folder_path = output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

search_metadata.keys()


# In[ ]:


hyp = search_metadata["hyperparameters_best"]
hyp


# In[ ]:


result_on_validation_df = search_metadata["result_on_test_df"]
result_on_validation_df