#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
# URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, _ = ld.getCOOs()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[3]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed = 8888)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[4]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "num_factors": Integer(2, 200),
}


# In[5]:


from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = PureSVDRecommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[6]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)


# In[7]:


import os

output_folder_path = "../result_experiments/2_PureSVD/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 50  # using 10 as an example
n_random_starts = int(n_cases*0.4)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# In[9]:


hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


# In[7]:

'''
import pandas as pd
recommender = PureSVDRecommender(URM_all) # <-----
at = 10

recommender.fit(num_factors = 32)

user_test_path = '../data/data_target_users_test.csv'
user_test_dataframe = pd.read_csv(filepath_or_buffer=user_test_path,
sep=",",
dtype={0:int})

subm_set = user_test_dataframe.to_numpy().T[0]


subm_res = {"user_id":[], "item_list":[]}

for user_id in subm_set:
	subm_res["user_id"].append(user_id)
	res = recommender.recommend(user_id, cutoff=at)
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


# In[ ]:
'''



