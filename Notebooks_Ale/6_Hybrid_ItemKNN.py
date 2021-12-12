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

output_folder_path = "../result_experiments/6_Hybrid_ItemKNN"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 300  # using 10 as an example
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# In[5]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "shrink": Integer(10, 500),
    "topK": Integer(200, 1000),
    "feature_weighting": Categorical(["none", "TF-IDF"]),
    "normalize": Categorical([True, False]),
    "ICM_weight": Integer(5, 500),
}


# In[6]:


from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = ItemKNN_CFCBF_Hybrid_Recommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[7]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_genre_all],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}, # Additiona hyperparameters for the fit function
)


# In[ ]:


if __name__ == '__main__':
    hyperparameterSearch.search(recommender_input_args,
                           hyperparameter_search_space = hyperparameters_range_dictionary,
                           n_cases = n_cases,
                           n_random_starts = n_random_starts,
                           output_folder_path = output_folder_path,
                           output_file_name_root = recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize = metric_to_optimize,
                           cutoff_to_optimize = cutoff_to_optimize,
                          )


# In[ ]:


'''
recommender = MultiThreadSLIM_SLIMElasticNetRecommender(R_F)
recommender.load_model(output_folder_path, file_name = MultiThreadSLIM_SLIMElasticNetRecommender.RECOMMENDER_NAME + "_best_model.zip" )

import pandas as pd
at = 10

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

'''


# In[ ]:




