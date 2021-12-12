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

URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[18]:


import os

ICM_all = sps.hstack([ICM_genre_all, ICM_subgenre_all, ICM_channel_all])

output_folder_path = "result_experiments/S-SLIM_BPR/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 10  # using 10 as an example
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# # SLIM Model

# In[19]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

#URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


# In[20]:


evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
#evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[28]:


from skopt.space import Real, Integer, Categorical

hyperparameters_range_dictionary = {
    "epochs": Categorical([2000]),
    "lambda_i": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "lambda_j": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    "learning_rate": Real(low = 4e-4, high = 5e-1, prior = 'log-uniform'),
    "topK": Integer(200, 800),
    "random_seed":Categorical([1234]),
    "sgd_mode": Categorical(["sgd", "adagrad", "adam"])
}

earlystopping_keywargs = {"validation_every_n": 10,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 10,
                          "validation_metric": metric_to_optimize,
                          }


# In[29]:


from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_S_BPR_Cython
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_class = SLIM_S_BPR_Cython

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)


# In[30]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
  
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_genre_all],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs     # Additiona hyperparameters for the fit function
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


from Recommenders.DataIO import DataIO

data_loader = DataIO(folder_path = output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

search_metadata.keys()


# In[ ]:


hyp = search_metadata["hyperparameters_best"]
hyperparameters_df


# In[ ]:


result_on_validation_df = search_metadata["result_on_test_df"]
result_on_validation_df


# In[ ]:


'''
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

recommender_object = MatrixFactorization_FunkSVD_Cython(URM_all.tocsr())

recommender_object.load_model(output_folder_path, 
                              file_name = recommender_object.RECOMMENDER_NAME + "_best_model_last.zip" )
'''


# In[ ]:


recommender = SLIM_S_BPR_Cython(URM_all.tocsr(), ICM_genre_all)
K = 10

recommender.fit(epochs=hyp["epochs"], lambda_i=hyp["lambda_i"], lambda_j=hyp["lambda_j"], 
                learning_rate=hyp["learning_rate"], topK=hyp["topK"], sgd_mode=hyp["sgd_mode"], random_seed=hyp["random_seed"])



# In[ ]:


import pandas as pd

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


# In[ ]:




