    #!/usr/bin/env python
    # coding: utf-8

    # In[16]:



import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


    # In[17]:

if __name__ == '__main__':

    import Basics.Load as ld

    URM_all, _, _, _, _= ld.getCOOs()

    ICM_all = ld.getICMall()

    # In[18]:


    import os

    output_folder_path = "result_experiments/SLIM_GENRE_CHANNEL_LENGTH3BAL/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    n_cases = 50  # using 10 as an example
    n_random_starts = int(n_cases*0.3)
    metric_to_optimize = "MAP"   
    cutoff_to_optimize = 10


    # # SLIM Model

    # In[19]:


    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=9284)


    # In[20]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    # In[28]:


    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {
        "topK": Categorical([18000]),
        "l1_ratio": Real(low=1e-5, high=1.0, prior='log-uniform'),
        "alpha": Real(low=1e-3, high=1.0, prior='uniform'),
        "mw": Real(low=1, high=3, prior='uniform'),
        "workers": Categorical([8])
    }

    '''
    earlystopping_keywargs = {"validation_every_n": 10,
                            "stop_on_validation": True,
                            "evaluator_object": evaluator_validation,
                            "lower_validations_allowed": 10,
                            "validation_metric": metric_to_optimize,
                            }
    '''

    # In[29]:


    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = MultiThreadSLIM_SLIM_S_ElasticNetRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                            evaluator_validation=evaluator_validation)


    # In[30]:


    from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_all],     # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}    # Additiona hyperparameters for the fit function
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



    