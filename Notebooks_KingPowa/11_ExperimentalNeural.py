#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


# In[17]:


if __name__ == "__main__":
    import Basics.Load as ld

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


    # In[18]:


    import os

    output_folder_path = "../result_experiments/Neural/"

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

    #URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


    # In[20]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    # In[28]:


    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {
                "epochs": Categorical([300]),
                "learning_rate": Real(low=1e-6, high=1e-2, prior="log-uniform"),
                "l2_reg": Real(low=1e-6, high=1e-2, prior="log-uniform"),
                "dropout": Real(low=0., high=0.8, prior="uniform"),
                "total_anneal_steps": Integer(100000, 600000),
                "anneal_cap": Real(low=0., high=0.6, prior="uniform"),
                "batch_size": Categorical([128, 256, 512, 1024]),

                "encoding_size": Integer(1, min(512, n_items)),
                "next_layer_size_multiplier": Integer(2, 10),
                "max_n_hidden_layers": Integer(1, 4),
                # Reduce max_layer_size if estimated last layer weights size exceeds 2 GB
                "max_layer_size": Categorical([min(5*1e3, int(2*1e9*8/64/n_items))]),
            }

    earlystopping_keywargs = {"validation_every_n": 5,
                            "stop_on_validation": True,
                            "evaluator_object": evaluator_validation,
                            "lower_validations_allowed": 5,
                            "validation_metric": metric_to_optimize,
                            }


    # In[29]:


    from Recommenders.FactorizationMachines.MultVAERecommender import MultVAERecommender_OptimizerMask
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = MultVAERecommender_OptimizerMask

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                            evaluator_validation=evaluator_validation)


    # In[30]:


    from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
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
    hyp


    # In[ ]:


    result_on_validation_df = search_metadata["result_on_test_df"]
    result_on_validation_df