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

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    ICM_length_all = ld.getICMlength()

    ICM_all = sps.hstack((ICM_genre_all, ICM_channel_all, ICM_length_all))


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

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


    # In[20]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    # In[28]:


    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {
        "topK": Integer(5, 2000),
        "l1_ratio": Real(low=1e-5, high=1.0, prior='log-uniform'),
        "alpha": Real(low=1e-3, high=1.0, prior='uniform'),
        "mw": Real(low=1, high=100, prior='uniform'),
        "workers": Categorical([2])
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


    # In[ ]:


    from Recommenders.DataIO import DataIO

    data_loader = DataIO(folder_path = output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    search_metadata.keys()


    # In[ ]:


    hyp = search_metadata["hyperparameters_best"]
    hyp

    # In[ ]:


    recommender = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_all.tocsr(), ICM_all)
    K = 10

    recommender.fit(hyp)



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


    submission.to_csv('../subs/submission {:%Y_%m_%d %H_%M_%S}_SLIMURM.csv'.format(now), index=False)


    # In[ ]:




