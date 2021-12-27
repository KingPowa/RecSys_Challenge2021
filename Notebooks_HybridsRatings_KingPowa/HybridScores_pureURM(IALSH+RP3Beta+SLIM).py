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

    import os

    ICM_all = sps.hstack([ICM_genre_all, ICM_subgenre_all, ICM_channel_all])
        
    n_cases = 500  # using 10 as an example
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

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.NonPersonalizedRecommender import GlobalEffects
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender_Hybrid

    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender

    recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train.tocsr())
    recommender3 = IALSRecommender_Hybrid(URM_train.tocsr(), ICM_all)
    recommender2 = RP3betaRecommender(URM_train.tocsr())

    ofp = "../models_temp/ScoresHybrid(IALSH+RP3+SLIMH)"

    recommender_class = ItemKNNScoresHybridMultipleRecommender

    CF_opt_hyp = {
        'TopPop': {},
        'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
        # 'GlobalEffects': {},
        'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    
    if not os.path.exists(ofp):
        os.makedirs(ofp)
        recommender1.fit(**CF_opt_hyp['SLIMER'])
        recommender3.fit(**CF_opt_hyp['IALSHyb'])
        recommender2.fit(**CF_opt_hyp['RP3beta'])
        recommender1.save_model(ofp, 'SLIM_S')
        recommender3.save_model(ofp, 'IALS_H')
        recommender2.save_model(ofp, 'RP3B')
    else:
        recommender1.load_model("C:\Programmi\Programmazione\Resources\GitHub\RecSys_Challenge2021\\result_experiments\SLIM_ONLY_URM\\", "SLIMElasticNetRecommender_best_model")
        recommender3.load_model(ofp, 'IALS_H')
        recommender2.load_model(ofp, 'RP3B')


    # In[28]:


    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {
        "alpha" : Real(low = 0, high = 1, prior = 'uniform'),
        "beta" :Real(low = 0, high = 1, prior = 'uniform'),
        "gamma" :Real(low = 0, high = 1, prior = 'uniform')
    }


    # In[29]:
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                            evaluator_validation=evaluator_validation)


    # In[30]:


    from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, recommender1, recommender2, recommender3],     # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}    # Additiona hyperparameters for the fit function
    )


    # In[ ]:

    ofp_hyp = ofp + "hypersearch/"


    hyperparameterSearch.search(recommender_input_args,
                        hyperparameter_search_space = hyperparameters_range_dictionary,
                        n_cases = n_cases,
                        n_random_starts = n_random_starts,
                        output_folder_path = ofp_hyp, # Where to save the results
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



