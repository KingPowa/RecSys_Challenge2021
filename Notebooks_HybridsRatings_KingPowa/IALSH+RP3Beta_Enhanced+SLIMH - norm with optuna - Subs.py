#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys
import cython
import numpy as np
import scipy.sparse as sps
import os
import optuna

sys.path.append('../RecSysRep/')


# In[17]:

if __name__ == "__main__":
    import Basics.Load as ld

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()

    ICM_all = sps.hstack([ICM_genre_all, ICM_subgenre_all, ICM_channel_all])

    ICM_weighted = ld.getICMselected('7')
        
    n_cases = 500  # using 10 as an example

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
    from Recommenders.KNN.ItemKNNSimilarityHybridRecommenderNormal import ItemKNNSimilarityHybridRecommenderNormal
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender_Hybrid

    
    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender
    from Recommenders.HybridScores.DifferentStructure import ThreeDifferentModelRecommender

    recommender2 = IALSRecommender_Hybrid(URM_train.tocsr(), ICM_all)

    ofp = "../all_models/URM_all/"

    def model_init(recommender, name, args):
        path_name = ofp + name + ".zip"
        print(path_name)
        if os.path.exists(path_name):
            print("Model found!")
            recommender.load_model(ofp, name)
        else:
            print("Model does not exists, creating...")
            if not os.path.exists(ofp):
                print("Main folder does not exist, creating...")
                os.makedirs(ofp)
            recommender.fit(**args)
            recommender.save_model(ofp, name)
    


    CF_opt_hyp = {
        'SLIMBPR' : {"epochs": 440, "lambda_i": 0.007773815998802306, "lambda_j": 0.003342522366982381, "learning_rate": 0.010055161410725193, "topK": 4289, "random_seed": 1234, "sgd_mode": "sgd"},
        'hybridnorm' : {'norm':2, 'alpha': 0.5, 'beta': 0.04, 'gamma': 0.1},
        'KNNweigh' : {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    # Create the enhanced RP3

    recommender_1 = RP3betaRecommender(URM_all)
    recommender_2 = ItemKNNCBFRecommender(URM_all, ICM_weighted)

    model_init(recommender_1, 'RP3beta', CF_opt_hyp['RP3beta'])
    model_init(recommender_2, 'KNNweigh', CF_opt_hyp['KNNweigh'])

    recommender3 = ItemKNNSimilarityHybridRecommenderNormal(URM_all, recommender_1.W_sparse, recommender_2.W_sparse)
    recommender3.fit(0.9546136842105264)

    recommender_3 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all) 
    recommender_4 = SLIM_BPR_Cython(URM_all)

    model_init(recommender_3, 'SLIM_pure', CF_opt_hyp['SLIMER'])
    model_init(recommender_4, 'SLIM_BPR', CF_opt_hyp['SLIMBPR'])

    recommender1 = ItemKNNSimilarityHybridRecommenderNormal(URM_all, recommender_3.W_sparse, recommender_4.W_sparse)
    recommender1.fit(0.9876793526315789)

    model_init(recommender2, 'IALS_Hyb', CF_opt_hyp['IALSHyb'])

    recommender_final = ThreeDifferentModelRecommender(URM_all, recommender1, recommender2, recommender3)
    recommender_final.fit(**CF_opt_hyp['hybridnorm'])

    import pandas as pd
    K=10

    user_test_path = '../data/data_target_users_test.csv'
    user_test_dataframe = pd.read_csv(filepath_or_buffer=user_test_path,
                                    sep=",",
                                    dtype={0:int})

    subm_set = user_test_dataframe.to_numpy().T[0]


    subm_res = {"user_id":[], "item_list":[]}

    for user_id in subm_set:
        subm_res["user_id"].append(user_id)
        res = recommender_final.recommend(user_id, K)
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


