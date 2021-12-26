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

    ICM_all = sps.hstack([ICM_genre_all, ICM_subgenre_all, ICM_channel_all])

    output_folder_path = "result_experiments/S-SLIM_BPR/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    n_cases = 100  # using 10 as an example
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
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender

    recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all.tocsr())
    recommender2 = IALSRecommender(URM_all.tocsr())
    recommender3 = RP3betaRecommender(URM_all.tocsr())

    ofp = "temp/"

    recommender_class = ItemKNNScoresHybridMultipleRecommender

    CF_opt_hyp = {
            'TopPop': {},
            'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
            # 'GlobalEffects': {},
            'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
            'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
            'RP3beta': {'topK': 100, 'alpha': 1.0042367418834082, 'beta': 0.6027649914044608, 'normalize_similarity': True, 'implicit': True},
            'hybrid': {'alpha': 0.7984294523480804, 'beta': 0.048746451899799086, 'gamma': 0.13381674771504035}
        }


    recommender1.fit(**CF_opt_hyp['SLIMER'])
    recommender2.fit(**CF_opt_hyp['IALS'])
    recommender3.fit(**CF_opt_hyp['RP3beta'])

    recommender_final = ItemKNNScoresHybridMultipleRecommender(URM_all, recommender1, recommender2, recommender3)
    recommender_final.fit(**CF_opt_hyp['hybrid'])

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

