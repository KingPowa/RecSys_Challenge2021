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
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender_Hybrid

    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommenderOld

    recommender1 = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_all.tocsr(), ICM_all)
    recommender2 = IALSRecommender_Hybrid(URM_all.tocsr(), ICM_all)
    recommender3 = RP3betaRecommender(URM_all.tocsr())

    ofp = "../models_subs/ScoresHybrid(IALSH+RP3+SLIMH)"

    recommender_class = ItemKNNScoresHybridMultipleRecommenderOld

    CF_opt_hyp = {
        'TopPop': {},
        'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
        # 'GlobalEffects': {},
        'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'hybrid': {'alpha': 0.9999, 'beta': 0.1468889927870315, 'gamma': 0.31314578694563466},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    if not os.path.exists(ofp):
        os.makedirs(ofp)
        recommender1.fit(**CF_opt_hyp['SLIMgensub'])
        recommender2.fit(**CF_opt_hyp['IALSHyb'])
        recommender3.fit(**CF_opt_hyp['RP3beta'])
        recommender1.save_model(ofp, 'SLIM_Sall')
        recommender2.save_model(ofp, 'IALS_Hall')
        recommender3.save_model(ofp, 'RP3Ball')
    else:
        recommender1.load_model(ofp, 'SLIM_Sall')
        recommender2.load_model(ofp, 'IALS_Hall')
        recommender3.load_model(ofp, 'RP3Ball')

    recommender_final = ItemKNNScoresHybridMultipleRecommenderOld(URM_all, recommender1, recommender2, recommender3)
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

