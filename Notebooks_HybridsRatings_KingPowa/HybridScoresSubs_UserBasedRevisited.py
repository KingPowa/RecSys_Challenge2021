import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


# In[17]:

if __name__ == "__main__":
    import Basics.Load as ld

    URM_all, _, _, _, _ = ld.getCOOs()
    ICM_all = ld.getICMall()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)

    groups = ld.load_group()

    # In[18]:


    import os

    # # SLIM Model

    # In[19]:


    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    #URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)


    # In[20]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender # metti il tuo

    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender

    recommender1 = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_all.tocsr(), ICM_all)
    recommender2 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all)

    ofp = "../models_subs/UserWise/"

    CF_opt_hyp = {
        'TopPop': {},
        'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
        # 'GlobalEffects': {},
        'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'SLIMBPRHyb': {"epochs": 1443, "lambda_i": 8.900837513818856e-05, "lambda_j": 1.2615223007492727e-05, "learning_rate": 0.0037706733838839264, "topK": 6181, "random_seed": 1234, "sgd_mode": "sgd"},
        'hybridnew': {'alpha': 0.4313969596049573, 'beta': 0.08471824287830569}, # 0.47...
        'hybrid': {'alpha': 0.9999, 'beta': 0.1468889927870315, 'gamma': 0.31314578694563466},
        'SLIMgroup2' : {'topK': 8534, 'l1_ratio': 0.016193774361742207, 'alpha': 0.040103387689439426, 'workers': 8, 'multiplier_ICM': 0.10963881335030604}, # 0.48026
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    def model_init(recommender, name, args):
        path_name = ofp + name
        if os.path.exists(path_name):
            recommender.load_model(ofp, name)
        else:
            if not os.path.exists(ofp):
                os.makedirs(ofp)
            recommender.fit(**args)
            recommender.save_model(ofp, name)

    model_init(recommender1, 'SLIM_pure_all', CF_opt_hyp['SLIMER'])
    model_init(recommender2, 'SLIM_S_group2_all', CF_opt_hyp['SLIMgroup2'])

    user_in_group_2 = groups[2]

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
        res = None
        if user_id in users_in_group_2:
            res = recommender2.recommend(user_id, K)
        else:
            res = recommender1.recommend(user_id, K)
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

