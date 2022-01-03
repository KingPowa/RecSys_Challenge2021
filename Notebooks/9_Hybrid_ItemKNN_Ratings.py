#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../RecSysRep/')


# In[3]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
ICM_sel_7 = ld.getICMselected('7')
ICM_sel_9 = ld.getICMselected('9')
ICM_all = ld.getICMall()
# URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)


# In[4]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=8888)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[5]:


import os

ofp = "../models_temp/Similarity_Hybrid/"

models_to_combine_best = {
                            'RP3ICMnew': {'alpha': 1.029719677583138, 'beta': 1.0630164752134375, 'topK': 6964, 'normalize_similarity': True},
                            'RP3ICM' : {"topK": 2550, "alpha": 1.3058102610510849, "beta": 0.5150718337969987, "normalize_similarity": True, "implicit": True},
                            'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
                            'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
                            'SLIMBPR' : {"epochs": 440, "lambda_i": 0.007773815998802306, "lambda_j": 0.003342522366982381, "learning_rate": 0.010055161410725193, "topK": 4289, "random_seed": 1234, "sgd_mode": "sgd"},
                            'SLIMweig': {'l1_ratio': 0.0005247075138160404, 'topK': 4983, 'alpha': 0.06067400905430761, 'workers': 8, 'mw': 2.308619939318322},
                            'SLIMER': {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 2},
                            'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
                            'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
                            'sel7': {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True},
                            'sel9': {"shrink": 5212, "topK": 923, "feature_weighting": "TF-IDF", "normalize": True},
                            'sel3': {'shrink': 2211, 'topK': 188, 'feature_weighting': 'TF-IDF', 'normalize': True},
                            'ER_BPR': {'alpha': 0.58},
                            'ICM_all': {"shrink": 5675, "topK": 2310, "feature_weighting": "BM25", "normalize": False},
                         }


# In[6]:


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


# In[7]:


from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommenderICM
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender


# In[8]:


from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridTwoRecommender
from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender
from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridOfHybridRecommender


# In[ ]:


recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
recommender3 = SLIM_BPR_Cython(URM_train)
recommender2 = ItemKNNCBFRecommender(URM_train, ICM_all)

recommenderHybrid = ItemKNNScoresHybridTwoRecommender(URM_train, recommender1, recommender2)

model_init(recommender1, 'SLIMER', models_to_combine_best['SLIMER'])
# model_init(recommender2, 'KNNweigh', models_to_combine_best['icm_weighted'])
model_init(recommender3, 'SLIMBPR_8888', models_to_combine_best['SLIMBPR'])
model_init(recommender2, 'KNN_all', models_to_combine_best['ICM_all'])


# In[12]:



recommenderHybrid.fit(alpha = 0.86)


# In[ ]:


# def rank_models(evaluator, recommenders):
#     for r in recommenders:
#         r_d, _ = evaluator.evaluateRecommender(r)
#         print(r.RECOMMENDER_NAME, r_d.loc[10]['MAP'])


# # In[ ]:


# rank_models(evaluator_validation, [recommender1, recommender3])



# In[17]:


import numpy as np

# TWO RECOMMENDERS

def test_percentage(recommender_1, recommender_2, evaluator, step):
    recommender = ItemKNNScoresHybridOfHybridRecommender(URM_train, recommender_1, recommender_2)
    results = []
    alp_space = np.linspace(0.3, 0.95, step, True)
    for alp in alp_space:
        recommender.fit(alpha = alp)
        r_d, _ = evaluator.evaluateRecommender(recommender)
        print(alp, ":", r_d.loc[10]['MAP'])
        results.append(r_d.loc[10]['MAP'])
    
    return alp_space, results


# In[18]:


alp_space, results = test_percentage(recommenderHybrid, recommender3, evaluator_validation, 5)



# # In[ ]:


# import numpy as np

# # THREE RECOMMENDERS

# def test_percentage(recommender_1, recommender_2, recommender_3, evaluator, step):
#     recommender = ItemKNNScoresHybridMultipleRecommender(URM_train, recommender_1, recommender_2, recommender_3)
#     results = []
#     alp_space = np.linspace(12, 18, step, True)
#     for alp in alp_space:
#         recommender.fit(alpha = alp, beta = 5, gamma = 1)
#         r_d, _ = evaluator.evaluateRecommender(recommender)
#         print(alp, ":", r_d.loc[10]['MAP'])
#         results.append(r_d.loc[10]['MAP'])
    
#     return alp_space, results


# # In[ ]:


# alp_space, results = test_percentage(recommender1, recommender2, recommender3, evaluator_validation, 5)


# # In[ ]:


# import matplotlib.pyplot as plt

# _ = plt.figure(figsize=(16, 9))
# plt.plot(alp_space,results, label='MAP variability')
# plt.ylabel('MAP')
# plt.xlabel('User Group')
# plt.legend()
# plt.show()

# plt.savefig('userwise.png')


# * # SLIMER + KNNweight: 0.2499 with alpha = 0.95
# * # SLIMER + SLIMBPR: 0.2507 with alpha = 0.58
# * # (SLIMER + SLIMBPR) + KNN_selected_3: 0.2509 with alpha 0.96
# * # SLIMER + ICM_all: 0.249 with alpha = 0.86

# In[ ]:



# In[ ]:

'''

recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all)
# recommender1.fit(**models_to_combine_best['SLIMER'])
# recommender1.save_model(ofp + 'SLIMER_ALL/', 'SLIMER_ALL')
recommender1.load_model(ofp + 'SLIMER_ALL/', 'SLIMER_ALL')

recommender2 = ItemKNNCBFRecommender(URM_all, ICM_all)
# recommender2.fit(**models_to_combine_best['ICM_all'])
# recommender2.save_model(ofp + 'ItemKNN_ALL/', 'ItemKNN_ALL')
recommender2.load_model(ofp + 'ItemKNN_ALL/', 'ItemKNN_ALL')

recommenderHybrid = ItemKNNScoresHybridTwoRecommender(URM_all, recommender1, recommender2)
recommenderHybrid.fit(alpha = 0.86)

recommender = recommenderHybrid

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


submission.to_csv('./submission {:%Y_%m_%d %H_%M_%S}.csv'.format(now), index=False)


# In[ ]:




from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
CFrecommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
CFrecommender.load_model('../result_experiments/BEST_SLIM_PURECF/', file_name = CFrecommender.RECOMMENDER_NAME + "_my_own_save.zip")

from Recommenders.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
FW_recommender = CFW_D_Similarity_Linalg(URM_train, ICM_stacked, CFrecommender.W_sparse)
FW_recommender.load_model('../result_experiments/BEST_FW/', file_name = FW_recommender.RECOMMENDER_NAME + "_my_own_save.zip")


# In[ ]:


import scipy.sparse as sps
ICM_length_all_5km = ld.getICMlength('5km')
URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, _ = ld.getCOOs()
ICM_stacked = sps.hstack((ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_length_all_5km))


# In[ ]:


import pandas as pd
argsort_features = np.argsort(-FW_recommender.D_best)

selection_quota = 0.1
        
n_to_select = int(selection_quota*len(argsort_features))
selected_features = argsort_features[:n_to_select]
        
ICM_selected = ICM_stacked.todense()[:,selected_features]
pd.DataFrame(ICM_selected).to_csv('ICM_selected_1.csv', index=False, header=True)


# In[ ]:



'''