#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../RecSysRep/')


# In[2]:


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
ICM_all = ld.getICMall()


# In[3]:


from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=3456)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# In[4]:


import os

ofp = "../models_temp/Similarity_Hybrid/"

models_to_combine_best = {
                            'RP3ICMnew': {'alpha': 1.029719677583138, 'beta': 1.0630164752134375, 'topK': 6964, 'normalize_similarity': True},
                            'RP3ICM' : {"topK": 2550, "alpha": 1.3058102610510849, "beta": 0.5150718337969987, "normalize_similarity": True, "implicit": True},
                            'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
                            'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
                            'SLIMBPR' : {"epochs": 440, "lambda_i": 0.007773815998802306, "lambda_j": 0.003342522366982381, "learning_rate": 0.010055161410725193, "topK": 4289, "random_seed": 1234, "sgd_mode": "sgd"},
                            'SLIMweig': {'l1_ratio': 0.0005247075138160404, 'topK': 4983, 'alpha': 0.06067400905430761, 'workers': 8, 'mw': 2.308619939318322},
                            'SLIMER': {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
                            'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
                            'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
                            'sel7': {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True},
                            'sel9': {"shrink": 5212, "topK": 923, "feature_weighting": "TF-IDF", "normalize": True},
                            'sel3': {'shrink': 2211, 'topK': 188, 'feature_weighting': 'TF-IDF', 'normalize': True},
                            'ICM_all': {"shrink": 5675, "topK": 2310, "feature_weighting": "BM25", "normalize": False},
                            'ER_BPR': {'alpha': 0.58},
                            'TopPop': {},
                            'UserKNN': {"topK": 469, "similarity": "cosine", "shrink": 588, "normalize": True, "feature_weighting": "TF-IDF", "URM_bias": False},
                         }


# In[5]:


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


# In[6]:


from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommenderICM
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender


# In[7]:


from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridTwoRecommender
from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender
from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridTwoRecommender_PRUNE


# In[8]:


recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
recommender2 = SLIM_BPR_Cython(URM_train)
recommender3 = UserKNNCFRecommender(URM_train)
# recommender3 = ItemKNNCBFRecommender(URM_train, ICM_sel_9)

model_init(recommender1, 'SLIMER', models_to_combine_best['SLIMER'])
# model_init(recommender2, 'KNNweigh', models_to_combine_best['icm_weighted'])
model_init(recommender2, 'SLIMBPR', models_to_combine_best['SLIMBPR'])
model_init(recommender3, 'UserKNN', models_to_combine_best['UserKNN'])
# model_init(recommender3, 'TopPop', models_to_combine_best['TopPop'])


# !pip install optuna
import optuna

class Objective(object):
    def __init__(self, URM_train, recommender1, recommender2, recommender3, evaluator):
        # Hold this implementation specific arguments as the fields of the class.
        self.URM_train = URM_train
        self.evaluator = evaluator
        self.recommender_1 = recommender1
        self.recommender_2 = recommender2
        self.recommender_3 = recommender3

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        alpha = trial.suggest_uniform('alpha', 0, 1)
        beta = trial.suggest_uniform('beta', 0, 1)
        gamma = trial.suggest_uniform('gamma', 0, 1)

        recommender_final = ItemKNNScoresHybridMultipleRecommender(self.URM_train.tocsr(), self.recommender_1, self.recommender_2, self.recommender_3)
        recommender_final.fit(alpha, beta, gamma)
        result_dict, _ = self.evaluator.evaluateRecommender(recommender_final)

        map_v = -result_dict.loc[10]['MAP']
        print(map_v)
        if map_v < -0.245: # minimum acceptable map
            return map_v
        else:
            # Calculate the penalty.
            penalty = 0.245 + map_v
            trial.report(penalty, 0)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return map_v


# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(Objective(URM_train, recommender1, recommender2, recommender3, evaluator_validation), n_trials=500)

print(study.best_params)




