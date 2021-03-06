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

    from Recommenders.HybridScores.DifferentStructure import ThreeDifferentModelRecommender

    recommender1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train.tocsr())
    recommender2 = IALSRecommender_Hybrid(URM_train.tocsr(), ICM_all)

    ofp = "../all_models/URM_train_1234/"

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
        'KNNweigh' : {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    # Create the enhanced RP3

    recommender_1 = RP3betaRecommender(URM_train)
    recommender_2 = ItemKNNCBFRecommender(URM_train, ICM_weighted)

    model_init(recommender_1, 'RP3beta', CF_opt_hyp['RP3beta'])
    model_init(recommender_2, 'KNNweigh', CF_opt_hyp['KNNweigh'])

    recommender3 = ItemKNNSimilarityHybridRecommenderNormal(URM_train, recommender_1.W_sparse, recommender_2.W_sparse)
    recommender3.fit(0.9546136842105264)

    model_init(recommender1, 'SLIM_pure', CF_opt_hyp['SLIMER'])
    model_init(recommender2, 'IALS_Hyb', CF_opt_hyp['IALSHyb'])



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
            w_1 = trial.suggest_uniform('w_1', 0.5, 1)
            w_2 = trial.suggest_uniform('w_2', 0, 0.8)
            w_3 = trial.suggest_uniform('w_3', 0, 1)
            #omega = trial.suggest_uniform('omega', 0.1, 0.9)
    
            recommender_final = ThreeDifferentModelRecommender(self.URM_train.tocsr(), self.recommender_1, self.recommender_2, self.recommender_3)
            recommender_final.fit(2, w_1, w_2, w_3)
            result_dict, _ = self.evaluator.evaluateRecommender(recommender_final)

            map_v = result_dict.values[0][0]
            if map_v > 0.247: # minimum acceptable map
                return map_v
            else:
                # Calculate the penalty.
                penalty = 0.247 - map_v
                trial.report(penalty, 0)
    
                # Prune trial to notify that the parameters are infeasible.
                raise optuna.TrialPruned()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(Objective(URM_train, recommender1, recommender2, recommender3, evaluator_validation), n_trials=500)
    
    print(study.best_params)



