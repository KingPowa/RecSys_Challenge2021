
import sys
sys.path.append('../RecSysRep/')
import Basics.Load as ld
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import numpy as np
from scipy.sparse import *
import os
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
 
if __name__ == '__main__':

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)
    ICM_length_all_5bal = ld.getICMlength('5bal')
    ICM_length_all_3bal = ld.getICMlength('3bal')
    ICM_length_all_5km = ld.getICMlength('5km')
    ICM_length_all_3km = ld.getICMlength('5km')


    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed = 1234)

    profile_length = np.ediff1d(csr_matrix(URM_train).indptr)
    sorted_users = np.argsort(profile_length)

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.NonPersonalizedRecommender import GlobalEffects
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

    MAP_recommender_per_group = {}

    collaborative_recommender_class = {
                                    "P3alpha": P3alphaRecommender,
                                    "RP3beta": RP3betaRecommender,
                                    "SLIMER" : MultiThreadSLIM_SLIMElasticNetRecommender,
                                    "TopPop": TopPop,
                                    # "GlobalEffects": GlobalEffects,
                                    # "SLIMBPR": SLIM_BPR_Cython,
                                    # "IALS": IALSRecommender
                                    }

    content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender,
                                }
    
    KNN_optimal_hyperparameters = {
        'genre' : {"shrink": 1327, "topK": 622, "feature_weighting": "TF-IDF", "normalize": True},
        'subgenre': {"shrink": 663, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        'channel': {"shrink": 2000, "topK": 382, "feature_weighting": "TF-IDF", "normalize": False},
        '3bal': {"shrink": 948, "topK": 2750, "feature_weighting": "TF-IDF", "normalize": True},
        '3km': {"shrink": 2000, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        '5bal':{"shrink": 1188, "topK": 1156, "feature_weighting": "none", "normalize": True},
        '5km': {"shrink": 1663, "topK": 10, "feature_weighting": "BM25", "normalize": True},
    }

    KNN_ICMs = {
        'genre' : ICM_genre_all,
        'subgenre': ICM_subgenre_all,
        'channel': ICM_channel_all,
        '3bal': ICM_length_all_3bal,
        '3km': ICM_length_all_3km,
        '5bal': ICM_length_all_5bal,
        '5km': ICM_length_all_5km,
    }

    CF_optimal_hyperparameters = {
        'TopPop': {},
        # 'GlobalEffects': {},
        'SLIMER':  {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {'topK': 100, 'alpha': 1.0042367418834082, 'beta': 0.6027649914044608, 'normalize_similarity': True, 'implicit': True},
    }

    recommender_object_dict = {}

    for label, recommender_class in collaborative_recommender_class.items():
        recommender_object = recommender_class(URM_train)
        recommender_object.fit(**CF_optimal_hyperparameters[label])
        recommender_object_dict[label] = recommender_object

    for label, recommender_class in content_recommender_class.items():
        for icm_label, value in KNN_ICMs.items():
            recommender_object = recommender_class(URM_train, KNN_ICMs[icm_label])
            recommender_object.fit(**KNN_optimal_hyperparameters[icm_label])
            recommender_object_dict[label + '_' + icm_label] = recommender_object
    
    n_groups = 10
    block_size = int(len(profile_length)*(n_groups/100))
    cutoff = 10

    for group_id in range(0, n_groups):
    
        start_pos = group_id*block_size
        end_pos = min((group_id+1)*block_size, len(profile_length))
        
        users_in_group = sorted_users[start_pos:end_pos]
        
        users_in_group_p_len = profile_length[users_in_group]
        
        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id, 
            users_in_group.shape[0],
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))
        
        
        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]
        
        evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff], ignore_users=users_not_in_group)
        
        for label, recommender in recommender_object_dict.items():
            result_df, _ = evaluator_test.evaluateRecommender(recommender)
            if label in MAP_recommender_per_group:
                MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
            else:
                MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]
        
    print(MAP_recommender_per_group)
    
    import matplotlib.pyplot as plt

    _ = plt.figure(figsize=(16, 9))
    for label, recommender in recommender_object_dict.items():
        results = MAP_recommender_per_group[label]
        plt.scatter(x=np.arange(0,len(results)), y=results, label=label)
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.legend()
    plt.show()
    
    plt.savefig('userwise.png')

    





