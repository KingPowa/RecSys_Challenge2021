
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
    ICM_selected = ld.getICMselected('7')
    ICM_length_all_5bal = ld.getICMlength('5bal')
    ICM_length_all_3bal = ld.getICMlength('3bal')
    ICM_length_all_5km = ld.getICMlength('5km')
    ICM_length_all_3km = ld.getICMlength('5km')


    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed = 1222)

    profile_length = np.ediff1d(csr_matrix(URM_train).indptr)
    sorted_users = np.argsort(profile_length)

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.NonPersonalizedRecommender import GlobalEffects
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython_Hybrid
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommenderICM
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender_Hybrid
    from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython_Hybrid

    MAP_recommender_per_group = {}

    collaborative_recommender_class = {
                                    "P3alpha": P3alphaRecommender,
                                    "RP3beta": RP3betaRecommender,
                                    "SLIMER" : MultiThreadSLIM_SLIMElasticNetRecommender,
                                    "TopPop": TopPop,
                                    # "GlobalEffects": GlobalEffects,
                                    "SLIMBPR": SLIM_BPR_Cython,
                                    "IALS": IALSRecommender
                                    }

    content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender   
                                }

    hybrid_recommender_class = {#"IALSHyb": IALSRecommender_Hybrid,
                                #"SLIMgensub" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                                #"SLIMweig" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                                #"SLIM_BPR_Hyb" : SLIM_BPR_Cython_Hybrid,
                                #"MF_Hyb" : MatrixFactorization_BPR_Cython_Hybrid,
                                "RP3ICM" : RP3betaRecommenderICM,
                                "RP3ICMnew" : RP3betaRecommenderICM
    }

    hybrid_recommender_matrices = {#"IALSHyb": 'icm_all',
                                   #"SLIMgensub": 'icm_genre_subgenre',
                                   #"SLIM_BPR_Hyb" : 'icm_all',
                                   #"MF_Hyb" : 'icm_all',
                                   'RP3ICM' : 'icm_all',
                                   'RP3ICMnew' : 'icm_weighted'
                                   #'RP3ICM_new' : 'icm_weighted',
                                   #"SLIMweig" : 'icm_weighted'
    }
    
    KNN_optimal_hyperparameters = {
        'genre' : {"shrink": 1327, "topK": 622, "feature_weighting": "TF-IDF", "normalize": True},
        'subgenre': {"shrink": 663, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        'channel': {"shrink": 2000, "topK": 382, "feature_weighting": "TF-IDF", "normalize": False},
        # '3bal': {"shrink": 948, "topK": 2750, "feature_weighting": "TF-IDF", "normalize": True},
        # '3km': {"shrink": 2000, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        # '5bal':{"shrink": 1188, "topK": 1156, "feature_weighting": "none", "normalize": True},
        '5km': {"shrink": 1663, "topK": 10, "feature_weighting": "BM25", "normalize": True},
        'icm_weighted': {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True}
    }

    KNN_ICMs = {
        'genre' : ICM_genre_all,
        'subgenre': ICM_subgenre_all,
        'channel': ICM_channel_all,
        # '3bal': ICM_length_all_3bal,
        # '3km': ICM_length_all_3km,
        # '5bal': ICM_length_all_5bal,
        '5km': ICM_length_all_5km,
        'icm_weighted' : ICM_selected
    }

    hybrid_ICMS = {
        'icm_all': hstack([ICM_genre_all, ICM_channel_all, ICM_length_all_5km]),
        #'icm_genre_subgenre': hstack([ICM_genre_all, ICM_subgenre_all]),
        'icm_weighted' : ICM_selected
    }

    CF_optimal_hyperparameters = {
        'TopPop': {},
        'RP3ICMnew': {'alpha': 1.029719677583138, 'beta': 1.0630164752134375, 'topK': 6964, 'normalize_similarity': True},
        'RP3ICM' : {"topK": 2550, "alpha": 1.3058102610510849, "beta": 0.5150718337969987, "normalize_similarity": True, "implicit": True},
        'MF_Hyb': {"sgd_mode": "adam", "epochs": 390, "num_factors": 1, "batch_size": 1, "positive_reg": 0.0014765160794342439, "negative_reg": 1e-05, "learning_rate": 0.0007053433729996733},
        'SLIM_BPR_Hyb' : {"epochs": 1443, "lambda_i": 8.900837513818856e-05, "lambda_j": 1.2615223007492727e-05, "learning_rate": 0.0037706733838839264, "topK": 6181, "random_seed": 1234, "sgd_mode": "sgd"},
        'IALS' : {"num_factors": 29, "epochs": 50, "confidence_scaling": "log", "alpha": 0.001, "epsilon": 0.001, "reg": 0.01},
        'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
        'SLIMBPR' : {"epochs": 440, "lambda_i": 0.007773815998802306, "lambda_j": 0.003342522366982381, "learning_rate": 0.010055161410725193, "topK": 4289, "random_seed": 1234, "sgd_mode": "sgd"},
        'SLIMweig': {'l1_ratio': 0.0005247075138160404, 'topK': 4983, 'alpha': 0.06067400905430761, 'workers': 8, 'mw': 2.308619939318322},
        'SLIMER': {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    recommender_object_dict = {}

    n_groups = 5
    block_size = int(len(profile_length)*(n_groups/100))
    cutoff = 10

    recommenders_group_based = {i : {} for i in range (0, n_groups)}

    print(recommenders_group_based)

    for label, recommender_class in hybrid_recommender_class.items():
        recommender_object = recommender_class(URM_train, hybrid_ICMS[hybrid_recommender_matrices[label]])
        recommender_object.fit(**CF_optimal_hyperparameters[label])
        recommender_object_dict[label] = recommender_object

    for label, recommender_class in collaborative_recommender_class.items():
        recommender_object = recommender_class(URM_train)
        recommender_object.fit(**CF_optimal_hyperparameters[label])
        recommender_object_dict[label] = recommender_object

    for label, recommender_class in content_recommender_class.items():
        for icm_label, value in KNN_ICMs.items():
            recommender_object = recommender_class(URM_train, KNN_ICMs[icm_label])
            recommender_object.fit(**KNN_optimal_hyperparameters[icm_label])
            recommender_object_dict[label + '_' + icm_label] = recommender_object

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
            recommenders_group_based[group_id][label] = result_df.loc[cutoff]["MAP"]
        
        recommenders_group_based[group_id] = {k: v for k, v in sorted(recommenders_group_based[group_id].items(), key=lambda item: item[1])}
        
    print(MAP_recommender_per_group)

    # Get list of best recommenders name per group
    for group_id, rect_dict in recommenders_group_based.items():
        print(f"GROUP {group_id}\n{rect_dict}\n\n")
    
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

    





