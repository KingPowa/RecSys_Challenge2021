
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


    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed = 9090)

    ICM_all = ld.getICMall()

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.NonPersonalizedRecommender import GlobalEffects
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
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
    from Recommenders.Neural.MultVAERecommender import MultVAERecommender
    from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
    from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

    from Recommenders.KNN.ItemKNNScoresHybridMultipleRecommender import ItemKNNScoresHybridMultipleRecommender
    from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
    from Recommenders.KNN.ItemKNNSimilarityHybridRecommenderNormal import ItemKNNSimilarityHybridRecommenderNormal
    from Recommenders.HybridScores.DifferentStructure import ThreeDifferentModelRecommender
    from Recommenders.HybridScores.DifferentStructure import TwoDifferentModelRecommender

    MAP_recommender_per_group = {}

    collaborative_recommender_class = {
                                    #"P3alpha": P3alphaRecommender,
                                    #"RP3beta": RP3betaRecommender,
                                    "SLIMER" : MultiThreadSLIM_SLIMElasticNetRecommender,
                                    #"TopPop": TopPop,
                                    # "GlobalEffects": GlobalEffects,
                                    "SLIMBPR": SLIM_BPR_Cython,
                                    #"IALS": IALSRecommender,
                                    'UserKNN': UserKNNCFRecommender,
                                    'EASE': EASE_R_Recommender,
                                    'Neural': MultVAERecommender_OptimizerMask
                                    }

    content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender   
                                }

    hybrid_recommender_class = {#"IALSHyb": IALSRecommender_Hybrid,
                                "SLIMgensub" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                                "SLIMweig" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                                "SLIMall" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender,
                                "SLIMall2" : MultiThreadSLIM_SLIM_S_ElasticNetRecommender
                                #"SLIM_BPR_Hyb" : SLIM_BPR_Cython_Hybrid,
                                #"MF_Hyb" : MatrixFactorization_BPR_Cython_Hybrid,
                                #"RP3ICM" : RP3betaRecommenderICM,
                                #"RP3ICMnew" : RP3betaRecommenderICM
    }

    hybrid_recommender_matrices = {"IALSHyb": 'icm_all',
                                   "SLIMgensub": 'icm_genre_subgenre',
                                   "SLIM_BPR_Hyb" : 'icm_all',
                                   #"MF_Hyb" : 'icm_all',
                                   'RP3ICM' : 'icm_all',
                                   'RP3ICMnew' : 'icm_weighted',
                                   #'RP3ICM_new' : 'icm_weighted',
                                   "SLIMweig" : 'icm_weighted',
                                   "SLIMall" : 'icm_all',
                                   "SLIMall2" : 'icm_all'
    }

    similarity_hybrid_recommender = { #'RP3+KNN_S' : ['RP3beta', 'ItemKNNCBF_icm_weighted'],
                                      #'SLIM_mixed_S' : ['SLIMER', 'SLIMBPR'],
                                      #'SLIM+KNN_S' : ['SLIMER', 'ItemKNNCBF_icm_weighted']
                                    }

    score_hybrid_recommender = { #'SLIM_mixed' : ['SLIMER', 'SLIMBPR'],
                                 'SLIMall+EASE' : ['SLIMall', 'EASE'],
                                 'SLIMall2+EASE' : ['SLIMall2', 'EASE']
     }

    similarity_hybrid_recommender2 = { #'RP3+KNN+SLIM_mixed_S' : ['RP3+KNN_S', 'SLIM_mixed_S'],
                                       #'SLIM+KNN+SLIM_BPR_S' : ['SLIM+KNN_S', 'SLIMBPR']
    }

    
    KNN_optimal_hyperparameters = {
        'genre' : {"shrink": 1327, "topK": 622, "feature_weighting": "TF-IDF", "normalize": True},
        'subgenre': {"shrink": 663, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        'channel': {"shrink": 2000, "topK": 382, "feature_weighting": "TF-IDF", "normalize": False},
        # '3bal': {"shrink": 948, "topK": 2750, "feature_weighting": "TF-IDF", "normalize": True},
        # '3km': {"shrink": 2000, "topK": 10, "feature_weighting": "BM25", "normalize": False},
        # '5bal':{"shrink": 1188, "topK": 1156, "feature_weighting": "none", "normalize": True},
        '5km': {"shrink": 1663, "topK": 10, "feature_weighting": "BM25", "normalize": True},
        'icm_weighted': {"shrink": 4000, "topK": 985, "feature_weighting": "TF-IDF", "normalize": True},
        'icm_all': {"shrink": 5675, "topK": 2310, "feature_weighting": "BM25", "normalize": False}
    }

    KNN_ICMs = {
        #'genre' : ICM_genre_all,
        #'subgenre': ICM_subgenre_all,
        #'channel': ICM_channel_all,
        # '3bal': ICM_length_all_3bal,
        # '3km': ICM_length_all_3km,
        # '5bal': ICM_length_all_5bal,
        #'5km': ICM_length_all_5km,
        'icm_weighted' : ICM_selected,
        'icm_all': ICM_all
    }

    hybrid_ICMS = {
        'icm_all': ICM_all,
        'icm_genre_subgenre': hstack([ICM_genre_all, ICM_subgenre_all]),
        'icm_weighted' : ICM_selected
    }

    CF_optimal_hyperparameters = {
        'TopPop': {},
        'RP3+KNN_S' : {'alpha' : 0.955},
        'SLIM_mixed_S' : {'alpha' : 0.978},
        'EASE': {"topK": None, "normalize_matrix": False, "l2_norm": 2712.02456},
        'Neural': {"epochs": 295, "learning_rate": 9.598708662230564e-05, "l2_reg": 1.4122381520119457e-05, "dropout": 0.5759445339123964, "total_anneal_steps": 147387, "anneal_cap": 0.4376000433422989, "batch_size": 256, "encoding_size": 185, "next_layer_size_multiplier": 6, "max_n_hidden_layers": 1, "max_layer_size": 5000.0},
        'RP3+KNN+SLIM_mixed_S': {'alpha': 0.7603833333333334},
        'SLIMall+EASE': {'alpha': 0.725},
        'SLIMall2+EASE' : {'alpha': 0.725},
        'SLIM+KNN_S' : {'alpha': 0.9210344},
        'SLIM+KNN+SLIM_BPR_S' : {'alpha': 0.98857378},
        'SLIM_mixed' : {'norm': 2, 'alpha': 0.58},
        'SLIMall' : {'l1_ratio': 0.0001043005302985496, 'topK': 1135, 'alpha': 0.06322495726710943, 'workers': 8, 'mw': 1.2937257828415842},
        'SLIMall2' : {'topK': 4997, 'l1_ratio': 0.00036065147205858187, 'alpha': 0.03203200564199538, 'mw': 1.0785132020354402, 'workers': 8},
        'UserKNN' : {'topK': 448, 'similarity': 'cosine', 'shrink': 756, 'normalize': True, 'feature_weighting': 'TF-IDF', 'URM_bias': True},
        'RP3ICMnew': {'alpha': 1.029719677583138, 'beta': 1.0630164752134375, 'topK': 6964, 'normalize_similarity': True},
        'RP3ICM' : {"topK": 2550, "alpha": 1.3058102610510849, "beta": 0.5150718337969987, "normalize_similarity": True, "implicit": True},
        'MF_Hyb': {"sgd_mode": "adam", "epochs": 390, "num_factors": 1, "batch_size": 1, "positive_reg": 0.0014765160794342439, "negative_reg": 1e-05, "learning_rate": 0.0007053433729996733},
        'SLIM_BPR_Hyb' : {"epochs": 1443, "lambda_i": 8.900837513818856e-05, "lambda_j": 1.2615223007492727e-05, "learning_rate": 0.0037706733838839264, "topK": 6181, "random_seed": 1234, "sgd_mode": "sgd"},
        'IALS' : {'num_factors': 34, 'epochs': 599, 'confidence_scaling': 'linear', 'alpha': 0.003519435539271083, 'epsilon': 0.09222402080721787, 'reg': 2.4127708108457617e-05},
        'SLIMgensub': {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1},
        'SLIMBPR' : {"epochs": 440, "lambda_i": 0.007773815998802306, "lambda_j": 0.003342522366982381, "learning_rate": 0.010055161410725193, "topK": 4289, "random_seed": 1234, "sgd_mode": "sgd"},
        'SLIMweig': {'l1_ratio': 0.0005247075138160404, 'topK': 4983, 'alpha': 0.06067400905430761, 'workers': 8, 'mw': 2.308619939318322},
        'SLIMER': {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8},
        'P3alpha': {'topK': 4834, 'alpha': 1.764994849187595, 'normalize_similarity': True, 'implicit': True},
        'RP3beta': {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True},
        'IALSHyb': {"num_factors": 28, "epochs": 10, "confidence_scaling": "linear", "alpha": 0.43657990940994623, "epsilon": 0.35472063248578317, "reg": 0.0001698292271931609, "mw": 0.06122362507952762}
    }

    recommender_object_dict = {}

    n_groups = 3

    recommenders_group_based = {i : {} for i in range (0, n_groups)}

    print(recommenders_group_based)

    ofp = "../userwise_model/"

    for label, recommender_class in hybrid_recommender_class.items():
        recommender_object = recommender_class(URM_train, hybrid_ICMS[hybrid_recommender_matrices[label]])
        if os.path.exists(ofp + label + '.zip'):
            print("Model found!")
            recommender_object.load_model(ofp, label)
        else:
            print("Model not found... Fitting now!")
            recommender_object.fit(**CF_optimal_hyperparameters[label])
            recommender_object.save_model(ofp, label)
        recommender_object_dict[label] = recommender_object

    for label, recommender_class in collaborative_recommender_class.items():
        recommender_object = recommender_class(URM_train)
        if os.path.exists(ofp + label + '.zip'):
            print("Model found!")
            recommender_object.load_model(ofp, label)
        else:
            print("Model not found... Fitting now!")
            recommender_object.fit(**CF_optimal_hyperparameters[label])
            recommender_object.save_model(ofp, label)
        recommender_object_dict[label] = recommender_object

    for label, recommender_class in content_recommender_class.items():
        for icm_label, value in KNN_ICMs.items():
            recommender_object = recommender_class(URM_train, KNN_ICMs[icm_label])
            if os.path.exists(ofp + label + '_' + icm_label + '.zip'):
                print("Model found!")
                recommender_object.load_model(ofp, label + '_' + icm_label)
            else:
                print("Model not found... Fitting now!")
                recommender_object.fit(**KNN_optimal_hyperparameters[icm_label])
                recommender_object.save_model(ofp, label + '_' + icm_label)
            recommender_object_dict[label + '_' + icm_label] = recommender_object

    # for label, recommender_class in similarity_hybrid_recommender.items():
    #     recommender1 = recommender_object_dict[recommender_class[0]]
    #     recommender2 = recommender_object_dict[recommender_class[1]]
    #     recommender_object = ItemKNNSimilarityHybridRecommenderNormal(URM_train, recommender1.W_sparse, recommender2.W_sparse)
    #     recommender_object.fit(**CF_optimal_hyperparameters[label])
    #     recommender_object_dict[label] = recommender_object
    
    # for label, recommender_class in similarity_hybrid_recommender2.items():
    #     recommender1 = recommender_object_dict[recommender_class[0]]
    #     recommender2 = recommender_object_dict[recommender_class[1]]
    #     recommender_object = ItemKNNSimilarityHybridRecommenderNormal(URM_train, recommender1.W_sparse, recommender2.W_sparse)
    #     recommender_object.fit(**CF_optimal_hyperparameters[label])
    #     recommender_object_dict[label] = recommender_object

    # for label, recommender_class in score_hybrid_recommender.items():
    #     recommender1 = recommender_object_dict[recommender_class[0]]
    #     recommender2 = recommender_object_dict[recommender_class[1]]
    #     recommender_object = TwoDifferentModelRecommender(URM_train, recommender1, recommender2)
    #     recommender_object.fit(**CF_optimal_hyperparameters[label])
    #     recommender_object_dict[label] = recommender_object

    for group_id in range(0, n_groups):
    
        start_pos = group_id*block_size
        if group_id != n_groups - 1:
            end_pos = min((group_id+1)*block_size, len(profile_length))
        else:
            end_pos = -1
        
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

    





