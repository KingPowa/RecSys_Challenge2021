
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
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
 
if __name__ == '__main__':

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, _ = ld.getCOOs()
    ICM_length_all_5km = ld.getICMlength('5km')
    ICM_length_all_3km = ld.getICMlength('5km')

    ICM_stacked = hstack([ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_length_all_5km])


    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed = 1234)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

    optimal_hyperparameters = {'topK': 6000, 'l1_ratio': 0.0005495104968035837, 'alpha': 0.08007142704041009, 'workers': 8} 

    CFrecommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    CFrecommender.fit(**optimal_hyperparameters)

    output_folder_path = "../result_experiments/11_FW/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    n_cases = 100
    n_random_starts = int(n_cases*0.3)
    metric_to_optimize = "MAP"   
    cutoff_to_optimize = 10

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(5, 6000)
    hyperparameters_range_dictionary["add_zeros_quota"] = Real(low = 0, high = 1, prior = 'uniform')
    hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

    recommender_class = CFW_D_Similarity_Linalg

    hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_stacked, CFrecommender.W_sparse],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )

    hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender_class.RECOMMENDER_NAME,
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )

    CFrecommender_ALL = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all)
    CFrecommender_ALL.fit(**optimal_hyperparameters)

    recommender = CFW_D_Similarity_Linalg(URM_all, ICM_stacked, CFrecommender_ALL.W_sparse)

    from Recommenders.DataIO import DataIO

    data_loader = DataIO(folder_path = output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    best_hyperparameters = search_metadata["hyperparameters_best"]
    recommender.fit(**best_hyperparameters)

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


    submission.to_csv('../subs/submission {:%Y_%m_%d %H_%M_%S}.csv'.format(now), index=False)



    

    





