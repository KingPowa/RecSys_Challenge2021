    #!/usr/bin/env python
    # coding: utf-8

    # In[16]:



import sys
import cython
import numpy as np
import scipy.sparse as sps

sys.path.append('../RecSysRep/')


    # In[17]:

if __name__ == '__main__':

    import Basics.Load as ld

    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()
    ICM_length_all = ld.getICMlength("5km")

    ICM_all = sps.hstack((ICM_genre_all, ICM_channel_all, ICM_length_all))


    # # SLIM Model

    # In[19]:


    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8, seed=1234)


    # In[20]:


    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender_test

    recommender1 = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_train.tocsr(), ICM_all)

    hyp = {"l1_ratio" : 0.025887359156206147, "topK": 2140, "alpha": 0.009567288586539689, "workers": 8, "mw": 1}

    recommender1.fit(**hyp)

    print(evaluator_validation.evaluateRecommender(recommender1))

    recommender1.URM_train = recommender1.URM_original.tocsr()

    print(evaluator_validation.evaluateRecommender(recommender1))




