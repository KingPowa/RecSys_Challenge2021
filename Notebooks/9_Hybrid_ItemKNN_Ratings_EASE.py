#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('../RecSysRep/')


import Basics.Load as ld

URM_all, _, _, _, _ = ld.getCOOs()
ICM_all = ld.getICMall()


import os

ofp = "../models_temp/Similarity_Hybrid/"


from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIM_S_ElasticNetRecommender




hyp = {'l1_ratio': 0.0001043005302985496, 'topK': 1135, 'alpha': 0.06322495726710943, 'workers': 4, 'mw': 1.2937257828415842}


recommender1 = MultiThreadSLIM_SLIM_S_ElasticNetRecommender(URM_all, ICM_all)
if __name__ == '__main__':
    recommender1.fit(**hyp)

recommender1.save_model(ofp + 'S_SLIM_ALL_ALL/', 'S_SLIM_ALL_ALL')


