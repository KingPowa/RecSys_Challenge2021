#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
sys.path.append('../RecSysRep/')


# In[7]:

if __name__ == "__main__":

    import Basics.Load as ld
    import numpy as np
    import scipy.sparse as sps


    URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all = ld.getCOOs()

    # URM_train, URM_val = ld.getSplit(URM_train_val, 5678, 0.8)

    hyp = {"topK": 1049, "alpha": 1.1626473723475605, "beta": 0.6765017195261293, "normalize_similarity": True, "implicit": True}

    # # SLIM Model

    # In[8]:


    # In[9]:


    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender


    # In[10]:

    recommender = RP3betaRecommender(URM_all)

    recommender.fit(**hyp)
    recommender.save_model('./', 'RP3_ALL')



