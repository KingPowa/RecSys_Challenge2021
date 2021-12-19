import pandas as pd
import numpy as np

def getMeanEntropy(feature_CSR, URM_all_CSR):
    matrix_usergenres = URM_all_CSR * feature_CSR
    df_usergenres = pd.DataFrame(matrix_usergenres.toarray())
    # df_usergenres
    # print(df_usergenres.idxmax(axis=1))
    total_views_per_user = df_usergenres.sum(axis=1)
    # print(total_views_per_user)
    df_prob = df_usergenres.div(total_views_per_user, axis=0)
    df_ent = df_prob.mul(-np.log2(df_prob))
    df_ent = df_ent.fillna(0)
    df_entropy = df_ent.sum(axis=1)

    return df_entropy
