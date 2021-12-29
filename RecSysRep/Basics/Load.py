import pandas as pd
import scipy.sparse as sps
import numpy as np

def getDataframes():
    
    URM_path = '../data/data_train.csv'
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, 
                                    sep=",",
                                    dtype={0:int, 1:int, 2:float})
    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction"]
    URM_all_dataframe.head(n=10)

    ICM_genre_path = '../data/data_ICM_genre.csv'
    ICM_genre_all_dataframe = pd.read_csv(filepath_or_buffer=ICM_genre_path, 
                                    sep=",",
                                    dtype={0:int, 1:int, 2:float})
    ICM_genre_all_dataframe.columns = ["ItemID", "GenreID", "Match"]
    ICM_genre_all_dataframe.head(n=10)

    ICM_subgenre_path = '../data/data_ICM_subgenre.csv'
    ICM_subgenre_all_dataframe = pd.read_csv(filepath_or_buffer=ICM_subgenre_path, 
                                    sep=",",
                                    dtype={0:int, 1:int, 2:float})
    ICM_subgenre_all_dataframe.columns = ["ItemID", "SubgenreID", "Match"]
    ICM_subgenre_all_dataframe.head(n=10)

    ICM_channel_path = '../data/data_ICM_channel.csv'
    ICM_channel_all_dataframe = pd.read_csv(filepath_or_buffer=ICM_channel_path, 
                                    sep=",",
                                    dtype={0:int, 1:int, 2:float})
    ICM_channel_all_dataframe.columns = ["ItemID", "ChannelID", "Match"]
    ICM_channel_all_dataframe.head(n=10)

    ICM_event_path = '../data/data_ICM_event.csv'
    ICM_event_all_dataframe = pd.read_csv(filepath_or_buffer=ICM_event_path, 
                                    sep=",",
                                    dtype={0:int, 1:int, 2:float})
    ICM_event_all_dataframe.columns = ["ItemID", "EpisodeID", "Match"]
    ICM_event_all_dataframe.head(n=10)

    return URM_all_dataframe, ICM_genre_all_dataframe, ICM_subgenre_all_dataframe, ICM_channel_all_dataframe, ICM_event_all_dataframe

def getCOOs():
    
    URM_all_dataframe, ICM_genre_all_dataframe, ICM_subgenre_all_dataframe, ICM_channel_all_dataframe, ICM_event_all_dataframe = getDataframes()

    URM_all = sps.coo_matrix((URM_all_dataframe["Interaction"].values, 
                              (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)))

    # URM_all.tocsr()

    ICM_genre_all = sps.coo_matrix((ICM_genre_all_dataframe["Match"].values, 
                              (ICM_genre_all_dataframe["ItemID"].values, ICM_genre_all_dataframe["GenreID"].values)))

    # ICM_genre_all.tocsr()

    ICM_subgenre_all = sps.coo_matrix((ICM_subgenre_all_dataframe["Match"].values, 
                              (ICM_subgenre_all_dataframe["ItemID"].values, ICM_subgenre_all_dataframe["SubgenreID"].values)))

    # ICM_subgenre_all.tocsr().data

    ICM_channel_all = sps.coo_matrix((ICM_channel_all_dataframe["Match"].values, 
                              (ICM_channel_all_dataframe["ItemID"].values, ICM_channel_all_dataframe["ChannelID"].values)))

    # ICM_channel_all.tocsr()

    ICM_event_all = sps.coo_matrix((ICM_event_all_dataframe["Match"].values, 
                              (ICM_event_all_dataframe["ItemID"].values, ICM_event_all_dataframe["EpisodeID"].values)))


    return URM_all, ICM_genre_all, ICM_subgenre_all, ICM_channel_all, ICM_event_all

def getSplit(URM_all, seed = 1234, split = 0.8):
    
    np.random.seed(seed)

    train_test_split = split

    n_interactions = URM_all.nnz


    train_mask = np.random.choice([True,False], n_interactions, p=[train_test_split, 1-train_test_split])
    train_mask

    URM_train = sps.csr_matrix((URM_all.data[train_mask],
                                (URM_all.row[train_mask], URM_all.col[train_mask])))

    val_mask = np.logical_not(train_mask)

    URM_val = sps.csr_matrix((URM_all.data[val_mask],
                                (URM_all.row[val_mask], URM_all.col[val_mask])))

    return URM_train, URM_val

def getURMWeighted():
    URM_path = '../data/URM_wgh.csv'
    URM_weighted = pd.read_csv(filepath_or_buffer=URM_path)
    matr =sps.coo_matrix((URM_weighted["2"].values, 
                              (URM_weighted["0"].values, URM_weighted["1"].values)))
    return matr

def getICMlength(param = "3bal"):
    ICM_length_path = '../data/ICM_' + param + '_length.csv'
    ICM_length_all_dataframe = pd.read_csv(filepath_or_buffer=ICM_length_path)
    ICM_length_all = sps.coo_matrix(ICM_length_all_dataframe)
    return ICM_length_all
    
def getICMselected(param = "7"):
    ICM_selected_path = '../data/ICM_selected_' + param + ".csv"
    ICM_selected_dataframe = pd.read_csv(filepath_or_buffer=ICM_selected_path)
    ICM_selected = sps.coo_matrix(ICM_selected_dataframe)
    return ICM_selected
