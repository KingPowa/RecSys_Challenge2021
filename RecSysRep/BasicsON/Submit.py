import numpy as np
import pandas as pd

def getSubmission(my_recommender, at = 10, **hyp)
	# CHOOSE ALGORITHM HERE
	recommender = my_recommender # <-----
	K = at

	recommender.fit(**hyp)

	user_test_path = 'RecSys_Challenge2021/data/data_target_users_test.csv'
	user_test_dataframe = pd.read_csv(filepath_or_buffer=user_test_path,
									sep=",",
									dtype={0:int})

	subm_set = user_test_dataframe.to_numpy().T[0]


	subm_res = {"user_id":[], "item_list":[]}

	for user_id in subm_set:
		subm_res["user_id"].append(user_id)
		res = recommender.recommend(user_id, at=K)
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


	submission.to_csv('RecSys_Challenge2021/subs/submission {:%Y_%m_%d %H_%M_%S}.csv'.format(now), index=False)
