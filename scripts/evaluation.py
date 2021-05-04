from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import scripts.data as d
from scripts.recommender import CollaborativeFiltering


def clean_items_from_matrix(df):
    drop_list = []
    for col in df.columns:
        cond = (df[col].isin([np.nan, 0])).all()
        if cond:
            drop_list.append(col)
    return df.drop(drop_list, axis=1)


def create_ratings_table():
    portfolio_pp, profile_pp, transcript_pp = d.read_and_preprocess()

    events_binary = pd.get_dummies(transcript_pp["event"])
    transcript_comb = pd.concat([transcript_pp, events_binary], axis=1)

    # Create the user-offer matrix based on the proportion of times an offer is completed
    user_offer_actions = transcript_comb.groupby(["person", "offer_id"]).agg(
        offer_completed_sum=pd.NamedAgg(column="offer completed", aggfunc="sum"),
        offer_viewed_sum=pd.NamedAgg(column="offer viewed", aggfunc="sum"),
        offer_received_sum=pd.NamedAgg(column="offer received", aggfunc="sum"),
    )

    offer_completed_ratio = np.divide(
        user_offer_actions["offer_completed_sum"],
        user_offer_actions["offer_received_sum"],
    )

    return offer_completed_ratio


def remove_informational_offers(ratings_table, information_offer_ids=[]):
    if not information_offer_ids:
        information_offer_ids = [
            "3f207df678b143eea3cee63160fa8bed",
            "5a8bc65990b245e5a138643cd4eb9837",
        ]
    return ratings_table[~(ratings_table["offer_id"].isin(information_offer_ids))]


class RecommenderEval:
    def __init__(self):
        print("Class initialised")

    def clean_data(self, ratings, min_ratings=5):
        print(
            "cleaning data only to contain users with at least {} ratings".format(
                min_ratings
            )
        )

        original_size = ratings.shape[0]

        ratings = remove_informational_offers(ratings)

        user_count = ratings[["user_id", "offer_id"]]
        user_count = user_count.groupby("user_id").count()
        user_count = user_count.reset_index()
        user_ids = user_count[user_count["offer_id"] >= min_ratings]["user_id"]

        ratings = ratings[ratings["user_id"].isin(user_ids)]
        new_size = ratings.shape[0]
        print("reduced dataset from {} to {}".format(original_size, new_size))
        return ratings

    def split_users(self, num_folds=5):
        kf = KFold(n_splits=num_folds)
        return kf

    def split_data(self, min_rank, ratings, test_users, train_users):
        train = ratings[ratings["user_id"].isin(train_users)]
        test_temp = ratings[ratings["user_id"].isin(test_users)]

        test_temp["rating_timestamp"] = np.arange(len(test_temp))

        test_temp["rank"] = test_temp.groupby("user_id")["rating_timestamp"].rank(
            ascending=False
        )
        test = test_temp[test_temp["rank"] > min_rank]

        additional_training_data = test_temp[test_temp["rank"] >= min_rank]
        train = train.append(additional_training_data)

        return test, train


if __name__ == "__main__":

    # Load ratings (stacked)
    stacked_ratings = create_ratings_table()
    stacked_ratings = stacked_ratings.reset_index()
    stacked_ratings.columns = ["user_id", "offer_id", "rating"]

    single_user = "00857b24b13f4fe0ad17b605f00357f5"
    single_user_ratings = stacked_ratings[stacked_ratings["user_id"] == single_user]

    recommender_eval = RecommenderEval()

    # Clean ratings (remove users where minimum number of ratings is below a threshold)
    clean_ratings = recommender_eval.clean_data(ratings=stacked_ratings, min_ratings=3)
    users = clean_ratings["user_id"].unique()

    # Split users into training and test using KFolds
    kf = recommender_eval.split_users()

    validation_no = 0
    paks, raks, maes = Decimal(0.0), Decimal(0.0), Decimal(0.0)

    min_rank = 4
    for train, test in kf.split(users):
        overlap = set(train).intersection(set(test))
        if overlap:
            print("overlap is {overlap}")
        validation_no += 1
        test_data, train_data = recommender_eval.split_data(
            min_rank, clean_ratings, users[test], users[train]
        )

        train_data_1 = train_data.copy()
        train_data_1.set_index(["user_id", "offer_id"], inplace=True)
        train_matrix = train_data_1["rating"].unstack()

        cf_recommender = CollaborativeFiltering(top_k=3, basis="item")

        cf_recommender.train(train_matrix)

        counter = 0
        users_precision_list = []
        users_recall_list = []
        for user in users[test]:
            counter += 1
            single_user_ratings = stacked_ratings[stacked_ratings["user_id"] == user]
            user_train_data = train_data[train_data["user_id"] == user]
            user_test_data = test_data[test_data["user_id"] == user]
            if not user_test_data.empty:
                print(
                    f"Generating recommendations for user {user} ({counter}/{len(users[test])})"
                )
                user_recs = cf_recommender.recommend_for_user(user=user, n=10)

                offered_and_redeemed = user_test_data[user_test_data["rating"] > 0][
                    "offer_id"
                ]
                offered_not_redeemed = user_test_data[user_test_data["rating"] == 0][
                    "offer_id"
                ]

                tp_offers = [r for r in user_recs if r in offered_and_redeemed]
                fp_offers = [r for r in user_recs if r in offered_not_redeemed]
                fn_offers = [r for r in offered_and_redeemed if r not in user_recs]
                tn_offers = [r for r in offered_not_redeemed if r not in user_recs]

                with np.errstate(divide="ignore", invalid="ignore"):
                    user_precision = np.divide(
                        len(tp_offers), (len(tp_offers) + len(fp_offers))
                    )
                    user_recall = np.divide(
                        len(tp_offers), (len(tp_offers) + len(fn_offers))
                    )
                print("next")

                users_precision_list.append(user_precision)
                users_recall_list.append(user_recall)

            if counter % 10 == 0:
                print(counter)
        print("end of loop")
        # paks += user_precision
        # raks += user_recall

        # # Append Precision and Recall to list
        #
        # #
        # paks += pak
        # raks += rak
        # results = {'pak': paks / self.folds,
        #            'rak': raks / self.folds,
        #            'mae': maes / self.folds}
