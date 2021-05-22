import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

import scripts.data as d
from scripts.recommender import CollaborativeFiltering, ContentBasedFiltering


def clean_items_from_matrix(df):
    drop_list = []
    for col in df.columns:
        cond = (df[col].isin([np.nan, 0])).all()
        if cond:
            drop_list.append(col)
    return df.drop(drop_list, axis=1)


def create_content_table(basis):
    portfolio_pp, profile_pp, _ = d.read_and_preprocess()
    if basis == "item":
        content_table = portfolio_pp.copy()

        offer_type_one_hot = pd.get_dummies(content_table["offer_type"])
        content_table = pd.concat([content_table, offer_type_one_hot], axis=1)

        drop_columns = ["channels", "offer name", "offer_type"]
        content_table = content_table.drop(drop_columns, axis=1)
        content_table = content_table.set_index("id")

        return content_table
    elif basis == "user":
        return profile_pp


def create_stacked_ratings_table():
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

    stacked_ratings = offer_completed_ratio.reset_index()
    stacked_ratings.columns = ["user_id", "offer_id", "rating"]

    return stacked_ratings


def unstack_ratings(train_data):
    train_data_1 = train_data.copy()
    train_data_1.set_index(["user_id", "offer_id"], inplace=True)
    train_matrix = train_data_1["rating"].unstack()
    return train_matrix


def remove_informational_offers(ratings_table, information_offer_ids=[]):
    if not information_offer_ids:
        information_offer_ids = [
            "3f207df678b143eea3cee63160fa8bed",
            "5a8bc65990b245e5a138643cd4eb9837",
        ]
    return ratings_table[~(ratings_table["offer_id"].isin(information_offer_ids))]


class Evaluation:
    def __init__(self, min_ratings_threshold, n_folds, min_rank, k):
        print("Class initialised")
        self.min_ratings_threshold = min_ratings_threshold  # Minimum number of recommendations needed for the user to be counted in evaluation
        self.n_folds = n_folds  # Number of folds
        self.min_rank = min_rank  # For users in the test set, the number of ratings included in the training set
        self.k = k  # Number of recommendations, k, for which the recommender should be evaluated

    def clean_data(self, ratings):
        print(
            "cleaning data only to contain users with at least {} ratings".format(
                self.min_ratings_threshold
            )
        )

        original_size = ratings.shape[0]

        ratings = remove_informational_offers(ratings)

        user_count = ratings[["user_id", "offer_id"]]
        user_count = user_count.groupby("user_id").count()
        user_count = user_count.reset_index()
        user_ids = user_count[user_count["offer_id"] >= self.min_ratings_threshold][
            "user_id"
        ]

        ratings = ratings[ratings["user_id"].isin(user_ids)]
        new_size = ratings.shape[0]
        print("reduced dataset from {} to {}".format(original_size, new_size))
        return ratings

    def split_users(self):
        kf = KFold(n_splits=self.n_folds, random_state=1)
        return kf

    def split_data(self, ratings, test_users, train_users):
        train = ratings[ratings["user_id"].isin(train_users)]
        test_temp = ratings[ratings["user_id"].isin(test_users)]

        test_temp["rating_timestamp"] = np.arange(len(test_temp))

        test_temp["rank"] = test_temp.groupby("user_id")["rating_timestamp"].rank(
            ascending=False
        )
        test = test_temp[test_temp["rank"] > self.min_rank]

        additional_training_data = test_temp[test_temp["rank"] <= self.min_rank]
        train = train.append(additional_training_data)

        return test, train

    @staticmethod
    def train_recommender(recommender_type, basis, n_sim, training_data, content_table):
        if recommender_type == "cf":
            recommender = CollaborativeFiltering(n_sim=n_sim, basis=basis)
            recommender.train(training_data)
        elif recommender_type == "cbf":
            recommender = ContentBasedFiltering(n_sim=n_sim, basis=basis)
            recommender.train(training_data, content_table)
        else:
            print("Recommender not supported by evaluation")
            pass
        return recommender

    def run(self, clean_ratings, users, n_sim, rec_type, basis_type, kfold):

        precision_all_folds_list = []
        recall_all_folds_list = []
        for validation_no, (train, test) in enumerate(kfold.split(users)):
            print(f"Split no. {validation_no} / 5")

            test_data, train_data = evaluation.split_data(
                clean_ratings, users[test], users[train]
            )

            # Preparing data to be fed into CF recommender (may change for CBF)
            # TODO - Turn into 'Preprocess training fold?' which would be different per recommender type
            train_matrix = unstack_ratings(train_data)

            content_table = (
                create_content_table(basis_type) if rec_type == "cbf" else None
            )
            # Initialise and train recommender based on training fold
            recommender = evaluation.train_recommender(
                recommender_type=rec_type,
                basis=basis_type,
                n_sim=n_sim,
                training_data=train_matrix,
                content_table=content_table,
            )

            # Evaluate the recommender
            users_precision_list = []
            users_recall_list = []

            for user in tqdm(users[test]):
                # Test data sets for a single user
                user_test_data = test_data[test_data["user_id"] == user]

                if not user_test_data.empty:
                    # Produce top k recommendations for the user
                    user_recs = recommender.recommend_for_user(user=user, n_recommend=k)

                    # Calculating TP, FP, FN and TN based on the top k recs
                    offered_and_redeemed = list(
                        user_test_data[user_test_data["rating"] > 0]["offer_id"]
                    )

                    offered_not_redeemed = list(
                        user_test_data[user_test_data["rating"] == 0]["offer_id"]
                    )

                    tp_offers = [r for r in user_recs if r in offered_and_redeemed]
                    fp_offers = [r for r in user_recs if r in offered_not_redeemed]
                    fn_offers = [r for r in offered_and_redeemed if r not in user_recs]

                    # Calculating user precision and recall at k for user
                    with np.errstate(divide="ignore", invalid="ignore"):
                        user_precision = np.divide(
                            len(tp_offers), (len(tp_offers) + len(fp_offers))
                        )
                        user_recall = np.divide(
                            len(tp_offers), (len(tp_offers) + len(fn_offers))
                        )

                    # Append to list
                    users_precision_list.append(user_precision)
                    users_recall_list.append(user_recall)

            # Average precision at k for this fold
            precision_fold = np.nanmean(users_precision_list)
            recall_fold = np.nanmean(users_recall_list)

            precision_all_folds_list.append(precision_fold)
            recall_all_folds_list.append(recall_fold)

            print("Precision for fold is:", precision_fold)
            print("Recall for fold is:", recall_fold)

        # A list of the average precision at k for all folds
        precision_all_folds = np.nanmean(precision_all_folds_list)
        recall_all_folds = np.nanmean(recall_all_folds_list)

        print("Overall precision is:", precision_all_folds)
        print("Overall recall is:", recall_all_folds)

        evaluation_results = pd.DataFrame(
            dict(
                fold=np.array(range(self.n_folds)) + 1,
                precision=precision_all_folds_list,
                recall=recall_all_folds_list,
            ),
        )
        # Append average over all folds
        evaluation_results = evaluation_results.append(
            {
                "fold": "average",
                "precision": precision_all_folds,
                "recall": recall_all_folds,
            },
            ignore_index=True,
        )
        return evaluation_results

    @staticmethod
    def export_results(evaluation_results, rec_type):
        evaluation_results.to_csv(
            f"data_cache/evaluation_results_{rec_type}.csv", index=False
        )


if __name__ == "__main__":
    # Load ratings (stacked)
    stacked_ratings = create_stacked_ratings_table()

    # Set parameters for Evaluation
    min_ratings_threshold = 3  # Minimum number of recommendations needed for the user to be counted in evaluation
    n_folds = 5  # Number of folds
    min_rank = 2  # For users in the test set, the number of ratings included in the training set
    k = 3  # Number of recommendations, k, for which the recommender should be evaluated

    # Set parameters for recommender
    n_sim = 1  # Neighbourhood of similarity for recommender
    # rec_type = "cbf"  # Type of recommender algorithm
    basis_type = "item"  # Users vs Items - what is used to calculate similarity metrics

    evaluation = Evaluation(min_ratings_threshold, n_folds, min_rank, k)

    # Clean ratings (remove users where minimum number of ratings is below a threshold)
    clean_ratings = evaluation.clean_data(ratings=stacked_ratings)
    users = clean_ratings["user_id"].unique()

    # Split users into training and test using KFolds
    train_test_split = evaluation.split_users()

    rec_types = ["cf", "cbf"]  # Type of recommender algorithm

    for rec_type in rec_types:
        recommender_evaluation = evaluation.run(
            clean_ratings, users, n_sim, rec_type, basis_type, train_test_split
        )

        # Export evaluation results
        evaluation.export_results(recommender_evaluation, rec_type)
