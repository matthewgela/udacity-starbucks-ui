import numpy as np
import pandas as pd
from tqdm import tqdm

import scripts.data as d


def create_user_offer_matrix():
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

    user_offer_matrix = offer_completed_ratio.unstack()

    return user_offer_matrix


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


def compute_similarity_array(df, method, item_name):
    similarity_matrix = np.zeros(df.shape[1])

    print(f"Calculating similarities for {item_name}")
    for i, compare_item in enumerate(tqdm((df.columns))):
        mask = df[item_name].notna() & df[compare_item].notna()
        if method == "cosine":
            numerator = sum(df.loc[mask, item_name] * df.loc[mask, compare_item])
            denominator = np.sqrt(sum(df.loc[mask, item_name] ** 2)) * np.sqrt(
                sum(df.loc[mask, compare_item] ** 2)
            )
        elif method == "jaccard":
            numerator = np.sum(((df[item_name] > 0) & (df[compare_item] > 0)) & mask)
            denominator = np.sum(((df[item_name] > 0) | (df[compare_item] > 0)) & mask)

        similarity_matrix[i] = numerator / denominator if denominator != 0 else np.nan

    similarity_matrix_df = pd.DataFrame(similarity_matrix, columns=[item_name])
    similarity_matrix_df.index = df.columns
    return similarity_matrix_df


def compute_similarity_matrix(df, method):
    similarity_matrix = np.zeros([df.shape[1], df.shape[1]])
    for i, offer1 in enumerate(df.columns):
        print(f"Calculating similarities for user {offer1}")
        for j, offer2 in enumerate(tqdm((df.columns))):
            mask = df[offer1].notna() & df[offer2].notna()
            if method == "cosine":
                numerator = sum(df.loc[mask, offer1] * df.loc[mask, offer2])
                denominator = np.sqrt(sum(df.loc[mask, offer1] ** 2)) * np.sqrt(
                    sum(df.loc[mask, offer2] ** 2)
                )
            elif method == "jaccard":
                numerator = np.sum(((df[offer1] > 0) & (df[offer2] > 0)) & mask)
                denominator = np.sum(((df[offer1] > 0) | (df[offer2] > 0)) & mask)

            similarity_matrix[i, j] = (
                numerator / denominator if denominator != 0 else np.nan
            )

    similarity_matrix_df = pd.DataFrame(similarity_matrix, columns=df.columns)
    similarity_matrix_df.index = df.columns
    return similarity_matrix_df


def create_mean_rating(user_offer_matrix):
    return user_offer_matrix.mean(axis=1, skipna=True)


def normalise_user_offer_matrix(user_offer_matrix, mean_offer_rating):
    return user_offer_matrix.sub(mean_offer_rating, axis="rows")


def remove_informational_offers(ratings_table, information_offer_ids=[]):
    if not information_offer_ids:
        information_offer_ids = [
            "3f207df678b143eea3cee63160fa8bed",
            "5a8bc65990b245e5a138643cd4eb9837",
        ]
    drop_cols = [
        offer for offer in information_offer_ids if offer in ratings_table.columns
    ]

    return ratings_table.drop(drop_cols, axis=1)


class BaseRecommender:
    def __init__(self, n_sim, basis, similarity_method):
        self.user_offer_matrix = None
        self.similarity_matrix = None
        self.basis = basis
        self.n_sim = n_sim
        self.similarity_method = similarity_method

    def _compute_similarity(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_similar_offers(
        self,
        user,
        offer,
    ):

        # if not self.compute_similarity_matrix:
        #     self.similarity_matrix = self._compute_similarity(self.user_offer_matrix, item_id=offer)

        user_record = self.user_offer_matrix.loc[user].dropna()
        user_offers = user_record.index

        condition = (self.similarity_matrix.index.isin(user_offers)) & (
            self.similarity_matrix.index != offer
        )
        offer_col = self.similarity_matrix.loc[condition, offer].sort_values(
            ascending=False
        )
        neighbourhood = offer_col[: self.n_sim]

        similar_offers_df = pd.concat([neighbourhood, user_record], axis=1)
        similar_offers_df = similar_offers_df.dropna(how="any")
        similar_offers_df.columns = ["similarity", "rating"]

        return similar_offers_df

    def _compute_similar_users(self, user, offer):

        # if not self.compute_similarity_matrix:
        #     self.similarity_matrix = self._compute_similarity(user_offer_matrix=self.user_offer_matrix, item_id=user)

        user_col = self.similarity_matrix[user]

        offer_ratings = self.user_offer_matrix[offer]

        users_df = pd.concat([user_col, offer_ratings], axis=1)
        users_df = users_df.dropna(how="any")
        users_df.columns = ["similarity", "rating"]

        users_df = users_df.sort_values(by="similarity", ascending=False)
        similar_users_df = users_df[users_df["similarity"] > 0.9]
        # similar_users_df = users_df.head(self.n_sim)

        return similar_users_df

    def _compute_predicted_rating(self, user, similar_offers_df):
        similar_offers_df["prediction_contribution"] = (
            similar_offers_df["similarity"] * similar_offers_df["rating"]
        )
        predicted_rating = (
            similar_offers_df["prediction_contribution"].sum()
            / similar_offers_df["similarity"].sum()
        )
        return predicted_rating

    def compute_all_ratings_for_user(self, user):
        user_ratings = self.user_offer_matrix.loc[user]
        all_offers = user_ratings.index
        rated_offers = user_ratings.copy().dropna().index
        not_rated_offers = [offer for offer in all_offers if offer not in rated_offers]

        user_ratings_pre = user_ratings.copy()
        user_ratings_with_predictions = user_ratings_pre.copy()

        item_id = user if self.basis == "user" else None

        if not self.compute_similarity_matrix:
            self.similarity_matrix = self._compute_similarity(
                self.user_offer_matrix, item_id=item_id
            )

        for offer in not_rated_offers:
            if self.basis == "item":
                neighbourhood = self._compute_similar_offers(
                    user=user,
                    offer=offer,
                )
            elif self.basis == "user":
                neighbourhood = self._compute_similar_users(
                    user=user,
                    offer=offer,
                )

            prediction_rating = self._compute_predicted_rating(
                user=user,
                similar_offers_df=neighbourhood,
            )

            user_ratings_with_predictions[offer] = prediction_rating

        ratings_table = pd.DataFrame(
            dict(
                {
                    "Offer": user_ratings_pre.index,
                    "Ratings": user_ratings_pre.values,
                    "Ratings with predictions": user_ratings_with_predictions.values,
                }
            )
        )

        return ratings_table

    def recommend(self, test_users, n):
        test_user_recommendations = []
        counter = 0
        for user in test_users:
            counter += 1
            print(
                f"Generating recommendations for user {user} ({counter}/{len(test_users)})"
            )
            user_ratings_df = self.compute_all_ratings_for_user(user)
            not_rated_before_mask = user_ratings_df["Ratings"].isna()
            new_ratings_table = user_ratings_df.loc[not_rated_before_mask].copy()
            new_ratings_table.sort_values(
                by="Ratings with predictions", ascending=False, inplace=True
            )
            test_user_recommendations.append(list(new_ratings_table["Offer"][:n]))
        return dict(zip(test_users, test_user_recommendations))

    def recommend_for_user(self, user, n_recommend, return_all_ratings=False):
        user_ratings_df = self.compute_all_ratings_for_user(user)
        user_ratings_df.sort_values(
            by="Ratings with predictions", ascending=False, inplace=True
        )
        if return_all_ratings:
            return user_ratings_df
        else:
            not_rated_before_mask = user_ratings_df["Ratings"].isna()
            new_ratings_table = user_ratings_df.loc[not_rated_before_mask].copy()
            return list(new_ratings_table["Offer"][:n_recommend])


class CollaborativeFiltering(BaseRecommender):
    def __init__(self, n_sim, basis, similarity_method):
        super().__init__(n_sim, basis, similarity_method)

    def _compute_similarity(self, user_offer_matrix, item_id=None):
        if self.basis == "item":
            if item_id:
                similarity_matrix = compute_similarity_array(
                    df=user_offer_matrix,
                    method=self.similarity_method,
                    item_name=item_id,
                )
            else:
                similarity_matrix = compute_similarity_matrix(
                    df=user_offer_matrix, method=self.similarity_method
                )
        elif self.basis == "user":
            if item_id:
                similarity_matrix = compute_similarity_array(
                    df=user_offer_matrix.T,
                    method=self.similarity_method,
                    item_name=item_id,
                )
            else:
                similarity_matrix = compute_similarity_matrix(
                    df=user_offer_matrix.T, method=self.similarity_method
                )

        return similarity_matrix

    def train(self, user_offer_matrix, compute_similarity_matrix=True):
        self.compute_similarity_matrix = compute_similarity_matrix
        self.user_offer_matrix = remove_informational_offers(user_offer_matrix)
        if compute_similarity_matrix:
            self.similarity_matrix = self._compute_similarity(self.user_offer_matrix)


class ContentBasedFiltering(BaseRecommender):
    def __init__(self, n_sim, basis, similarity_method):
        super().__init__(n_sim, basis, similarity_method)

    def _compute_similarity(self, content_table):
        if self.basis == "item":
            similarity_matrix = compute_similarity_matrix(
                df=content_table.T, method=self.similarity_method
            )
        elif self.basis == "user":
            pass

        return similarity_matrix

    def train(self, user_offer_matrix, content_table):
        self.user_offer_matrix = remove_informational_offers(user_offer_matrix)
        self.similarity_matrix = self._compute_similarity(content_table)


if __name__ == "__main__":
    # Collaborative filtering recommender
    user_offer_matrix = create_user_offer_matrix()
    # cf_recommender = CollaborativeFiltering(n_sim=3, basis="item", similarity_method="jaccard")
    cf_recommender = CollaborativeFiltering(
        n_sim=15, basis="user", similarity_method="jaccard"
    )
    test_list = ["0009655768c64bdeb2e877511632db8f"]
    cf_recommender.train(user_offer_matrix, compute_similarity_matrix=False)
    recs = cf_recommender.recommend(test_list, 3)

    # Content based filtering recommender
    content_table = create_content_table(basis="item")
    cbf_recommender = ContentBasedFiltering(
        n_sim=3, basis="item", similarity_method="jaccard"
    )
    cbf_recommender.train(user_offer_matrix, content_table)
    recs_cbf = cbf_recommender.recommend(test_list, 3)
    print("recs done")
