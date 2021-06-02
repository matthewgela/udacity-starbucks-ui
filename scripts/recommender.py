import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import scripts.data as d
import scripts.preprocessing as pp


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


def fn_similarity(df, item_name, method, compare_item):
    mask = df[item_name].notna() & df[compare_item].notna()
    if method == "cosine":
        numerator = sum(df.loc[mask, item_name] * df.loc[mask, compare_item])
        denominator = np.sqrt(sum(df.loc[mask, item_name] ** 2)) * np.sqrt(
            sum(df.loc[mask, compare_item] ** 2)
        )
    elif method == "jaccard":
        numerator = np.sum(((df[item_name] > 0) & (df[compare_item] > 0)) & mask)
        denominator = np.sum(((df[item_name] > 0) | (df[compare_item] > 0)) & mask)
    return numerator / denominator if denominator != 0 else np.nan


def compute_similarity_array_multiprocessing(df, method, item_name):
    import multiprocessing as mp

    pool = mp.Pool(2)
    results = [
        pool.apply(fn_similarity, args=(df, item_name, method, compare_item))
        for compare_item in tqdm(df.columns)
    ]
    pool.close()

    results_array = np.array(results)

    similarity_matrix_df = pd.DataFrame(results_array, columns=[item_name])
    similarity_matrix_df.index = df.columns
    return similarity_matrix_df


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


def compute_similarity_matrix_vectorised(df, method):
    if method == "cosine":
        pass
    elif method == "jaccard":
        n = df.shape[1]
        # Get the row, col indices that are to be set in output array
        r, c = np.tril_indices(n, -1)

        # Use those indicees to slice out respective columns
        p1 = df.values[:, c]
        p2 = df.values[:, r]

        # Perform n11 and n00 vectorized computations across all indexed columns
        n11v = ((p1 > 0) & (p2 > 0)).sum(0)
        n00v = (((p1 > 0) | (p2 > 0)) & ~(np.isnan(p1) | np.isnan(p2))).sum(0)

        # Finally, setup output array and set final division computations
        out = np.eye(n)

        default = np.empty(n00v.shape)
        default[:] = np.nan
        out[c, r] = np.divide(n11v, n00v, out=default, where=n00v != 0)

        out = out + out.T - np.diag(np.diag(out))

        similarity_matrix_df = pd.DataFrame(out, columns=df.columns)
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
    def __init__(self, basis, similarity_method, n_sim=None, similarity_threshold=None):
        self.user_offer_matrix = None
        self.similarity_matrix = None
        self.basis = basis
        self.n_sim = n_sim
        self.similarity_method = similarity_method
        self.similarity_threshold = similarity_threshold

    def _compute_similarity(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_similar_offers(
        self,
        user,
        offer,
    ):

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

        user_col = self.similarity_matrix[user]

        offer_ratings = self.user_offer_matrix[offer]

        users_df = pd.concat([user_col, offer_ratings], axis=1)
        users_df = users_df.dropna(how="any")
        users_df.columns = ["similarity", "rating"]

        similar_users_df = users_df[users_df["similarity"] != 0].sort_values(
            by="similarity", ascending=False
        )

        if self.similarity_threshold:
            similar_users_df = similar_users_df[
                similar_users_df["similarity"] > self.similarity_threshold
            ]
        if self.n_sim:
            similar_users_df = similar_users_df.head(self.n_sim)

        return similar_users_df

    def _compute_predicted_rating(self, user, similar_offers_df):
        if not similar_offers_df.empty:
            similar_offers_df["prediction_contribution"] = (
                similar_offers_df["similarity"] * similar_offers_df["rating"]
            )

            predicted_rating = (
                similar_offers_df["prediction_contribution"].sum()
                / similar_offers_df["similarity"].sum()
            )
            return predicted_rating
        else:
            return -1

    def compute_all_ratings_for_user(self, user):
        user_ratings = self.user_offer_matrix.loc[user]
        all_offers = user_ratings.index
        rated_offers = user_ratings.copy().dropna().index
        not_rated_offers = [offer for offer in all_offers if offer not in rated_offers]

        user_ratings_pre = user_ratings.copy()
        user_ratings_with_predictions = user_ratings_pre.copy()

        item_id = user if self.basis == "user" else None

        if not self.compute_similarity_matrix and not self.load_similarity_matrix:
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
            not_rated_before_mask = (user_ratings_df["Ratings"].isna()) & (
                user_ratings_df["Ratings with predictions"] > 0
            )
            new_ratings_table = user_ratings_df.loc[not_rated_before_mask].copy()
            return list(new_ratings_table["Offer"][:n_recommend])


class CollaborativeFiltering(BaseRecommender):
    def __init__(self, basis, similarity_method, n_sim=None, similarity_threshold=None):
        super().__init__(basis, similarity_method, n_sim, similarity_threshold)

    def _compute_similarity(self, user_offer_matrix, item_id=None):
        if self.basis == "item":
            if item_id:
                similarity_matrix = compute_similarity_array(
                    df=user_offer_matrix,
                    method=self.similarity_method,
                    item_name=item_id,
                )
            else:
                from time import time

                start = time()
                similarity_matrix = compute_similarity_matrix_vectorised(
                    df=user_offer_matrix, method=self.similarity_method
                )
                end = time() - start
                print(f"item similarity computed in: {end} seconds")
        elif self.basis == "user":
            if item_id:
                from time import time

                start = time()
                try:
                    df_path = f"data_cache/similarity_matrix_{self.basis}.csv"
                    similarity_matrix = pd.read_csv(
                        df_path, usecols=["person", item_id], index_col="person"
                    )
                except FileNotFoundError:
                    similarity_matrix = compute_similarity_array(
                        df=user_offer_matrix.T,
                        method=self.similarity_method,
                        item_name=item_id,
                    )

                end = time() - start
                print(f"{end} seconds")
            else:
                from time import time

                start = time()
                similarity_matrix = compute_similarity_matrix_vectorised(
                    df=user_offer_matrix.T, method=self.similarity_method
                )
                end = time() - start
                print(f"whole user matrix: {end} seconds")

        return similarity_matrix

    def train(
        self,
        user_offer_matrix,
        compute_similarity_matrix=False,
        load_similarity_matrix=True,
    ):
        self.compute_similarity_matrix = compute_similarity_matrix
        self.load_similarity_matrix = load_similarity_matrix
        self.user_offer_matrix = remove_informational_offers(user_offer_matrix)
        if load_similarity_matrix:
            self.similarity_matrix = self._load_similarity_matrix()
        elif compute_similarity_matrix:
            self.similarity_matrix = self._compute_similarity(self.user_offer_matrix)

    def _load_similarity_matrix(self):
        try:
            df_path = f"data_cache/similarity_matrix_{self.basis}.csv"
            print(f"Loading cached file: {df_path}")
            similarity_matrix = pp.read_data(df_path, file_type="csv", index_col=0)
            print("Loaded successfully.")
            # similarity_matrix = similarity_matrix.to_pandas()
            print("Converted to pandas")
        except FileNotFoundError:
            print("No cached similarity file, creating from scratch")
            similarity_matrix = self._compute_similarity(self.user_offer_matrix)
            self._export_similarity_matrix(similarity_matrix)
        return similarity_matrix

    def _export_similarity_matrix(self, similarity_matrix):
        print("Exporting similarity matrix to cache")

        table = pa.Table.from_pandas(similarity_matrix)
        pq.write_table(table, f"data_cache/similarity_matrix_{self.basis}.parquet")


class ContentBasedFiltering(BaseRecommender):
    def __init__(self, basis, similarity_method, n_sim=None, similarity_threshold=None):
        super().__init__(basis, similarity_method, n_sim, similarity_threshold)

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
    cf_recommender.train(
        user_offer_matrix, compute_similarity_matrix=False, load_similarity_matrix=False
    )
    recs = cf_recommender.recommend(test_list, 3)

    # Content based filtering recommender
    # content_table = create_content_table(basis="item")
    # cbf_recommender = ContentBasedFiltering(
    #     n_sim=3, basis="item", similarity_method="jaccard"
    # )
    # cbf_recommender.train(user_offer_matrix, content_table)
    # recs_cbf = cbf_recommender.recommend(test_list, 3)
    print("recs done")
