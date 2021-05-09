import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from scripts.recommender import compute_similarity_matrix


def clean_items_from_matrix(df):
    drop_list = []
    for col in df.columns:
        cond = (df[col].isin([np.nan, 0])).all()
        if cond:
            drop_list.append(col)
    return df.drop(drop_list, axis=1)


def create_test_ratings():
    test_ratings_matrix = [[1, 1, 0], [0, 0.01, 1], [0.33, 0.2, 0.7], [0.6, 0.45, 0.2]]
    index_list = ["user1", "user2", "user3", "user4"]
    column_list = ["offer1", "offer2", "offer3"]

    test_ratings_df = pd.DataFrame(
        data=test_ratings_matrix, index=index_list, columns=column_list
    )
    return test_ratings_df


def create_test_ratings_with_nas():
    test_ratings_matrix = [
        [1, 1, np.nan],
        [np.nan, 0.01, 1],
        [0.33, 0.2, np.nan],
        [0.6, 0.45, 0.2],
    ]
    index_list = ["user1", "user2", "user3", "user4"]
    column_list = ["offer1", "offer2", "offer3"]

    test_ratings_df = pd.DataFrame(
        data=test_ratings_matrix, index=index_list, columns=column_list
    )
    return test_ratings_df


def test_cosine_similarity(ratings_matrix):
    cosines_sklearn = cosine_similarity(ratings_matrix)
    cosines_custom = compute_similarity_matrix(unstacked_ratings.T, method="cosine")
    assert all(cosines_custom == cosines_sklearn)


if __name__ == "__main__":
    unstacked_ratings = create_test_ratings()

    # Test cosine similarity
    cosines_sklearn = cosine_similarity(unstacked_ratings.T)
    cosines_custom = compute_similarity_matrix(unstacked_ratings, method="cosine")
    assert all(cosines_custom == cosines_sklearn)

    unstacked_ratings_with_nas = create_test_ratings_with_nas()

    cosines_custom_with_nas = compute_similarity_matrix(
        unstacked_ratings_with_nas, method="cosine"
    )

    normalised_unstacked_ratings_with_nas = unstacked_ratings_with_nas.sub(
        unstacked_ratings_with_nas.mean(axis="rows", skipna=True)
    )

    cosine_custom_normalised = compute_similarity_matrix(
        normalised_unstacked_ratings_with_nas, method="cosine"
    )

    print("Done")

    # stacked_ratings = create_ratings_table()
    # stacked_ratings = stacked_ratings.reset_index()
    # stacked_ratings.columns = ["user_id", "offer_id", "rating"]
    #
    # single_user_ratings = stacked_ratings[stacked_ratings["user_id"] == single_user]
    #
    # cf_recommender = CollaborativeFiltering(top_k=3, basis="item")
    #
    # cf_recommender.train(train_matrix)
    #
    # user_recs = cf_recommender.recommend_for_user(user=user, n=10)
    #
