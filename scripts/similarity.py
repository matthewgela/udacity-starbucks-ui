import numpy as np
import pandas as pd

import scripts.data as d


def cosine_similarity_custom(df):
    similarity_matrix = np.zeros([df.shape[1], df.shape[1]])
    for i, offer1 in enumerate(df.columns):
        for j, offer2 in enumerate(df.columns):
            mask = df[offer1].notna() & df[offer2].notna()
            numerator = sum(df.loc[mask, offer1] * df.loc[mask, offer2])
            denominator = np.sqrt(sum(df.loc[mask, offer1] ** 2)) * np.sqrt(
                sum(df.loc[mask, offer2] ** 2)
            )
            similarity_matrix[i, j] = (
                numerator / denominator if denominator != 0 else np.nan
            )

    similarity_matrix_df = pd.DataFrame(similarity_matrix, columns=df.columns)
    similarity_matrix_df.index = df.columns
    return similarity_matrix_df


def similar_offers(
    user,
    offer,
    normalised_user_offer_matrix,
    item_similarity,
    mean_offer_rating,
    top_n=3,
):

    user_record = normalised_user_offer_matrix.loc[user].dropna()
    user_offers = user_record.index

    condition = (item_similarity.index.isin(user_offers)) & (
        item_similarity.index != offer1
    )
    offer_col = item_similarity.loc[condition, offer].sort_values(ascending=False)
    neighbourhood = offer_col[:top_n]

    predicted_rating_df = pd.concat([neighbourhood, user_record], axis=1)
    predicted_rating_df = predicted_rating_df.dropna(how="any")

    predicted_rating_df.columns = ["similarity", "rating"]
    predicted_rating_df["prediction_contribution"] = (
        predicted_rating_df["similarity"] * predicted_rating_df["rating"]
    )

    prediction = (
        predicted_rating_df["prediction_contribution"].sum()
        / predicted_rating_df["similarity"].sum()
    )
    prediction += mean_offer_rating.loc[offer]

    return predicted_rating_df, prediction


# Cosine similarity


if __name__ == "__main__":
    # Load data files
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

    mean_offer_rating = user_offer_matrix.mean(axis=0, skipna=True)

    normalised_user_offer_matrix = user_offer_matrix - mean_offer_rating

    item_similarity = cosine_similarity_custom(normalised_user_offer_matrix)

    offer1 = "ae264e3637204a6fb9bb56bc8210ddfd"
    user1 = "aa4862eba776480b8bb9c68455b8c2e1"

    neighbourhood, prediction = similar_offers(
        user1,
        offer1,
        normalised_user_offer_matrix,
        item_similarity,
        mean_offer_rating,
        top_n=3,
    )

    print("Done")
