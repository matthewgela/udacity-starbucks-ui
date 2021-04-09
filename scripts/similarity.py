import numpy as np
import pandas as pd

import scripts.data as d


def compute_similarity_matrix(df, method="cosine"):
    similarity_matrix = np.zeros([df.shape[1], df.shape[1]])
    for i, offer1 in enumerate(df.columns):
        for j, offer2 in enumerate(df.columns):
            mask = df[offer1].notna() & df[offer2].notna()
            if method == "cosine":
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


def compute_similarities():
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

    mean_offer_rating = user_offer_matrix.mean(axis=1, skipna=True)

    normalised_user_offer_matrix = user_offer_matrix.sub(mean_offer_rating, axis="rows")

    item_similarity = compute_similarity_matrix(
        df=normalised_user_offer_matrix, method="cosine"
    )

    # Output items
    normalised_user_offer_matrix.to_csv("data_cache/normalised_user_offer_matrix.csv")
    item_similarity.to_csv("data_cache/similarity_matrix.csv")
    mean_offer_rating.to_csv("data_cache/mean_rating.csv")


if __name__ == "__main__":
    # Load data files
    compute_similarities()

    print("Similarities computed.")
