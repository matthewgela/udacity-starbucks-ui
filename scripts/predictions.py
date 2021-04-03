import os
from typing import Dict

import pandas as pd
import preprocessing as pp
from similarity import compute_similarities


def compute_similar_offers(
    user,
    offer,
    user_offer_matrix,
    similarity_matrix,
    top_n=3,
):
    user_record = user_offer_matrix.loc[user].dropna()
    user_offers = user_record.index

    condition = (similarity_matrix.index.isin(user_offers)) & (
        similarity_matrix.index != offer
    )
    offer_col = similarity_matrix.loc[condition, offer].sort_values(ascending=False)
    neighbourhood = offer_col[:top_n]

    similar_offers_df = pd.concat([neighbourhood, user_record], axis=1)
    similar_offers_df = similar_offers_df.dropna(how="any")
    similar_offers_df.columns = ["similarity", "rating"]

    return similar_offers_df


def compute_predicted_rating(offer, similar_offers_df, mean_rating):
    similar_offers_df["prediction_contribution"] = (
        similar_offers_df["similarity"] * similar_offers_df["rating"]
    )

    predicted_rating = (
        similar_offers_df["prediction_contribution"].sum()
        / similar_offers_df["similarity"].sum()
    )
    predicted_rating += mean_rating.loc[offer]

    return predicted_rating


def compute_all_ratings_for_user(
    user, user_offer_matrix, mean_rating, similarity_matrix
):
    user_ratings = user_offer_matrix.loc[user]
    all_offers = user_ratings.index
    rated_offers = user_ratings.copy().dropna().index
    not_rated_offers = [offer for offer in all_offers if offer not in rated_offers]

    user_ratings_with_predictions = user_ratings.copy()
    for offer in not_rated_offers:
        similar_offers = compute_similar_offers(
            user=user,
            offer=offer,
            user_offer_matrix=user_offer_matrix,
            similarity_matrix=similarity_matrix,
            top_n=3,
        )

        prediction_rating = compute_predicted_rating(
            offer=offer, similar_offers_df=similar_offers, mean_rating=mean_rating
        )

        user_ratings_with_predictions[offer] = prediction_rating
        print(user_ratings_with_predictions)

    return user_ratings_with_predictions


if __name__ == "__main__":
    user1 = "0009655768c64bdeb2e877511632db8f"

    item_locations: Dict[str, str] = {
        "similarity_matrix": "data_cache/similarity_matrix.csv",
        "normalised_user_offer_matrix": "data_cache/normalised_user_offer_matrix.csv",
        "mean_rating": "data_cache/mean_rating.csv",
    }

    load_items = dict()
    for val in item_locations.values():
        if not os.path.isfile(val):
            compute_similarities()
            if os.path.isfile(val):
                print(f"file {val} exists after computing similarities")
            else:
                print(
                    f"file {val} doesn't exist after computing similarities, investigate"
                )
            break

    for key, value in item_locations.items():
        load_items[key] = pp.read_data(file_path=value, file_type="csv", index_col=0)
        print(f"{key} loaded from cache")

    user_ratings = compute_all_ratings_for_user(
        user=user1,
        user_offer_matrix=load_items["normalised_user_offer_matrix"],
        mean_rating=load_items["mean_rating"],
        similarity_matrix=load_items["similarity_matrix"],
    )

    print(user_ratings)