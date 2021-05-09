import pandas as pd

import scripts.preprocessing as pp
from scripts.recommender import CollaborativeFiltering, create_user_offer_matrix


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


def compute_predicted_rating(user, offer, similar_offers_df, mean_rating):
    similar_offers_df["prediction_contribution"] = (
        similar_offers_df["similarity"] * similar_offers_df["rating"]
    )
    predicted_rating = (
        similar_offers_df["prediction_contribution"].sum()
        / similar_offers_df["similarity"].sum()
    )
    predicted_rating += mean_rating.loc[user]

    return predicted_rating


def compute_all_ratings_for_user(
    user, user_offer_matrix, mean_rating, similarity_matrix
):
    user_ratings = user_offer_matrix.loc[user]
    all_offers = user_ratings.index
    rated_offers = user_ratings.copy().dropna().index
    not_rated_offers = [offer for offer in all_offers if offer not in rated_offers]

    user_ratings_pre = user_ratings.copy()
    user_ratings_pre += mean_rating.loc[user].values[0]
    user_ratings_with_predictions = user_ratings_pre.copy()

    for offer in not_rated_offers:
        print("offer: ", offer)
        similar_offers = compute_similar_offers(
            user=user,
            offer=offer,
            user_offer_matrix=user_offer_matrix,
            similarity_matrix=similarity_matrix,
            top_n=3,
        )

        prediction_rating = compute_predicted_rating(
            user=user,
            offer=offer,
            similar_offers_df=similar_offers,
            mean_rating=mean_rating,
        )

        user_ratings_with_predictions[offer] = prediction_rating
    print(user_ratings_with_predictions)

    return user_ratings_pre, user_ratings_with_predictions


def generate_prediction(user, recommender, num_recs, perform_mapping=False):
    if perform_mapping:
        id_mapping_table = pd.read_csv("data_cache/customer_id_mapping.csv")
        user = id_mapping_table.loc[
            id_mapping_table["membership number"] == int(user), "id"
        ].values[0]

    ratings_table = recommender.recommend_for_user(
        user, num_recs, return_all_ratings=True
    )

    offer_portfolio = pp.read_data(
        file_path="data_cache/portfolio_pp.csv", file_type=".csv", index_col=None
    )

    ratings_table_full = ratings_table.merge(
        offer_portfolio, how="left", left_on="Offer", right_on="id"
    )
    ratings_table_display_table = ratings_table_full[
        [
            "offer name",
            "offer_type",
            "difficulty",
            "reward",
            "Ratings",
            "Ratings with predictions",
        ]
    ]
    ratings_table_display_table = ratings_table_display_table.sort_values(
        by=["Ratings with predictions", "Ratings"], ascending=False
    )
    ratings_table_display_table.columns = [
        "Offer name",
        "Offer type",
        "Customer spend value (£)",
        "Customer reward value (£)",
        "Customer Ratings",
        "Ratings with predictions",
    ]
    ratings_table_display_table = ratings_table_display_table.round(decimals=2)
    ratings_table_display_table.fillna("-", inplace=True)

    return ratings_table_display_table


if __name__ == "__main__":
    user1 = "100006"

    # Train recommender model
    user_offer_matrix = create_user_offer_matrix()
    cf_recommender = CollaborativeFiltering(top_k=3, basis="item")
    cf_recommender.train(user_offer_matrix)

    user_ratings1 = generate_prediction(
        user=user1, recommender=cf_recommender, num_recs=3, perform_mapping=True
    )
    print("Done!")
