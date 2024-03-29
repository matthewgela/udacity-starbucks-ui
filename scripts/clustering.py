import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import scripts.data as d


def prepare_data_for_clustering(profile_pp, transcript_pp):
    events_binary = pd.get_dummies(transcript_pp["event"])
    transcript_comb = pd.concat([transcript_pp, events_binary], axis=1)

    # Aggregating to person level
    # Summing the total number of offers a user has received, viewed and completed
    user_actions = transcript_comb.groupby(["person"]).agg(
        num_offer_completed_sum=pd.NamedAgg(column="offer completed", aggfunc="sum"),
        num_offer_viewed_sum=pd.NamedAgg(column="offer viewed", aggfunc="sum"),
        num_offer_received_sum=pd.NamedAgg(column="offer received", aggfunc="sum"),
    )
    user_actions["received-to-viewed-ratio"] = 100 * (
        user_actions["num_offer_viewed_sum"] / user_actions["num_offer_received_sum"]
    )
    user_actions["received-to-completed-ratio"] = 100 * (
        user_actions["num_offer_completed_sum"] / user_actions["num_offer_received_sum"]
    )

    # # Average transaction amount
    user_transactions_summary = (
        transcript_comb[transcript_comb["amount"].notnull()]
        .groupby(["person"])
        .agg(
            average_transaction_value=pd.NamedAgg(column="amount", aggfunc=np.mean),
            number_of_transactions=pd.NamedAgg(column="amount", aggfunc="count"),
        )
    )

    # Adding transaction and offer activity aggregated at a user level to the demographic information
    # we already have to add more information for the clustering algorithm to use

    profile_with_offer_activity = profile_pp.merge(
        user_actions[["received-to-viewed-ratio", "received-to-completed-ratio"]],
        how="left",
        left_on="id",
        right_index=True,
    )
    profile_with_all_activity = profile_with_offer_activity.merge(
        user_transactions_summary, how="left", left_on="id", right_index=True
    )

    # Preparing the customer profile data for clustering
    col_drop_list = ["id", "became_member_on", "date_joined"]
    profile_with_all_activity = profile_with_all_activity.drop(
        col_drop_list, axis=1, errors="raise"
    )
    df_to_cluster = profile_with_all_activity.dropna(how="any")

    return df_to_cluster


def pipeline_preprocessor(df):
    # Setting up the pipeline for KMeans clustering
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_pipeline(preprocessor, clustering_algorithm):
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("clustering", clustering_algorithm)]
    )
    return pipeline


def train(model, df):
    model.fit(df)


def find_best_k(preprocessor, kmeans_kwargs, plot=True, k_min=1, k_max=8):
    sse = []
    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        pipe = build_pipeline(preprocessor, clustering_algorithm=kmeans)
        # pipe = Pipeline(steps=[("preprocessor", preprocessor), ("kmeans", kmeans)])
        train(pipe, df_to_cluster)
        # pipe.fit(df_to_cluster)
        sse.append(pipe.named_steps.clustering.inertia_)

    if plot:
        plt.style.use("fivethirtyeight")
        plt.plot(range(k_min, k_max), sse)
        plt.xticks(range(k_min, k_max))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

    return sse


def add_predictions_to_df(model, df, prediction_col="predicted_cluster"):
    if hasattr(model, "predict"):
        df[prediction_col] = model.predict(df)
    else:
        AttributeError("Model predictions not found")
    return df


def export_trained_model(model, file_prefix="customer_kmeans"):
    joblib.dump(model, f"models/{file_prefix}.joblib")


def export_clustered_data(df, file_prefix="profile_post_clustering"):
    df.to_csv(f"data_cache/{file_prefix}.csv", index=False)


def output_cluster_summaries(df, cluster_col="predicted_cluster", output_df=True):
    cluster_summaries = []
    for cluster_ind in range(df[cluster_col].nunique()):
        summary_df = df_to_cluster[df_to_cluster[cluster_col] == cluster_ind].describe(
            include="all"
        )
        cluster_summaries.append(summary_df)
        if output_df:
            summary_df.to_csv(
                f"data_cache/cluster_{cluster_ind}_summary.csv", index=False
            )

    return cluster_summaries


if __name__ == "__main__":

    #  ----------------- Initialising parameters -------------------- #

    # Initialising K-Means
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    # Algorithm choice
    algorithm = "kmeans"

    # Finding a good choice for the number of clusters, k, to use
    elbow_method = True

    #  ----------------- Loading and preparing data -------------------- #

    # Load data files
    portfolio_pp, profile_pp, transcript_pp = d.read_and_preprocess()

    # Prepare data for clustering
    df_to_cluster = prepare_data_for_clustering(profile_pp, transcript_pp)

    #  ----------------- Machine Learning begins here ------------------- #

    # Creating the pipeline for training
    preprocessor = pipeline_preprocessor(df_to_cluster)

    # Setting best model and parameters
    if algorithm == "kmeans":
        if elbow_method:
            # Finding the optimal number of clusters to use via grid search
            sse = find_best_k(preprocessor, kmeans_kwargs, plot=True, k_max=20)
        best_k = 7
        clustering_algorithm = KMeans(n_clusters=best_k)

    # Build model
    best_model = build_pipeline(preprocessor, clustering_algorithm)

    # Train model
    train(best_model, df_to_cluster)

    # Dataframe updated with predicted cluster
    df_to_cluster = add_predictions_to_df(best_model, df_to_cluster)

    # Exporting model and datasets
    cluster_summaries = output_cluster_summaries(df_to_cluster)
    export_trained_model(best_model)
    export_clustered_data(df_to_cluster)

    print("Done")
