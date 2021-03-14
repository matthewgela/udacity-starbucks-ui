import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import scripts.data as d

portfolio_pp, profile_pp, transcript_pp = d.read_and_preprocess()

# Preparing the customer profile data for clustering
select_columns = ["gender", "age", "income", "year_joined"]
df_to_cluster = profile_pp[select_columns].dropna(how="any")
df_to_cluster.to_csv("data_cache/profile_for_clustering.csv")

# Setting up the pipeline for KMeans clustering
numeric_features = df_to_cluster.select_dtypes(include=["int64", "float64"]).columns
categorical_features = df_to_cluster.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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

# Initialising K-Means
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# Finding a good choice for the number of clusters, k, to use
elbow_method = False

if elbow_method:
    sse = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("kmeans", kmeans)])
        pipe.fit(df_to_cluster)
        sse.append(pipe.named_steps.kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 8), sse)
    plt.xticks(range(1, 8))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

# Elbow was last found at k = 4
best_k = 4

# Fitting the pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=best_k))]
)
pipeline.fit(df_to_cluster)

# Predictions
pipe_pred = pipeline.predict(df_to_cluster)
df_to_cluster["predicted_cluster"] = pipe_pred

# Exporting the KMeans Model
joblib.dump(pipeline, "models/customer_kmeans.joblib")

df_to_cluster.to_csv("data_cache/profile_post_clustering.csv")

print("Done")
