import joblib

import scripts.data as d

portfolio_pp, profile_pp, transcript_pp = d.read_and_preprocess()

# Preparing the customer profile data for clustering
select_columns = ["gender", "age", "income", "year_joined"]
df_to_cluster = profile_pp[select_columns].dropna(how="any")

model = joblib.load("models/customer_kmeans.joblib")

model.predict(df_to_cluster.iloc[0])
