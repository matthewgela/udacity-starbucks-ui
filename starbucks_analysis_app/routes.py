import json

import joblib
import pandas as pd
import plotly
from flask import render_template, request

from scripts.data import customer_group_figures, return_figures
from scripts.predictions import generate_prediction
from scripts.recommender import (  # ContentBasedFiltering,; create_content_table,
    CollaborativeFiltering,
    create_user_offer_matrix,
)
from starbucks_analysis_app import app

# Train recommender model
user_offer_matrix = create_user_offer_matrix()
cf_recommender = CollaborativeFiltering(
    n_sim=15, basis="user", similarity_method="jaccard"
)
cf_recommender.train(user_offer_matrix, compute_similarity_matrix=False)

# # ContentBasedFiltering
# content_table = create_content_table(basis="item")
# cbf_recommender = ContentBasedFiltering(n_sim=2, basis="item", similarity_method="jaccard")
# cbf_recommender.train(user_offer_matrix, content_table)

# Recommender to use
recommender = cf_recommender

# Load clustering model
model = joblib.load("models/customer_kmeans.joblib")


@app.route("/", methods=["POST", "GET"])
@app.route("/cover", methods=["POST", "GET"])
def cover():
    if request.method == "GET":
        return render_template("cover.html")


@app.route("/index", methods=["POST", "GET"])
def index():
    # PLOTS
    figures = return_figures()

    # plot ids for the html id tag
    ids = ["figure-{}".format(i + 1) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    if request.method == "GET":
        return render_template(
            "index.html",
            ids=ids,
            figuresJSON=figuresJSON,
        )


@app.route("/customer", methods=["POST", "GET"])
def customer():
    if request.method == "GET":
        # Generating figures for predicted group
        figures = customer_group_figures(group=0)

        # plot ids for the html id tag
        ids = ["figure-{}".format(i + 1) for i, _ in enumerate(figures)]

        # Convert the plotly figures to JSON for javascript in html template
        figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("customer.html", ids=ids, figuresJSON=figuresJSON)

    if request.method == "POST":
        # Extract the input
        columns = [
            "gender",
            "age",
            "income",
            "year_joined",
            "average_transaction_value",
            "number_of_transactions",
            "received-to-viewed-ratio",
            "received-to-completed-ratio",
        ]
        print("customer form", request.form)
        form_values = [request.form[col] for col in columns]
        # Make DataFrame for model
        input_variables = pd.DataFrame([form_values], columns=columns)
        # Get the model's prediction
        prediction = model.predict(input_variables)[0]

        # Generating figures for predicted group
        figures = customer_group_figures(group=prediction)

        # plot ids for the html id tag
        ids = ["figure-{}".format(i + 1) for i, _ in enumerate(figures)]

        # Convert the plotly figures to JSON for javascript in html template
        figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

        # So prediction is never zero
        prediction += 1

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return render_template(
            "customer.html",
            original_input=dict(zip(columns, form_values)),
            result=prediction,
            ids=ids,
            figuresJSON=figuresJSON,
        )


@app.route("/recommendation", methods=["POST", "GET"])
def recommendation():

    # Render page
    if request.method == "GET":
        return render_template(
            "recommendation.html",
        )

    if request.method == "POST":

        data = request.form
        print(data)

        # Generate recommendation

        ratings_table = generate_prediction(
            user=data["id"],
            recommender=recommender,
            num_recs=3,
            perform_mapping=True,
        )

        top_3_predictions = list(
            ratings_table[ratings_table["Customer Ratings"] == "-"]["Offer name"][
                :3
            ].values
        )

        # return data['id']
        return render_template(
            template_name_or_list="recommendation.html",
            column_names_selected=ratings_table.columns.values,
            row_data_selected=list(ratings_table.values.tolist()),
            zip=zip,
            selected=data["id"],
            recommended_offers=top_3_predictions,
        )
