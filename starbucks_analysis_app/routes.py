import json

import joblib
import pandas as pd
import plotly
from flask import render_template, request

from scripts.data import customer_group_figures, return_figures, return_table
from starbucks_analysis_app import app

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

    # TABLE
    df = return_table()

    if request.method == "GET":
        return render_template(
            "index.html",
            ids=ids,
            figuresJSON=figuresJSON,
            column_names=df.columns.values,
            row_data=list(df.values.tolist()),
            # link_column="person",
            zip=zip,
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
