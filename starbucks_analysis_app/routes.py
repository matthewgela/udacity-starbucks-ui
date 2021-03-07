import json

import joblib
import pandas as pd
import plotly
from flask import render_template, request

from scripts.data import return_figures, return_table
from starbucks_analysis_app import app

model = joblib.load("models/customer_kmeans.joblib")


@app.route("/", methods=["POST", "GET"])
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
        return render_template("customer.html")
    if request.method == "POST":
        # Extract the input
        columns = ["gender", "age", "income", "year_joined"]
        gender, age, income, year_joined = [request.form[col] for col in columns]
        # Make DataFrame for model
        input_variables = pd.DataFrame(
            [[gender, age, income, year_joined]], columns=columns
        )
        # Get the model's prediction
        prediction = model.predict(input_variables)[0]

        # So prediction is never zero
        prediction += 1

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return render_template(
            "customer.html",
            original_input={
                "Gender": gender,
                "Age": age,
                "Income": income,
                "Year joined": year_joined,
            },
            result=prediction,
        )
