import json

import plotly
from flask import render_template

from scripts.data import return_figures, return_table
from starbucks_analysis_app import app


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

    # return render_template("index.html", ids=ids, figuresJSON=figuresJSON)
    return render_template(
        "index2.html",
        ids=ids,
        figuresJSON=figuresJSON,
        column_names=df.columns.values,
        row_data=list(df.values.tolist()),
        # link_column="person",
        zip=zip,
    )
