import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import scripts.preprocessing as pp

# TODO - Additional plots
# Income distribution
# Age vs Income
# Gender split


def read_and_preprocess():
    data_file_list = []
    data_files = ["portfolio", "profile", "transcript"]

    # Load data files and process them
    for file in data_files:
        try:
            print("Loading cached file:  {}".format(file))
            df_path = "data_cache/{}_pp.csv".format(file)
            df_pp = pp.read_data(file_path=df_path, file_type="csv")
            print("Loaded successfully.")
        except FileNotFoundError:
            print("No cached file for {}, loading from scratch".format(file))
            df_path = "data/{}.json".format(file)
            df = pp.read_data(file_path=df_path, file_type="json")
            df_pp = pp.preprocess_data(df=df, data_name=file)
            df_pp.to_csv("data_cache/{}_pp.csv".format(file), index=False)

        data_file_list.append(df_pp)

    return data_file_list


def return_table():
    profile_path = "data/profile.json"
    profile = pp.read_data(file_path=profile_path, file_type="json")
    profile_pp = pp.preprocess_data(df=profile, data_name="profile")
    table = profile_pp.dropna(how="any")
    table.loc[:, "membership number"] = range(len(table))
    table.loc[:, "membership number"] += 100000
    table.loc[:, "membership start date"] = table.loc[:, "date_joined"].dt.date
    table = table.drop(["id", "year_joined", "became_member_on", "date_joined"], axis=1)

    table["membership start date"] = pd.to_datetime(
        table["membership start date"]
    ).dt.strftime("%Y-%m-%d")

    table = table[
        ["membership number", "gender", "age", "income", "membership start date"]
    ]

    # TODO - change location of data file
    table.to_json(
        "starbucks_analysis_app/static/js/customer_data_table.json", orient="split"
    )

    return table[
        ["membership number", "gender", "age", "income", "membership start date"]
    ]


def return_figures():
    # Load data files and process them
    try:
        print("Loading cached figures")
        figures = joblib.load("data_cache/figures.pkl")
    except FileNotFoundError:
        print("No cached figures. Building from scratch...")
        portfolio_pp, profile_pp, transcript_pp = read_and_preprocess()

        # Graph 1 - Showing all attributes per offer
        graph_one, layout_one = offer_attributes_plot(
            df=portfolio_pp,
            offer_name_col="offer name",
            offer_attribute_cols=["reward", "difficulty", "duration"],
        )

        # Graph 2 - Starbucks app registrations (yearly)
        graph_two, layout_two = membership_join_date_plot(
            df=profile_pp, year_joined_col="year_joined"
        )

        # Graph 3 - Cumulative events over time
        graph_three, layout_three = cumulative_events_plot(
            df=transcript_pp, time_col="time", events_col="event"
        )

        # Graph 4 - Age distribution plot
        graph_four, layout_four = age_distribution_plot(df=profile_pp, age_col="age")

        ####################################################################

        # append all charts
        figures = [
            dict(data=graph_one, layout=layout_one),
            dict(data=graph_two, layout=layout_two),
            dict(data=graph_three, layout=layout_three),
            dict(data=graph_four, layout=layout_four),
        ]

        joblib.dump(figures, "data_cache/figures.pkl")

    return figures


def customer_group_figures(group: int = 1):
    customer_profile_df = pp.read_data(
        file_path="data_cache/profile_post_clustering.csv", file_type="csv"
    )
    customer_group_df = customer_profile_df[
        customer_profile_df["predicted_cluster"] == group
    ]

    gender_graph, gender_layout = gender_plot(df=customer_group_df, gender_col="gender")
    age_graph, age_layout = age_distribution_plot(df=customer_group_df, age_col="age")
    year_joined_graph, year_joined_layout = membership_join_date_plot(
        df=customer_group_df, year_joined_col="year_joined"
    )
    income_graph, income_layout = income_distribution_plot(
        df=customer_group_df, income_col="income"
    )

    num_transactions_graph, num_transactions_layout = distribution_plot(
        df=customer_group_df, col="number_of_transactions"
    )

    avg_transactions_graph, avg_transactions_layout = distribution_plot(
        df=customer_group_df, col="average_transaction_value"
    )

    received_to_viewed_ratio_graph, received_to_viewed_ratio_layout = distribution_plot(
        df=customer_group_df, col="received-to-viewed-ratio"
    )

    (
        received_to_completed_ratio_graph,
        received_to_completed_ratio_layout,
    ) = distribution_plot(df=customer_group_df, col="received-to-completed-ratio")

    group_figures = [
        dict(data=gender_graph, layout=gender_layout),
        dict(data=age_graph, layout=age_layout),
        dict(data=year_joined_graph, layout=year_joined_layout),
        dict(data=income_graph, layout=income_layout),
        dict(data=num_transactions_graph, layout=num_transactions_layout),
        dict(data=avg_transactions_graph, layout=avg_transactions_layout),
        dict(
            data=received_to_viewed_ratio_graph, layout=received_to_viewed_ratio_layout
        ),
        dict(
            data=received_to_completed_ratio_graph,
            layout=received_to_completed_ratio_layout,
        ),
    ]

    return group_figures


def income_distribution_plot(df: pd.DataFrame, income_col: str = "income"):
    # Graph 4 - Age distribution
    graph_four = [
        go.Histogram(x=df[income_col], nbinsx=20, marker_color="rgb(0,206,209)")
    ]

    layout_four = dict(
        title="Income distribution of members",
        xaxis=dict(
            title="Income",
        ),
        yaxis=dict(title="Number of members"),
        bargap=0.05,
    )
    return graph_four, layout_four


def age_distribution_plot(df: pd.DataFrame, age_col: str = "age"):
    # Graph 4 - Age distribution
    graph_four = [
        go.Histogram(x=df[age_col], nbinsx=20, marker_color="#330C73", opacity=0.75)
    ]

    layout_four = dict(
        title="Age distribution of members",
        xaxis=dict(
            title="Age",
        ),
        yaxis=dict(title="Number of members"),
        bargap=0.05,  # gap between bars of adjacent location coordinates
    )
    return graph_four, layout_four


def distribution_plot(df: pd.DataFrame, col: str = "age", marker_color="#330C73"):
    # Graph 4 - Age distribution
    dist_graph = [
        go.Histogram(x=df[col], nbinsx=20, marker_color=marker_color, opacity=0.75)
    ]

    dist_layout = dict(
        title=f"{col} distribution of members",
        xaxis=dict(
            title=col,
        ),
        yaxis=dict(title="Number of members"),
        bargap=0.05,  # gap between bars of adjacent location coordinates
    )
    return dist_graph, dist_layout


def gender_plot(df, gender_col="gender"):
    gender_counts = (
        df[gender_col].value_counts().rename_axis("gender").reset_index(name="counts")
    )

    color = np.array(["rgb(255,255,255)"] * gender_counts.shape[0])
    color[gender_counts["counts"] != gender_counts["counts"].max()] = "rgb(48, 47, 43)"
    color[gender_counts["counts"] == gender_counts["counts"].max()] = "rgb(130,0,0)"

    graph_three = [
        go.Bar(
            x=gender_counts["gender"],
            y=gender_counts["counts"],
            marker=dict(color=color.tolist()),
        )
    ]

    layout_three = dict(
        title="Gender distribution",
        xaxis=dict(
            title="Gender",
        ),
        yaxis=dict(title="Number of members"),
    )

    return graph_three, layout_three


def membership_join_date_plot(df, year_joined_col="year_joined"):
    # Graph 3 - Starbucks app registrations (yearly)
    new_members = (
        df[year_joined_col]
        .value_counts()
        .rename_axis("year")
        .reset_index(name="counts")
    )
    new_members.sort_values(by="year", inplace=True)

    color = np.array(["rgb(255,255,255)"] * new_members.shape[0])
    color[new_members["counts"] != new_members["counts"].max()] = "rgb(48, 47, 43)"
    color[new_members["counts"] == new_members["counts"].max()] = "rgb(130,0,0)"

    graph_three = [
        go.Bar(
            x=new_members["year"],
            y=new_members["counts"],
            marker=dict(color=color.tolist()),
        )
    ]

    layout_three = dict(
        title="Yearly membership growth",
        xaxis=dict(
            title="Year",
        ),
        yaxis=dict(title="Number of new members"),
    )

    return graph_three, layout_three


def cumulative_events_plot(df, time_col="time", events_col="event"):
    graph_six = []
    table_six = df[[time_col, events_col]]
    time_max = table_six[time_col].max()

    for event in table_six[events_col].unique():
        single_event = table_six[table_six[events_col] == event]
        event_count = single_event.groupby(time_col)[events_col].count()
        x_arr = np.array(event_count.index)
        y_arr = np.cumsum(event_count.values)
        if time_max != event_count.index.max():
            x_arr = np.append(x_arr, time_max)
            y_arr = np.append(y_arr, y_arr[(len(y_arr) - 1)])

        graph_six.append(go.Scatter(x=x_arr, y=y_arr, mode="lines+markers", name=event))

    layout_six = dict(
        title="Transaction activity over time",
        xaxis=dict(
            title="Time",
        ),
        yaxis=dict(title="Event count"),
    )
    return graph_six, layout_six


def offer_attributes_plot(
    df: pd.DataFrame,
    offer_name_col: str = "offer name",
    offer_attribute_cols=["reward", "difficulty", "duration"],
):
    graph_two = []
    col_list = offer_attribute_cols

    for col in col_list:
        graph_two.append(
            go.Bar(
                name=col,
                x=df[offer_name_col],
                y=df[col],
            )
        )

    layout_two = dict(
        title="Offer details",
        xaxis=dict(
            title="Offer",
        ),
        yaxis=dict(title="Value"),
    )
    return graph_two, layout_two


if __name__ == "__main__":
    # return_figures()
    return_table()
