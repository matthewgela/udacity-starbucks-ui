import joblib
import numpy as np
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
            df_pp.to_csv("data_cache/{}_pp.csv".format(file))

        data_file_list.append(df_pp)

    return data_file_list


def return_table():
    profile_path = "data/profile.json"
    profile = pp.read_data(file_path=profile_path, file_type="json")
    profile_pp = pp.preprocess_data(df=profile, data_name="profile")
    table = profile_pp.dropna(how="any")
    table["membership number"] = range(len(table))
    table["membership number"] += 100000
    table["membership start date"] = table["date_joined"].dt.date
    table = table.drop(["id", "year_joined", "became_member_on", "date_joined"], axis=1)
    return table[
        ["membership number", "gender", "age", "income", "membership start date"]
    ].head(20)


def return_figures():
    # Load data files and process them
    try:
        print("Loading cached figures")
        figures = joblib.load("data_cache/figures.pkl")
    except FileNotFoundError:
        print("No cached figures. Building from scratch...")
        portfolio_pp, profile_pp, transcript_pp = read_and_preprocess()
        # # Graph 1 - Difficulty vs Reward
        # graph_one = []
        # x_val = portfolio_pp["difficulty"]
        # y_val = portfolio_pp["reward"]
        # graph_one.append(go.Scatter(x=x_val, y=y_val, mode="markers"))
        #
        # layout_one = dict(
        #     title="Reward value of offer versus difficulty of redeeming offer",
        #     xaxis=dict(title="Difficulty", range=[0, portfolio_pp["difficulty"].max() + 2]),
        #     yaxis=dict(title="Reward", range=[0, portfolio_pp["reward"].max() + 2]),
        # )

        # Graph 2 - Showing all attributes per offer
        graph_two = []
        col_list = ["reward", "difficulty", "duration"]

        for col in col_list:
            graph_two.append(
                go.Bar(
                    name=col,
                    x=portfolio_pp["offer name"],
                    y=portfolio_pp[col],
                )
            )

        layout_two = dict(
            title="Offer details",
            xaxis=dict(
                title="Offer",
            ),
            yaxis=dict(title="Value"),
        )

        # Graph 3 - Starbucks app registrations (yearly)
        new_members = (
            profile_pp["year_joined"]
            .value_counts()
            .rename_axis("year")
            .reset_index(name="counts")
        )
        new_members.sort_values(by="year", inplace=True)

        color = np.array(["rgb(255,255,255)"] * new_members.shape[0])
        color[new_members["counts"] < 4000] = "rgb(255,128,0)"
        color[new_members["counts"] >= 4000] = "rgb(130,0,0)"

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

        # Graph 4 - Age distribution
        graph_four = [go.Histogram(x=profile_pp["age"], nbinsx=20)]

        layout_four = dict(
            title="Age distribution of members",
            xaxis=dict(
                title="Age",
            ),
            yaxis=dict(title="Number of members"),
        )

        # Graph 5 - Events comparison

        events = (
            transcript_pp["event"]
            .value_counts()
            .rename_axis("event")
            .reset_index(name="counts")
        )
        events.sort_values(by="event", inplace=True)

        g5_color = np.array(["rgb(255,255,255)"] * events.shape[0])
        g5_color[events["counts"] < 60000] = "rgb(153,255,153)"
        g5_color[events["counts"] >= 60000] = "rgb(0,204,0)"

        # graph_five = [
        #     go.Bar(
        #         x=events["event"], y=events["counts"], marker=dict(color=g5_color.tolist())
        #     )
        # ]
        #
        # layout_five = dict(
        #     title="Offer activity breakdown",
        #     xaxis=dict(
        #         title="Event",
        #     ),
        #     yaxis=dict(title="Number of occurrences"),
        # )

        # Graph 6 - Cumulative events over time
        graph_six = []
        table_six = transcript_pp[["time", "event"]]
        time_max = table_six["time"].max()

        for event in table_six["event"].unique():
            single_event = table_six[table_six["event"] == event]
            event_count = single_event.groupby("time")["event"].count()
            x_arr = np.array(event_count.index)
            y_arr = np.cumsum(event_count.values)
            if time_max != event_count.index.max():
                x_arr = np.append(x_arr, time_max)
                y_arr = np.append(y_arr, y_arr[(len(y_arr) - 1)])

            graph_six.append(
                go.Scatter(x=x_arr, y=y_arr, mode="lines+markers", name=event)
            )

        layout_six = dict(
            title="Transaction activity over time",
            xaxis=dict(
                title="Time",
            ),
            yaxis=dict(title="Event count"),
        )

        ####################################################################

        # append all charts
        figures = []
        # figures.append(dict(data=graph_one, layout=layout_one))
        figures.append(dict(data=graph_two, layout=layout_two))
        figures.append(dict(data=graph_three, layout=layout_three))
        figures.append(dict(data=graph_four, layout=layout_four))
        figures.append(dict(data=graph_six, layout=layout_six))

        joblib.dump(figures, "data_cache/figures.pkl")

    return figures


if __name__ == "__main__":
    return_figures()
