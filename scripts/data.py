import numpy as np
import plotly.graph_objs as go

import scripts.preprocessing as pp

# TODO - Additional plots
# Income distribution
# Age vs Income
# Gender split


def return_figures():
    # Load data files and process them
    portfolio_path = "data/portfolio.json"
    profile_path = "data/profile.json"
    transcript_path = "data/transcript.json"

    portfolio = pp.read_data(file_path=portfolio_path, file_type="json")
    profile = pp.read_data(file_path=profile_path, file_type="json")
    transcript = pp.read_data(file_path=transcript_path, file_type="json")

    portfolio_pp = pp.preprocess_data(df=portfolio, data_name="portfolio")
    profile_pp = pp.preprocess_data(df=profile, data_name="profile")
    transcript_pp = pp.preprocess_data(df=transcript, data_name="transcript")

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
        title="Offer attributes (Reward, Difficulty and Duration)",
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
        title="Starbucks app new members per year",
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

    graph_five = [
        go.Bar(
            x=events["event"], y=events["counts"], marker=dict(color=g5_color.tolist())
        )
    ]

    layout_five = dict(
        title="Starbucks app activity breakdown",
        xaxis=dict(
            title="Event",
        ),
        yaxis=dict(title="Number of occurrences"),
    )

    ####################################################################

    # append all charts
    figures = []
    # figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))
    figures.append(dict(data=graph_five, layout=layout_five))

    return figures


if __name__ == "__main__":
    return_figures()


# events = (
#     transcript_pp["event"]
#         .value_counts()
#         .rename_axis("event")
#         .reset_index(name="counts")
# )
# events.sort_values(by="event", inplace=True)
#
#
# graph_five = [go.Bar(x=events["event"], y=events["count"])]
#
# layout_five = dict(
#     title="Starbucks app activity breakdown",
#     xaxis=dict(
#         title="Event",
#     ),
#     yaxis=dict(title="Number of occurrences"),
# )
#
# figures.append(dict(data=graph_five, layout=layout_five))
