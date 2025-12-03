# %% ----------------------------------------
# -------------------------------------------

# issues_analysis.py

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from IPython.display import HTML, display
from pandas.tseries.holiday import USFederalHolidayCalendar

DATAPATH = os.path.join("issues_metrics", "raw_issues_data.pkl")

REPORTPATH = os.path.join(
    "issues_metrics",
    "issues_report.pkl",
)

# %% ----------------------------------------
# -------------------------------------------


# function to style html tables
# ------------------------------
def render_html_table(df):
    display(HTML(df.to_html(classes="wrapped-table", index=False)))
    display(
        HTML("""
    <style>
    .wrapped-table {
        table-layout: fixed;
        width: auto;
        word-wrap: break-word;
    }
    .wrapped-table td, .wrapped-table th {
        white-space: normal;
    }
    </style>
    """)
    )


# function to peek at head and tail
# ------------------------------
def peek(df, n=1):
    head = df.head(n)
    tail = df.tail(n)
    out = pd.concat([head, tail])
    return out


# %% ----------------------------------------
# Preprocessing
# -------------------------------------------


def process_datetime(df, date_cols):
    for col in date_cols:
        if "date_time" in col:
            name = col.replace("date_time", "datetime_opened")
        else:
            name = col.replace("date", "datetime")

        df[name] = pd.to_datetime(
            df[col],
            utc=True,
        )
        df[name] = df[name].dt.tz_convert("US/Eastern")
        df[name] = df[name].dt.tz_localize(None)
        df[col] = df[name].dt.date

    # use timezone-corrected datetime_opened for date_opened
    df["date_opened"] = df["datetime_opened"].dt.date

    return df


def preprocess(
    df,
    dev_usernames,
    start_date=False,
    end_date=False,
):
    df = process_datetime(
        df,
        [
            "date_time",  # -> datetime_opened
            "date_closed",  # + datetime_closed
            "comment_date",  # + comment_datetime
        ],
    )

    # filter on start_date
    # ------------------------------
    if isinstance(pd.to_datetime(start_date), pd.Timestamp):
        df = df.loc[
            df["date_opened"] >= pd.to_datetime(start_date).date()
        ]


    def assign_dev(row):
        if row in dev_usernames:
            return "developer"
        else:
            return "non-developer"

    df["opened_by"] = df["username"].apply(lambda x: assign_dev(x))

    # drop rows where username is
    # "github-actions[bot]"
    # ------------------------------
    df = df[df["username"] != "github-actions[bot]"]
    df = df.reset_index(drop=True)

    # adjust report for the specified "end_date",
    # removing any dates after "end_date"
    # ------------------------------
    if isinstance(pd.to_datetime(end_date), pd.Timestamp):
        end_date = pd.to_datetime(end_date).date()

        # remove issues opened after end_date
        # ------------------------------
        # - get issues numbers
        issues_to_remove = df.loc[df["date_opened"] >= end_date]["number"].unique()

        # - remove issues based on number
        df = df.loc[~df["number"].isin(issues_to_remove)].reset_index(drop=True)

        # clear fields for issues closed
        # after the end_date
        # ------------------------------

        invalid_dateclosed = df.loc[
            pd.to_datetime(df["datetime_closed"]).dt.date >= end_date
        ]["number"].unique()

        for issue_num in invalid_dateclosed:
            # set date_closed and datetime_closed to NaT
            df.loc[
                df["number"] == issue_num,
                ["date_closed", "datetime_closed"],
            ] = pd.NaT
            # set closed_by to ""
            df.loc[
                df["number"] == issue_num,
                "closed_by",
            ] = ""

        # clear fields for *only* the
        # comments made after the end_date
        # ------------------------------
        invalid_commentdate = pd.to_datetime(df["comment_datetime"]).dt.date >= end_date

        # set comment_date and comment_datetime to NaT
        # set comment_username and comment_contents to ""
        df.loc[
            invalid_commentdate,
            [
                "comment_date",
                "comment_datetime",
                "comment_username",
                "comment_contents",
            ],
        ] = [pd.NaT, pd.NaT, "", ""]

    # order columns
    # ------------------------------
    df = df[
        [
            "number",
            # "labels",
            # "milestone",
            "date_opened",
            "datetime_opened",
            "opened_by",
            "username",
            "issue_name",
            "issue_url",
            "date_closed",
            "datetime_closed",
            "closed_by",
            "comment_date",
            "comment_datetime",
            "comment_username",
            "comment_contents",
        ]
    ]

    return df


# %% ----------------------------------------
# -------------------------------------------


# number of issues by by username
# ------------------------------
def issues_by_user(
    df,
    show=True,
    return_df=False,
):
    issues_by_user = df[
        [
            "issue_name",
            "date_opened",
            "username",
        ]
    ].drop_duplicates()

    issues_by_user = issues_by_user.groupby("username").count().reset_index()

    issues_by_user = issues_by_user[
        [
            "username",
            "issue_name",
        ]
    ].rename(
        columns={
            "issue_name": "issues_opened",
        }
    )

    if show:
        display(
            pd.concat(
                [
                    issues_by_user,
                    pd.DataFrame(
                        {
                            "username": ["Total"],
                            "issues_opened": [issues_by_user["issues_opened"].sum()],
                        }
                    ),
                ]
            ).reset_index(drop=True)
        )

    if return_df:
        return issues_by_user

    return


# issues_by_user(df)

# %% -------------------------------------
# Get issues closed / merged
# ----------------------------------------


def issue_status_counts(
    data,
):
    df = data.copy()
    df = df[
        [
            "number",
            "date_closed",
        ]
    ]
    df = df.drop_duplicates().reset_index(drop=True)

    total = len(df["number"])
    closed_issues = df["date_closed"].notna().sum()
    outstanding_issues = total - closed_issues

    table = pd.DataFrame(
        {
            "Issue Status": [
                "New Issues",
                "Outstanding Issues",
                "Closed Issues",
            ],
            "Count": [
                total,
                outstanding_issues,
                closed_issues,
            ],
            "Percent": [
                100,
                round(outstanding_issues / total * 100, 2),
                round(closed_issues / total * 100, 2),
            ],
        }
    )

    return table


# %% -------------------------------------
# Get issues opened by users
# ----------------------------------------


def issues_opened_by_users(
    df,
    by_dev_status=False,
    return_df=False,
):
    if by_dev_status:
        by_col = "opened_by"
    else:
        by_col = "username"

    issues_by_user = df[
        [
            "issue_name",
            "date_opened",
            by_col,
        ]
    ].drop_duplicates()

    issues_by_user = issues_by_user.groupby(by_col).count().reset_index()
    issues_by_user = issues_by_user[
        [
            by_col,
            "issue_name",
        ]
    ].rename(
        columns={
            "issue_name": "issues_opened",
        }
    )

    table = pd.concat(
        [
            issues_by_user,
            pd.DataFrame(
                {
                    by_col: ["Total"],
                    "issues_opened": [issues_by_user["issues_opened"].sum()],
                }
            ),
        ]
    ).reset_index(drop=True)

    if return_df:
        return issues_by_user

    return table


# %% -------------------------------------
# Get Time to Response Metric
# ----------------------------------------


def process_issues_for_ttr(
    df,
    report_date,
):
    # unique non-bot issues
    # ------------------------------
    unique_issues = df.drop_duplicates(
        [
            "issue_name",
            "date_opened",
            "username",
        ]
    )

    # issues w/o comments
    # ------------------------------
    no_response = (
        unique_issues[
            unique_issues["comment_datetime"].apply(
                lambda x: not isinstance(
                    x,
                    pd.Timestamp,
                )
            )
        ]
        .reset_index(drop=True)
        .copy()
    )

    no_response["status"] = "no response"

    # get issues w/ comments
    # ------------------------------
    with_response = (
        unique_issues[
            unique_issues["comment_datetime"].apply(
                lambda x: isinstance(
                    x,
                    pd.Timestamp,
                )
            )
        ]
        .reset_index(drop=True)
        .copy()
    )
    unique_issues_with_response = list(with_response["number"].unique())

    # create indicator var for when username != comment_username
    # ------------------------------
    def assign_external_response(row):
        """
        Function to compare username and comment_username rows
        of a DataFrame and return 1 if they are not equal, else 0.
        """
        if row["username"] != row["comment_username"]:
            return 1
        else:
            return 0

    with_response["ext_response"] = with_response.apply(
        lambda x: assign_external_response(x), axis=1
    )

    # check that all records have valid ext_response value
    # ------------------------------
    if not set(with_response["ext_response"].unique()).issubset({0, 1}):
        raise ValueError("ext_response column should only contain 0 and 1")
    else:
        pass

    # split into two dataframes on ext_response indicator
    # ------------------------------
    external_responses_all = with_response[with_response["ext_response"] == 1].copy()
    without_ext = with_response[with_response["ext_response"] == 0].copy()

    # sort by id, date with oldest dates first
    external_responses_all = external_responses_all.sort_values(
        ["number", "comment_date"],
        ascending=[False, True],
    )
    external_responses_all = external_responses_all.reset_index(drop=True)

    # unique issues with an external response, keeping the first response instance
    # ------------------------------
    external_response = (
        external_responses_all.drop_duplicates(["number"]).reset_index(drop=True).copy()
    )
    external_response["drop"] = 1

    # unique issues with only self comments
    # ------------------------------
    self_response = without_ext.join(
        external_response[["number", "drop"]].set_index("number"),
        on="number",
        how="left",
    )

    # drop rows where top is 1
    # ------------------------------
    self_response = self_response[self_response["drop"] != 1].copy()
    self_response = self_response.drop_duplicates(["number"])
    self_response = self_response.reset_index(drop=True)

    cols_to_remove = ["ext_response", "drop"]

    self_response = self_response.drop(
        columns=cols_to_remove,
    )
    external_response = external_response.drop(
        columns=cols_to_remove,
    )

    self_response["status"] = "self comment"
    external_response["status"] = "external comment"

    # check that all records are accounted for after manipulations
    # ------------------------------
    if not len(self_response["number"]) + len(external_response["number"]) == len(
        unique_issues_with_response
    ):
        raise ValueError(
            "Number of unique issues with a response has changed,"
            " which indicates a problem with the data processing."
            " Please check the code and try again."
        )
    else:
        pass

    # confirm issue counts are correct after segmentation
    # ----------------------------------------
    if not len(self_response["number"]) + len(external_response["number"]) + len(
        no_response["number"]
    ) == len(unique_issues):
        raise ValueError(
            "Number of unique issues has changed after segmentation,"
            " which indicates a problem with the data processing."
            " Please check the code and try again."
        )
    else:
        pass

    # Determine time-to-respond metric
    # ------------------------------

    issues_segmented = pd.concat(
        [
            no_response,
            self_response,
            external_response,
        ]
    )

    issues_segmented = issues_segmented.sort_values("number", ascending=False)

    def assign_ttr_date(row, report_date):
        """
        Function to assign a date to use for time-to-respond metric based on
        the status of the issue.
        """
        # format report date
        report_date = pd.to_datetime(report_date)

        # assign ttr_date based on status
        if row["status"] == "no response":
            if pd.notnull(row["datetime_closed"]):
                return row["datetime_closed"]
            else:
                return report_date
        elif row["status"] == "self comment":
            if pd.notnull(row["datetime_closed"]):
                return row["datetime_closed"]
            else:
                return report_date
        elif row["status"] == "external comment":
            if pd.notnull(row["datetime_closed"]):
                # return whichever is earliest between datetime_closed
                # and comment_datetime
                return min(
                    pd.to_datetime(row["datetime_closed"]),
                    pd.to_datetime(row["comment_datetime"]),
                )
            else:
                if not isinstance(row["comment_datetime"], pd.Timestamp):
                    print("\n--- BAD TYPE DETECTED ---")
                    print(f"number: {row['number']}, status: {row['status']}")
                    print("comment_datetime type:", type(row["comment_datetime"]))
                    print("comment_datetime value:", row["comment_datetime"])
                return row["comment_datetime"]
        else:
            raise ValueError(
                "Invalid status value. Expected 'no response', 'self comment',"
                " or 'external comment'."
            )

    issues_segmented["ttr_date"] = issues_segmented.apply(
        lambda x: assign_ttr_date(
            x,
            report_date,
        ),
        axis=1,
    )

    # Compute time elapsed in business days
    # ------------------------------
    def business_hours_elapsed(df):
        # build holiday set
        start_holiday = df["datetime_opened"].min().floor("D")
        end_holiday = df["ttr_date"].max().floor("D")

        cal = USFederalHolidayCalendar()
        holidays = set(cal.holidays(start=start_holiday, end=end_holiday).date)

        def calc(row):
            start = row["datetime_opened"]
            end = row["ttr_date"]
            raw_elapsed = end - start

            all_dates = pd.date_range(
                start=start.floor("D"), end=end.floor("D"), freq="D"
            ).date

            days_to_exclude = {
                d for d in all_dates if d.weekday() >= 5 or d in holidays
            }
            business_delta = raw_elapsed - timedelta(days=len(days_to_exclude))
            business_delta = max(business_delta, timedelta(0))
            return round(business_delta.total_seconds() / 3600, 1)

        df["ttr_hours"] = df.apply(calc, axis=1)
        return df

    issues_segmented = business_hours_elapsed(issues_segmented)
    issues_segmented["ttr_days"] = round(issues_segmented["ttr_hours"] / 24, 2)

    return issues_segmented


# %% ---------------------------
# Time To Response Table
# ------------------------------


def generate_ttr_table(data):
    df = data.copy()

    def ttr_indicator(row):
        days = round(row["ttr_hours"] / 24, 2)

        if days <= 2:
            return 0
        elif (days > 2) and (days <= 14):
            return 1
        elif (days > 14) and (days <= 30):
            return 2
        elif (days > 30) and (days <= 90):
            return 3
        elif days > 90:
            return 4
        else:
            raise (
                ValueError(
                    f"Invalid input: {days} cannot be mapped to the specified bins."
                )
            )

    ttr_bins = {
        0: "< 02 days",
        1: "03 - 14 days",
        2: "15 - 30 days",
        3: "31 - 90 days",
        4: "> 90 days",
    }

    df["ttr_indicator"] = df.apply(lambda x: ttr_indicator(x), axis=1)

    def percent_bins_table(df, indicator_column="ttr_indicator"):
        if indicator_column not in df.columns:
            raise ValueError(
                f"Columns {indicator_column} not found in dataframe columns"
            )

        ttr_percent_bins = round(
            df[indicator_column].value_counts() / len(df[indicator_column]) * 100, 2
        )

        ttr_percent_bins = (
            ttr_percent_bins.reset_index()
            .sort_values(indicator_column)
            .reset_index(drop=True)
        )

        ttr_percent_bins = ttr_percent_bins.rename(columns={"count": "percent"})

        ttr_percent_bins["bins"] = ttr_percent_bins[indicator_column].map(ttr_bins)

        ttr_percent_bins = ttr_percent_bins[["bins", "percent"]]

        ttr_percent_bins["cumulative_percent"] = ttr_percent_bins["percent"].cumsum()

        return ttr_percent_bins

    ttr_issues_table = percent_bins_table(df)

    ttr_issues_table = ttr_issues_table.rename(
        columns={
            "bins": "Time Window",
            "percent": "Percent",
            "cumulative_percent": "Cumulative Percent",
        }
    )

    return ttr_issues_table


# %% ---------------------------
# Process report data for saving
# ------------------------------


def prep_report_data_for_saving(
    start_date,
    report_date,
    issues_status,
    opened_by_status,
    ttr_issues,
    nondev_ttr_issues,
    overwrite_historical_data=False,
):
    # load historical data and check for report_date
    # ----------------------------------------
    report_exists = os.path.exists(REPORTPATH)

    if report_exists:
        with open(REPORTPATH, "rb") as f:
            hist_report_data = pickle.load(f)

        hist_report_data["date_range"] = hist_report_data.apply(
            lambda x: f"{x['start_date']}_{x['report_date']}",
            axis=1,
        )

        report_date_range = f"{start_date}_{report_date}"

        if report_date_range in hist_report_data["date_range"].unique():
            if overwrite_historical_data:
                print(
                    f"Report data for the range {start_date} to {report_date} "
                    "already exists and will be overwritten."
                )
                # drop existing data for report_date
                hist_report_data = hist_report_data.loc[
                    hist_report_data["report_date"] != f"{report_date}"
                ]
            else:
                raise ValueError(
                    f"Report data for the range {start_date} to {report_date}"
                    "already exists."
                    " Since overwrite_historical_data is False, the"
                    " saving cannot proceed."
                )

    # issues_status metric
    # ----------------------------------------
    issues_status["report_date"] = f"{report_date}"
    issues_status["start_date"] = f"{start_date}"
    issues_status["metric"] = "issues_status"
    issues_status["indicator_name"] = "open_status"
    issues_status["value_type"] = "count"
    issues_status["sub_value_type"] = "cumulative_percent"

    issues_status = issues_status.rename(
        columns={
            "Issue Status": "indicator_value",
            "Count": "value",
            "Percent": "sub_value",
        }
    )

    # opened_by_dev_status metric
    # ----------------------------------------
    opened_by_status["report_date"] = f"{report_date}"
    opened_by_status["start_date"] = f"{start_date}"
    opened_by_status["metric"] = "opened_by_dev_status"
    opened_by_status["indicator_name"] = "opened_by"
    opened_by_status["value_type"] = "count"
    opened_by_status["sub_value_type"] = "NA"
    opened_by_status["sub_value"] = "NA"

    opened_by_status = opened_by_status.rename(
        columns={
            "opened_by": "indicator_value",
            "issues_opened": "value",
        }
    )

    # alltime_ttr_perc metric
    # ----------------------------------------

    ttr_issues["report_date"] = f"{report_date}"
    ttr_issues["start_date"] = f"{start_date}"
    ttr_issues["metric"] = "overall_time_to_respond"
    ttr_issues["indicator_name"] = "time_window"
    ttr_issues["value_type"] = "percent"
    ttr_issues["sub_value_type"] = "cumulative_percent"

    ttr_issues = ttr_issues.rename(
        columns={
            "Time Window": "indicator_value",
            "Percent": "value",
            "Cumulative Percent": "sub_value",
        }
    )

    # alltime_nondev_ttr_perc metric
    # ----------------------------------------

    nondev_ttr_issues["report_date"] = f"{report_date}"
    nondev_ttr_issues["start_date"] = f"{start_date}"
    nondev_ttr_issues["metric"] = "nondev_time_to_respond"
    nondev_ttr_issues["indicator_name"] = "time_window"
    nondev_ttr_issues["value_type"] = "percent"
    nondev_ttr_issues["sub_value_type"] = "cumulative_percent"

    nondev_ttr_issues = nondev_ttr_issues.rename(
        columns={
            "Time Window": "indicator_value",
            "Percent": "value",
            "Cumulative Percent": "sub_value",
        }
    )

    # concatenate all report data
    # ----------------------------------------
    report_data = pd.concat(
        [
            issues_status,
            opened_by_status,
            ttr_issues,
            nondev_ttr_issues,
        ],
        ignore_index=True,
    )

    if report_exists:
        report_data = pd.concat(
            [
                hist_report_data,
                report_data,
            ]
        )

    # order columns
    # ----------------------------------------
    report_data = report_data[
        [
            "report_date",
            "start_date",
            "metric",
            "indicator_name",
            "indicator_value",
            "value_type",
            "value",
            "sub_value_type",
            "sub_value",
        ]
    ]

    return report_data


# %% ----------------------------------------
# Define and run report
# -------------------------------------------


def run_report(
    start_date=False,
    end_date=False,
    display_tables=False,
    style_displayed_tables=True,
    save_report_data=True,
    overwrite_historical_data=False,
):
    """
    Run issue analysis report.

    Parameters
    ----------
    end_date : str or bool
        Manual end date for the report in "YYYY-MM-DD HH:MM:SS" format. If False,
        uses the current date.
    display_tables : bool
        Whether to display the report tables.
    style_displayed_tables : bool
        Whether to add HTML styling to the displayed tables
    save_report_data : bool
        Whether to save the report data.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing issue data.
    """
    run_date = datetime.now().date()

    # set report end date
    # ------------------------------
    if end_date is False:
        report_date = run_date
    else:
        report_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"Using report date of {report_date}\n")

    # Load pickle file of raw issue data generated by download_issues.py
    # ------------------------------
    with open(DATAPATH, "rb") as f:
        raw_issue_data = pickle.load(f)

    raw_issue_data = pd.DataFrame(raw_issue_data)

    df = raw_issue_data.copy()

    print(
        "Date range of opened issues:",
        f"\n   First_issue_opened : {df['date_opened'].min()} UTC",
        f"\n   Last_issue_opened  : {df['date_opened'].max()} UTC",
    )

    # set start date
    # ------------------------------
    if start_date is False:
        # datetime of earliest record in EST
        start_date = pd.to_datetime(df["date_time"].min(), utc=True)
        start_date = start_date.tz_convert("US/Eastern")
        # remove time from record
        start_date = start_date.tz_localize(None)

    start_date = pd.to_datetime(start_date).date()

    print(
        "\nDate range of report:",
        f"\n   Start : {start_date} EST",
        f"\n   End   : {report_date} EST",
    )

    # preprocess raw data
    # ------------------------------
    dev_usernames = [
        "jasmainak",
        "ntolley",
        "asoplata",
        "dylansdaniels",
        "katduecker",
        "carolinafernandezp",
        "gtdang",
        "kmilo9999",
        "samadpls",
        "Myrausman",
        "Chetank99",
    ]

    df = preprocess(
        df,
        dev_usernames,
        start_date=start_date,
        end_date=report_date,
    )

    # generate table of issues opened, closed
    # ------------------------------
    issues_status_overall = issue_status_counts(df)

    # generate table of issues opened by developer status
    # ------------------------------
    opened_by_status_table = issues_opened_by_users(
        df,
        by_dev_status=True,
    )

    # Generate overall time-to-response table
    # ------------------------------
    issues_segmented = process_issues_for_ttr(
        df,
        report_date,
    )
    ttr_issues_table = generate_ttr_table(issues_segmented)

    # Generate non-developer time-to-response table
    # ------------------------------
    nondev_issues_segmented = process_issues_for_ttr(
        df.loc[df["opened_by"] != "developer"].reset_index(drop=True),
        report_date,
    )

    nondev_ttr_issues_table = generate_ttr_table(nondev_issues_segmented)

    # Display styled or unstyled tables
    # ------------------------------
    if display_tables:
        if style_displayed_tables is False:
            display(
                opened_by_status_table,
                issues_status_overall,
                ttr_issues_table,
                nondev_ttr_issues_table,
            )
        elif style_displayed_tables is True:
            display(
                render_html_table(issues_status_overall),
                render_html_table(opened_by_status_table),
                render_html_table(ttr_issues_table),
                render_html_table(nondev_ttr_issues_table),
            )

    # Save report data
    # ------------------------------

    if save_report_data:
        report_data = prep_report_data_for_saving(
            start_date,
            report_date,
            issues_status_overall,
            opened_by_status_table,
            ttr_issues_table,
            nondev_ttr_issues_table,
            overwrite_historical_data,
        )

        with open(REPORTPATH, "wb") as f:
            pickle.dump(report_data, f)

        print(f"\nReport data saved to {REPORTPATH}")

    return df


if __name__ == "__main__":
    # set report parameters
    # ------------------------------
    start_date = False
    end_date = False
    display_tables = True
    style_displayed_tables = True
    save_report_data = True
    overwrite_historical_data = True

    # --> [DEV]
    if "dylandaniels" in os.getcwd():
        end_date = "2025-12-01"
    # --> [END DEV]

    processed_df = run_report(
        start_date,
        end_date,
        display_tables,
        style_displayed_tables,
        save_report_data,
        overwrite_historical_data,
    )

    # %% -------------------------------------
    # posthoc checks
    peek(processed_df)

    with open(REPORTPATH, "rb") as f:
        report_data = pickle.load(f)

    # only show data for the current report date
    display(
        report_data[report_data["report_date"] == str(pd.to_datetime(end_date).date())]
    )


# %% ----------------------------------------
# -------------------------------------------


# %% ---------------------------
# examine cases where ttr > 2 days
# ------------------------------

# long_response = issues_segmented[issues_segmented["ttr_days"] >= 2.0].copy()

# display(long_response["username"].value_counts().reset_index())


# def closed_indicator(row):
#     if pd.notnull(row["date_closed"]):
#         return "closed"
#     else:
#         return "open"


# long_response["closed"] = long_response.apply(lambda x: closed_indicator(x), axis=1)

# display(long_response.groupby(["closed", "status"])["number"].count().reset_index())

# display(
#     long_response.groupby(["username", "closed", "status"])["number"]
#     .count()
#     .reset_index()
# )
