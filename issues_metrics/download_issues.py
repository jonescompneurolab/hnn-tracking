# %% ----------------------------------------
# download_issues.py

import os
import pickle
import sys
import time
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}  # for GitHub Action
# HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}  # for local testing

OWNER = "jonescompneurolab"
REPO = "hnn-core"
DATAPATH = os.path.join(
    "issues_metrics",
    "raw_issues_data.pkl",
)

if not GITHUB_TOKEN:
    print("Error: GITHUB_TOKEN environment variable is not set.")
    sys.exit(1)

def get_start_end_dates(
    df,
    rerun_all=False,
    manual_end=False,
):
    """
    Determine START_DATE and END_DATE based on the historical issue
    data

    START_DATE should be the oldest unresolved issue date in the historical data (to
    minimize the number of API calls needed).

    END_DATE should be the day before the GitHub Action is run
    """

    # check that df of historical data exists
    if (rerun_all is False) and (not isinstance(df, pd.DataFrame)):
        warnings.warn(
            "Historical data not found. All data will be fetched \n"
            "anew even through rerun_all is set to False"
        )

    if not rerun_all:
        # ------- TODO ------- #
        # -------------------- #
        # This is just placeholder code for now
        #
        # Below, I'm using the most recently "opened" issue in
        # the historical data as the start date, though this
        # is not what we actually want to do in practice, as
        # issues will continue to accrue activity over time
        # and their status may change, which we need to keep
        # track of.
        #
        # We would ideally want to make the starting point
        # the oldest "unresolved" issue, which we would need
        # some logic from the analysis script to figure out
        START_DATE = df["date_opened"].max()

    else:
        START_DATE = "2024-08-01"
        START_DATE = datetime.strptime(START_DATE, "%Y-%m-%d").date()

    if manual_end is True:
        END_DATE = datetime.strptime(manual_end, "%Y-%m-%d").date()
    else:
        # Since the GitHub Action will be run on the first of
        # the month, use the previous day as the end date
        END_DATE = date.today() - timedelta(days=1)

    return START_DATE, END_DATE


def safe_request(url):
    """
    Make a get request to the GitHub API, with handling for rate limits
    """
    try:
        resp = requests.get(url, headers=HEADERS)
        if (
            resp.status_code == 403
            and "X-RateLimit-Remaining" in resp.headers
            and resp.headers["X-RateLimit-Remaining"] == "0"
        ):
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            reset_time = datetime.fromtimestamp(reset_ts).strftime("%Y-%m-%d %H:%M:%S")
            raise RuntimeError(
                f"GitHub rate limit exceeded. Try again after {reset_time}."
            )
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request to {url} failed: {e}")


def get_issues(START_DATE, END_DATE):
    issues = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=all&per_page=100&page={page}"
        resp = safe_request(url)
        data = resp.json()
        if not data:
            break
        for issue in data:
            # skip pull requests
            if "pull_request" in issue:
                continue
            created = datetime.strptime(
                issue["created_at"],
                "%Y-%m-%dT%H:%M:%SZ"
            ).date()
            if START_DATE <= created <= END_DATE:
                issues.append(
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "created_at": issue["created_at"],
                        "html_url": issue["html_url"],
                        "user": issue["user"]["login"],
                        "comments_url": issue["comments_url"],
                        "closed_at": issue.get("closed_at"),
                        "closed_by": issue.get("closed_by", {}).get("login")
                        if issue.get("closed_by") else "",
                        "labels": [label["name"] for label in issue.get("labels", [])],
                        "milestone": issue.get("milestone", {}).get("title")
                        if issue.get("milestone")
                        else "",
                    }
                )
        page += 1
        time.sleep(0.5)
    return issues


def get_comments(
    comments_url,
    max_comments=5,
):
    comments = []
    page = 1
    while len(comments) < max_comments:
        url = f"{comments_url}?per_page=100&page={page}"
        resp = safe_request(url)
        data = resp.json()
        if not data:
            break
        for comment in data:
            comments.append(
                {
                    "created_at": comment["created_at"],
                    "user": comment["user"]["login"],
                    "body": comment["body"],
                }
            )
            if len(comments) >= max_comments:
                break
        page += 1
        time.sleep(0.5)
    return comments


def get_issues_with_comments(
        rerun_all=False,
        max_comments=5,
        manual_end=False,
):

    if rerun_all:
        print(
            "The 'rerun_all' flag is enabled. All historical data \n"
            "will be fetched and the 'raw_issues_data.pkl' file will \n"
            "be entirely overwritten."
        )

    if os.path.exists(DATAPATH):
        with open(DATAPATH, "rb") as f:
            hist_issues_data = pickle.load(f)
    else:
        hist_issues_data = None
        rerun_all = True

    START_DATE, END_DATE = get_start_end_dates(
        hist_issues_data,
        rerun_all=rerun_all,
        manual_end=manual_end,
    )

    print(f"Fetching data between {START_DATE} and {END_DATE}")

    issues = get_issues(START_DATE, END_DATE)
    rows = []

    for issue in issues:
        comments = get_comments(
            issue["comments_url"],
            max_comments=max_comments,
        )
        if comments:
            for c in comments:
                rows.append(
                    [
                        issue["number"],
                        issue["created_at"],
                        issue["user"],
                        issue["title"],
                        issue["html_url"],
                        issue["closed_at"] or "",
                        issue["closed_by"],
                        ", ".join(issue["labels"]),
                        issue["milestone"],
                        c["user"],
                        c["created_at"],
                        c["body"].replace("\n", " ").strip(),
                    ]
                )
        else:
            rows.append(
                [
                    issue["number"],
                    issue["created_at"],
                    issue["user"],
                    issue["title"],
                    issue["html_url"],
                    issue["closed_at"] or "",
                    issue["closed_by"],
                    ", ".join(issue["labels"]),
                    issue["milestone"],
                    "",
                    "",
                    "",
                ]
            )
    columns = [
        "number",
        "date_time",
        "username",
        "issue_name",
        "issue_url",
        "date_closed",
        "closed_by",
        "labels",
        "milestone",
        "comment_username",
        "comment_date",
        "comment_contents",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    df["date_opened"] = pd.to_datetime(
        df["date_time"]
    ).dt.date

    print(f"Fetched and updated {len(df['number'].unique())} unique issues.")

    # append historical data to df if not re-running everything
    if (rerun_all is False) and (isinstance(hist_issues_data, pd.DataFrame)):
        # get historical data before the start date, as everything
        # on or after start date has been fetched anew
        hist_data_retained = hist_issues_data.loc[
            hist_issues_data["date_opened"] < START_DATE
        ].copy()

        df = pd.concat(
            [hist_data_retained, df],
            ignore_index=True,
        )

        df = df.sort_values(
            by="date_time",
            ignore_index=True,
        )

    return df


# %% ----------------------------------------

if __name__ == "__main__":

    manual_end = False

    # --> [DEV]
    if "dylandaniels" in os.getcwd():
        manual_end = "2025-12-01"
    # --> [END DEV]

    issues_data = get_issues_with_comments(
        rerun_all=True,
        max_comments=5,
        manual_end=manual_end,
    )

    issues_data.to_pickle(DATAPATH)

# %%
