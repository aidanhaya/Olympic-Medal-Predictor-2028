import pandas as pd
import numpy as np

def main():

    # get all data from csv files

    athletes_df = pd.read_csv('Data/summerOly_athletes.csv')
    hosts_df = pd.read_csv('Data/summerOly_hosts_edited.csv', encoding="cp1252")
    medals_df = pd.read_csv('Data/summerOly_medal_counts.csv')

    # remove 1906, 1980 olympics

    processed_df = athletes_df.copy()

    processed_df = processed_df[processed_df["Year"] != 1906]
    processed_df = processed_df[processed_df["Year"] != 1980]
    hosts_df = hosts_df[hosts_df["Year"] != 1980]
    medals_df = medals_df[medals_df["Year"] != 1980]

    # create dict of medals

    medal_dict = {}

    for row in medals_df.itertuples():
        noc = row.NOC
        total_medals = row.Total

        medal_dict[noc] = medal_dict.get(noc, 0) + total_medals

    # replace medals with ints

    medal_map = {
        "No Medal": 0,
        "No medal": 0,
        "Gold": 1,
        "Silver": 2,
        "Bronze": 3
    }

    processed_df["Medal"] = processed_df["Medal"].replace(medal_map).astype(int)

    # add host country info to processed_df

    # extracts host country name
    hosts_df['HostCountry'] = hosts_df['Host'].str.split(',').str[-1].str.strip()

    # merges processed_df and host_df, appends 'HostCountry' and 'Host' columns
    processed_df = processed_df.merge(
        hosts_df[['Year', 'HostCountry']],
        left_on=['Year', 'Team'],
        right_on=['Year', 'HostCountry'],
        how='left'
    )

    # converts host country to 1, NaN to 0
    processed_df['Host'] = processed_df['HostCountry'].notna().astype(int)

    # drops 'Host' column
    processed_df.drop(columns=['HostCountry'], inplace=True)

    # sorting by year and country

    processed_df.sort_values(
        by=['Year', 'Team', 'Sport', 'Event'],
        ascending=[True, True, True, True],
        inplace=True
    )

    # append total medals + total participants columns

    # TP = Total Players
    processed_df['TP'] = (
        processed_df
        .groupby(['Year', 'Team'])['Name']
        .transform('nunique')
    )

    # TM = Total Medals
    # TM array key: [bronze, silver, gold]
    def medal_array(s):
        if (s == 1).any():  # Gold
            return np.array([0, 0, 1])
        elif (s == 2).any():  # Silver
            return np.array([0, 1, 0])
        elif (s == 3).any():  # Bronze
            return np.array([1, 0, 0])
        else:
            return np.array([0, 0, 0])

    tm_df = (
        processed_df
        .groupby(['Year', 'Team', 'Event'])['Medal']
        .apply(medal_array)
        .reset_index(name='TM')
    )

    processed_df = processed_df.merge(
        tm_df,
        on=['Year', 'Team', 'Event'],
        how='left'
    )

    # remove sex, city
    processed_df.drop(columns=["Sex"], inplace=True)
    processed_df.drop(columns=["City"], inplace=True)

    # remove duplicate country-event-year data

    processed_df = processed_df.drop_duplicates(
        subset=['Year', 'Team', 'Event'],
        keep='first'
    )

    # store df as csv

    print(processed_df)

    processed_df.to_csv('Data/processed_data.csv', index=False)

if __name__ == '__main__':
    main()