import pandas as pd
import numpy as np

def parse_processed_data():
    processed_df = pd.read_csv('Data/processed_data.csv')
    return processed_df

def parse_tm_string(tm_str):
    # remove brackets, split by space, convert to int
    tm_list = [int(x) for x in tm_str.strip('[]').split()]
    return np.array(tm_list)

def main():
    processed_df = parse_processed_data()

    # TSE = total events per sport

    processed_df['TSE'] = (
        processed_df
        .groupby('Year')['Event']
        .transform('nunique')
    )

    # H_{mi} calculations

    # Convert string representation of array back to list
    processed_df['TM'] = processed_df['TM'].apply(parse_tm_string)

    country_year_df = (
        processed_df
        .groupby(['Team', 'Year'])['TM']
        .apply(lambda x: np.sum(np.stack(x), axis=0))
        .reset_index(name='TM')
    )

    # Total participants (delegation size)
    tp_df = (
        processed_df
        .groupby(['Team', 'Year'])['TP']
        .sum()
        .reset_index(name='TP')
    )

    # Host indicator
    host_df = (
        processed_df
        .groupby(['Team', 'Year'])['Host']
        .max()
        .reset_index()
    )

    # Total events at the Games
    tse_df = (
        processed_df
        .groupby('Year')['Event']
        .nunique()
        .reset_index(name='TSE')
    )

    country_year_df = country_year_df.sort_values('Year')

    country_year_df['H'] = (
        country_year_df
        .groupby('Team')['TM']
        .transform(
            lambda s: s
            .apply(lambda x: 0.15 * x[0] + 0.35 * x[1] + 0.45 * x[2])
            .cumsum()
            .shift(fill_value=0)
        )
    )

    final_df = (
        country_year_df
        .merge(tp_df, on=['Team', 'Year'])
        .merge(host_df, on=['Team', 'Year'])
        .merge(tse_df, on='Year')
        .sort_values('Year')
        .reset_index(drop=True)
    )

    final_df.to_csv('Data/X_df.csv', index=False)

    print(final_df)

if __name__ == "__main__":
    main()