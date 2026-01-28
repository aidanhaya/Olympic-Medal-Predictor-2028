import pandas as pd

# retrieve X_df (up to 2024)

df = pd.read_csv("Data/X_df.csv")

# calculate H based on prior years

H_2028 = (
    df
    .sort_values("Year")
    .groupby("Team")["H"]
    .last()
    .reset_index()
)

H_2028["Year"] = 2028

# cxa TP from 2024 (assuming similar levels)

TP_2028 = (
    df[df["Year"] == 2024][["Team", "TP"]]
    .copy()
)

TP_2028["Year"] = 2028

# mark US as host

Host_2028 = pd.DataFrame({
    "Team": df["Team"].unique(),
    "Year": 2028,
    "Host": 0
})

Host_2028.loc[Host_2028["Team"] == "United States", "Host"] = 1

# get TSE, add 22 (reported increase in 2028)

NEW_EVENTS_2028 = 22

TSE_2028 = df[df["Year"] == 2024][["TSE"]].iloc[0, 0] + NEW_EVENTS_2028

# merge into X_2028

X_2028 = (
    H_2028
    .merge(TP_2028, on=["Team", "Year"])
    .merge(Host_2028, on=["Team", "Year"])
)

X_2028["TSE"] = TSE_2028

X_2028 = X_2028[["Team", "Year", "H", "TP", "Host", "TSE"]]

# append to X_df

X_full = pd.concat([df, X_2028], ignore_index=True)
X_full.to_csv("Data/X_df_with_2028.csv", index=False)

print(X_full)