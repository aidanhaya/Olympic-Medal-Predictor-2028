import numpy as np
import pandas as pd
import warnings
import datetime as dt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

def parse_tm_string(tm_str):
    tm_list = [int(x) for x in tm_str.strip('[]').split()]
    return np.array(tm_list)

def main():

    start_time = dt.datetime.now()

    df = pd.read_csv("Data/X_df_with_2028.csv")

    # split 2028 and historical data

    hist_df = df[df["Year"] < 2028].sort_values("Year")
    df_2028 = df[df["Year"] == 2028]

    # chronological train-test split

    split_idx = int(len(hist_df) * 0.9)

    train_df = hist_df.iloc[:split_idx]
    test_df  = hist_df.iloc[split_idx:]

    X_train = train_df[['H', 'TP', 'TSE', 'Host', 'Year', 'Team']]
    Y_train = np.stack(train_df['TM'].apply(parse_tm_string))

    X_test  = test_df[['H', 'TP', 'TSE', 'Host', 'Year', 'Team']]
    Y_test  = np.stack(test_df['TM'].apply(parse_tm_string))

    # categorical data, preprocessing

    numeric_features = ["H", "TSE", "Host", "TP", "Year"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_features = ["Team"]
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=int))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ))
    ])

    # training the model

    pipeline.fit(X_train, Y_train)

    # evaluating model accuracy

    Y_pred = pipeline.predict(X_test)

    MSE = mean_squared_error(Y_test, Y_pred)
    r2  = r2_score(Y_test, Y_pred)

    print("\n---HISTORICAL RESULTS---\n")
    print(f"MSE: {MSE}")
    print(f"R2 SCORE: {r2}")

    # predicting 2028 results

    X_2028 = df_2028[['H', 'TP', 'TSE', 'Host', 'Year', 'Team']]
    Y_2028_pred = pipeline.predict(X_2028)

    pred_2028 = df_2028[['Team']].copy()
    pred_2028['Pred_TM'] = list(Y_2028_pred)

    # round Pred_TM for printing, display results

    pred_2028['Pred_TM'] = pred_2028['Pred_TM'].apply(
        lambda x: np.round(x).astype(int)
    )

    print("\n---2028 PREDICTIONS---\n")
    print(
        pred_2028
        .assign(Total=lambda x: x['Pred_TM'].apply(sum)).round(2)
        .sort_values("Total", ascending=False)
        .head(10)
    )

    end_time = dt.datetime.now()
    runtime = (end_time - start_time).total_seconds()
    print(f"\n---RUNTIME: {runtime:.2f} seconds---\n")

if __name__ == '__main__':
    main()
