import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Loads data from the data file and returns two dataframes
def load_data():
    lap_times = pd.read_parquet('data/LapTimes.parquet')
    results = pd.read_parquet('data/RaceResults.parquet')
    results = results[results['Position'].notna()]
    return lap_times, results

# Features that include Lap time consistency, Position gained, and DNF flag and normalizing lap consistecny
def build_features(lap_times, results):
    lap_times['LapTime_sec'] = lap_times['Time'].dt.total_seconds()

    # Compute consistency per driver (across all laps/rounds)
    driver_consistency = lap_times.groupby('DriverId')['LapTime_sec'].agg(['mean', 'std']).reset_index()
    driver_consistency.columns = ['DriverId', 'avg_lap_overall', 'lap_std_overall']

    # Compute average lap time per round
    round_avg = lap_times.groupby('Round')['LapTime_sec'].mean().reset_index()
    round_avg.columns = ['Round', 'round_avg_lap']

    # Merge consistency and round average to results
    df = results.merge(driver_consistency, on='DriverId', how='left')
    df = df.merge(round_avg, on='Round', how='left')

    df['position_gain'] = df['GridPosition'] - df['Position']
    df['dnf'] = (df['Status'] != 'Finished').astype(int)
    df['consistency_score_raw'] = -df['lap_std_overall']
    df['pace_delta'] = df['avg_lap_overall'] - df['round_avg_lap']

    # For ML features, use named columns
    df['lap_std'] = df['lap_std_overall'].fillna(0)
    df['avg_lap'] = df['avg_lap_overall'].fillna(0)


    return df
#Target
def add_target(df):
    df['outperformed'] = (df['Position'] < df['GridPosition']).astype(int)
    return df


# Metrics: Driving Consistency Index
def compute_consistency_index(df):
    driver_stats = df.groupby('FullName').agg({'consistency_score_raw': 'mean' ,'position_gain': 'mean', 'dnf' : 'mean' }).reset_index()
    for col in ['consistency_score_raw', 'position_gain']:
        driver_stats[col] = (driver_stats[col] - driver_stats[col].mean()) / driver_stats[col].std()
    driver_stats['consistency_index'] = (0.5 * driver_stats['consistency_score_raw'] + 0.4 * driver_stats['position_gain'] - 0.3 * driver_stats['dnf'])
    return driver_stats.sort_values(by = 'consistency_index', ascending = False)

#Cluch Factor
def compute_clutch_factor(df):
    clutch = df.groupby('FullName').agg({'position_gain': 'mean'}).reset_index()
    clutch['clutch_factor'] = clutch['position_gain']
    return clutch.sort_values(by='clutch_factor', ascending=False)

#Machine Learning Layer
def train_model(df):
    features = df[['GridPosition', 'lap_std', 'avg_lap', 'pace_delta']].fillna(0)
    target = df['outperformed']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    return model, features


def feature_importance(model, features):
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
    return importance


#Machine Learning Driver Score
def compute_ml_driver_score(df, model):
    weights = model.feature_importances_
    w_grid, w_std, w_avg, w_delta = weights
    df['ml_score'] = (
        w_grid * (-df['GridPosition']) + w_std * (-df['lap_std']) + w_avg * (-df['avg_lap']) + w_delta * (-df['pace_delta']))
    driver_score = df.groupby('FullName')['ml_score'].mean().reset_index()
    return driver_score.sort_values(by='ml_score', ascending=False)

#Vizualization
def plot_top_drivers(driver_stats):
    top = driver_stats.head(10)

    plt.figure()
    sns.barplot(data=top, x='consistency_index', y='FullName')
    plt.title("Top 10 Most Consistent Drivers")
    plt.show()


def plot_ml_scores(driver_scores):
    top = driver_scores.head(10)

    plt.figure()
    sns.barplot(data=top, x='ml_score', y='FullName')
    plt.title("Top Drivers by ML Score")
    plt.show()


#main
if __name__ == "__main__":
    lap_times, results = load_data()
    df = build_features(lap_times, results)
    df = add_target(df)
    model, features = train_model(df)
    importance = feature_importance(model, features)
    print("\nFeature Importance:\n", importance)
    consistency = compute_consistency_index(df)
    clutch = compute_clutch_factor(df)
    ml_scores = compute_ml_driver_score(df, model)
    print("\nTop Consistency:\n", consistency.head(10))
    print("\nTop Clutch:\n", clutch.head(10))
    print("\nTop ML Score:\n", ml_scores.head(10))
    plot_top_drivers(consistency)
    plot_ml_scores(ml_scores)
