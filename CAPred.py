import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("./results.csv")
df['date'] = pd.to_datetime(df['date'])
df  = df[df['date'] >= '2021-01-01']
df = df[df['date'] <= '2024-04-04']


copa_teams = ('Argentina', 'Peru', 'Chile', 'Canada', 'Mexico', 'Ecuador', 'Venezuela', 'Jamaica', 'United States', 'Uruguay', 'Panama', 'Bolivia', 'Brazil', 'Colombia', 'Paraguay', 'Costa Rica')

# Filter the DataFrame
df = df[(df['home_team'].isin(copa_teams)) | (df['away_team'].isin(copa_teams))]
df.dropna(inplace=True)
print(df.sort_values('date').tail())

ranking = pd.read_csv("./ranking.csv")
ranking['rank_date'] = pd.to_datetime( ranking['rank_date'])
ranking = ranking[ranking['rank_date'] >= '2021-01-01'].reset_index(drop=True)
ranking['country_full'] = ranking['country_full'].str.replace('USA', 'United States')
ranking = ranking.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().ffill().reset_index()

df_wc_ranked = df.merge(ranking[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"], axis=1)

df_wc_ranked = df_wc_ranked.merge(ranking[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "away_team"], right_on=["rank_date", "country_full"], suffixes=("_home", "_away")).drop(["rank_date", "country_full"], axis=1)

print(df_wc_ranked[(df_wc_ranked.home_team == "Brazil") | (df_wc_ranked.away_team == "Brazil")].tail(10))

df = df_wc_ranked
print(df.dtypes)


df['rank_difference'] = df['rank_home'] - df['rank_away']
def calculate_recent_win_rate(team, matches=5):
   
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(matches)
    wins = team_matches[((team_matches['home_team'].equals(team)) & (team_matches['home_score'] > team_matches['away_score'])) |
                        ((team_matches['away_team'].equals(team)) & (team_matches['away_score'] > team_matches['home_score']))]
    return len(wins) / matches if matches > 0 else 0
def calculate_recent_draw_rate(team, matches=5):
 
    team_matches = df[(df['home_team'].equals( team)) | (df['away_team'] == team)].tail(matches)
    draws = team_matches[team_matches['home_score'] == team_matches['away_score']]
    return len(draws) / matches if matches > 0 else 0

def calculate_recent_loss_rate(team, matches=5):
 
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(matches)
    losses = team_matches[((team_matches['home_team'].equals(team)) & (team_matches['home_score'] < team_matches['away_score'])) |
                          ((team_matches['away_team'].equals(team)) & (team_matches['away_score'] < team_matches['home_score']))]
    return len(losses) / matches if matches > 0 else 0

CopaMatches= pd.read_csv('./CopaMatches.csv')

CopaMatches.pop('home_score')
CopaMatches.pop('away_score')


#Match outcome will be 1 home win, 0 draw, and -1 away win.
def match_outcome(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return -1
    else:
        return 0

df['match_outcome'] = df.apply(match_outcome, axis=1)

print(df.head())
print(df.info())

def total_points_home(row): 
    home_team = row['home_team']
   
    lastrow = df[df['home_team'] == home_team].tail(1)
    return lastrow['total_points_home'].values[0]

def total_points_away(row): 
    away_team = row['away_team']
   
    lastrow = df[df['away_team'] == away_team].tail(1)
    return lastrow['total_points_away'].values[0]

def rankHome(row): 
    home_team = row['home_team']
    lastrow = df[df['home_team'] == home_team].tail(1)
    return lastrow['rank_home'].values[0]
def rankAway(row): 
    away_team = row['away_team']
    lastrow = df[df['away_team'] == away_team].tail(1)
    return lastrow['rank_away'].values[0]
CopaMatches['total_points_home'] = CopaMatches.apply(total_points_home, axis=1)
CopaMatches['total_points_away'] = CopaMatches.apply(total_points_away, axis=1)
CopaMatches['rank_home'] = CopaMatches.apply(rankHome, axis=1)
CopaMatches['rank_away'] = CopaMatches.apply(rankAway, axis=1)
CopaMatches['rank_difference'] = CopaMatches['rank_home'] - CopaMatches['rank_away']



features = [
    'total_points_home', 'rank_home', 
    'total_points_away' , 'rank_away', 
    'rank_difference'
]

target = 'match_outcome'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Predict outcomes for CopaMatches
X_CopaMatches = CopaMatches[features]
CopaMatches['predicted_outcome'] = model.predict(X_CopaMatches)

# Display the predictions
print(CopaMatches.head())
CopaMatches.to_csv('GroupStagePredictions')