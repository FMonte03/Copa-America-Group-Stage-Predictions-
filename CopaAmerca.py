import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("./CopaAmericaFullFeatures.csv")
df['date'] = pd.to_datetime(df['date'])

df  = df[df['date'] > '2024-06-19']

copa_teams = ('Argentina', 'Peru', 'Chile', 'Canada', 'Mexico', 'Ecuador', 'Venezuela', 'Jamaica', 'United States', 'Uruguay', 'Panama', 'Bolivia', 'Brazil', 'Colombia', 'Paraguay', 'Costa Rica')

# Filter the DataFrame
df = df[(df['home_team'].isin(copa_teams)) | (df['away_team'].isin(copa_teams))]

df.to_csv('CopaMatches.csv')


