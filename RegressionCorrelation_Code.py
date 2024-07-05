"""
DS2500
Michael, Rishi, Jeff
4/12/24

Final Project Code

This code does our multiple linear regression, gets correlation, and
graphs linear regression.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = ["duration_ms", "msPlayed_x", "normalized_danceability",
           "normalized_energy", "normalized_key", "normalized_loudness",
           "normalized_mode", "normalized_speechiness",
           "normalized_acousticness", "normalized_instrumentalness",
           "normalized_liveness", "normalized_valence",
           "normalized_tempo"]
FEATURES = COLUMNS[2:]

def mult_regress(df):
    """ Given the data frame of spotify data, this function does multiple
    linear regression on msPlayed and all of the other features"""
    X = df.drop(columns = "duration_ms")
    X = X.drop (columns = "msPlayed_x")
    Y = df["msPlayed_x"]
    xtr, xte, ytr, yte = train_test_split(X, Y, random_state=0)
    lr = LinearRegression()
    lr.fit(xtr, ytr)
    ytr_pred = lr.predict(xtr)
    print(round(r2_score(ytr, ytr_pred), 5))
    
def get_r_vals(df,FEATURES):
    """ Given the spotify dataset and the features, get the correlation
    coefficient between each feature and msplayed"""
    rr_vals = [np.corrcoef(df[feature], df["msPlayed_x"]) for
               feature in FEATURES]
    r_vals = []
    for a in rr_vals:
        r_vals.append(a[1][0])
    r = [round(elem, 4) for elem in r_vals ]
    return r

def plot_regress(ndf, df, feature, xlab, ylab, t, tt):
    """ Given the spotify dataset and the feature wanted, plot a regression model
    for each feature and msplayed"""
    hi = df[["msPlayed_x", feature]]
    sns.regplot(x="msPlayed_x", y=feature, data=hi)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(t)
    plt.show()
    
    boo = ndf[["msPlayed_x", feature]]
    sns.regplot(x="msPlayed_x", y=feature, data=boo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tt)
    plt.show()

def main():
    # Getting data from csv
    data = pd.read_csv("Clean_Data.csv")
    df = data[COLUMNS]
    
    # Running multiple linear regression
    mult_regress(df)
    
    # Getting individual r vals for msplayed and every feature
    standard_r_vals = get_r_vals(df,FEATURES)
    print("R vals for all of Jeffs songs:", standard_r_vals)

    # Do all of the above but for songs that have been played for >10 mins
    ndf = df[(df[["msPlayed_x"]] >= 1200000).all(axis=1)]
    mult_regress(ndf)
    chosen_r_vals = get_r_vals(ndf,FEATURES)
    print("R vals for songs Jeff has listened to for >10 mins:", chosen_r_vals)
    
    # Plotting instrumentalness for both datasets
    plot_regress(ndf, df, "normalized_instrumentalness", "Miliseconds Played",
                 "Instrumentalness", "Instumentalness vs Miliseconds Played",
                 "Instumentalness vs Miliseconds Played (>10 mins)")
    
    # Plotting liveness for both datasets
    plot_regress(ndf, df, "normalized_liveness", "Miliseconds Played",
                 "Liveness", "Liveness vs Miliseconds Played",
                 "Liveness vs Miliseconds Played (>10 mins)")
    
    # Plotting valence for both datasets
    plot_regress(ndf, df, "normalized_speechiness", "Miliseconds Played",
                 "Speechiness", "Speechiness vs Miliseconds Played",
                 "Speechiness vs Miliseconds Played (>10 mins)")
    
main()