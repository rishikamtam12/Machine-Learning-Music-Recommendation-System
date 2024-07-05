"""
This code min-max normalizes and cleans our data,
At the end it provides it in the form of a new csv

Rishi Kamtam
"""
import pandas as pd
import numpy as np

features_lst = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
"acousticness", "instrumentalness", "liveness",	"valence", "tempo"]

def normalize_column(column):
    """ given a column of data, return the column but with min-max normalized data"""
    min_val = column.min()
    max_val = column.max()
    
    normalized_column = (column - min_val) / (max_val - min_val)
    
    return normalized_column

def data_cleaning_scientific_form(df, column_name):
    """ This turns every number too small (uses exponential form) into its actual
    numerical version"""
    col = list(df[column_name])
    lst_final = []
    for string in col:
        if "e" in str(string):
            string = str(string).split("e")
            integer = float(string[0])**float(string[1])
            lst_final.append(integer)
        else:
            lst_final.append(float(string))
    return lst_final


# Read data into dataframe
df = pd.read_csv("streaming_history.csv") 

# Get rid of any numbers in exponential form
for feature in features_lst:
    l = data_cleaning_scientific_form(df, feature)
    df[feature] = l
    
    
# Normalize all columns of feature data
for feature in features_lst:
    df[f"normalized_{feature}"] = normalize_column(df[feature])
    
    
# Get rid of any songs that have zero ms listened
df = df[(df[["msPlayed_x"]] != 0).all(axis=1)]

# Save dataframe as CSV
df.to_csv("Clean_Data.csv")












