
"""
DS 2500 Project
"""

# Importing libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import statistics
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics 
import spotipy as util 
import warnings



# Use the warnings.filterwarnings() function to suppress specific warnings
warnings.filterwarnings("ignore")



username = '31bzljpdqfebkgflpbfan5kcevqu'
client_id ='d66545eaa173409abb29b0fd69cc9c13'
client_secret = 'e1624bd5e6104e45a426b7a0a5abba69'
redirect_uri = 'http://localhost:7777/callback'
scope = 'user-read-recently-played'
token = util.prompt_for_user_token(username=username, 
                                    scope=scope, 
                                    client_id=client_id,   
                                    client_secret=client_secret,     
                                    redirect_uri=redirect_uri)



# Reading csv into data frame
DF = pd.read_csv("Clean_Data.csv")



# Finding mean length of data set
MEAN_LEN = statistics.mean(DF["msPlayed_x"])


# Defining features and labels
FEATURES = ["normalized_danceability", "normalized_energy", "normalized_key",
"normalized_loudness", "normalized_mode", "normalized_speechiness",
"normalized_acousticness", "normalized_instrumentalness",
"normalized_liveness",	"normalized_valence", "normalized_tempo"]

LABEL = ["likes_song"]









def add_column(df, col_name, new_col, condition):
    """
    Given a datafame, column name, new column name, and a condition
    """
    
    df[new_col] = np.where(df[col_name] > condition, "Yes", "No")
    
    return df


def k_estimate(df, col_name):
    """
    Given a dataframe and column name, finds the amount of data, and 
    square roots it to find a viable k value
    """
    
    count = df[col_name].count()
    
    k_value = math.sqrt(count)

    return round(k_value)





def k_fold(df, features, labels, low_range, high_range, scoring):
    
    """
    Given a data frame, features and labels from the data frame,the range of
    k values, and the label to compute a specific score, returns a dictionary 
    with the k values as the key and the associated scores as the value
    """

    features = df[features] 
    labels = df[labels]  
    
    k_values = list(range(low_range, high_range))
    
    
    kf = KFold(n_splits = 5, random_state = 0, shuffle = True)
    
    results = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_validate(knn, features, labels, cv = kf,
scoring = scoring)
    
    
        means = scores["test_score"].mean()
        results[k] = means

    return results




def find_optimal_k(dct):
    """
    Given a dictionary, finds the highest (optimal)value and
    returns the associated key
    """
    
    max_key = max(dct, key = dct.get)
    
    return max_key


def classifier(df, features, labels, k):
    """
    Given a data frame, the features and labels from the data frame, 
    a k value, and a label, returns the knn classifier, the tested labels,
    predicted labels, and the associated F1 score
    
    """
    X = df[features]
    y = df[labels]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    y_pred = knn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Yes')
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')

      
    return knn_classifier, y_test, y_pred, accuracy, precision, recall, f1




def get_features(track_id: str, token: str) -> dict:
    """
    """
    sp = util.Spotify(auth=token)
    try:
        features = sp.audio_features([track_id])
        return features[0]
    except:
        return None
    
    
    
    
def user_input(token = token):
    """
    Given a url to a song, returns a dictionary with the features
    """
    url = str(input("Enter Url:"))
    features = get_features(url, token)
   
    
    return features




def get_input_feat(dct):
    """
    Given a dictionary, finds the features that are needed and gets the
    specific data of just the features returns a list with the values
    of the features
    """
    result = {}

    for key, value in dct.items():
        result[key] = value
        if key == 'tempo':
            break
     
    return result


def normalize_dictionary(dct, df):
    """
    Given a dictionary and dataframe normalizes all the values in the 
    dictionary and returns the values in a list
    """
    normalized_dict = {}
    
    for key, value in dct.items():
        column_values = df[key]
        min_val = column_values.min()
        max_val = column_values.max()
        normalized_value = (value - min_val) / (max_val - min_val) 
        normalized_dict[key] = normalized_value
        
     
    lst = normalized_dict.values()
    
    return lst
    


def lst_to_df(lst):
    """
    Given a list, converts the list to a pandas dataframe
    """
    
    df_input = pd.DataFrame(lst).T
    
    return df_input
    
        


def predict_song(features, knn_classifier):
    """
    Given the features of the song and a knn classifier, classifies the song
    into "yes" the person would like it or "no" the person would not like it
    and returns the result
    """
    predicted_label = knn_classifier.predict(features)
    
    return predicted_label




def create_heatmap(tested, predicted, labels, xlabel, ylabel, title):
    """
    Given the tested labels, the predicted labels, and the labels that 
    are being used for the classification, generates a heatmap for the 
    classifier with a title and tick labels
    """
    confusion = metrics.confusion_matrix(tested, predicted, labels=labels)
    
    sns.heatmap(confusion, annot=True, xticklabels=xlabel,
                yticklabels=ylabel)
    
    # Adjust the position of the ticks
    plt.xticks([0.5, 1.5], xlabel)
    plt.yticks([0.5, 1.5], ylabel)  
    
    plt.title(title)
    
    plt.show()
    
    
def k_linechart(dct, xlabel, ylabel, title):
    """
    Given a dictionary, lebsl for both axis's and a title, 
    plots the keys of the dct on the x and the values of dct on the y
    """
    keys = list(dct.keys())
    values = list(dct.values())

    plt.plot(keys, values)
    
    plt.xticks(keys)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.show()    
    






def main():


    
     # Gets updated dataframe by adding a column with the labels (Yes/No)
     add_column(DF, "msPlayed_x", "likes_song", MEAN_LEN)
     
     # Finds square root of k value 
     k_est = k_estimate(DF, "likes_song")
     

     # Gets dictionary, K value is the key and mean score is the value
     k_dct = k_fold(DF, FEATURES, LABEL, k_est - 5, k_est + 5, "accuracy")
     

    
     # Finds optimal k value for accuracy within given range 
     optimal_k = find_optimal_k(k_dct)
     print(f"{optimal_k} is the optimal value for accuracy between the K\
 value range of 60 and 70")
    
    
     # Builds the knn classifier
     knn_classifier = classifier(DF, FEATURES, LABEL, optimal_k)[0]
     
     # Gets the accuracy, precision, recall, and f1 scores and prints them
     accuracy_sc = classifier(DF, FEATURES, LABEL,  optimal_k)[3]
     precision_sc = classifier(DF, FEATURES, LABEL,  optimal_k)[4]
     recall_sc = classifier(DF, FEATURES, LABEL,  optimal_k)[5]
     f1_sc = classifier(DF, FEATURES, LABEL,  optimal_k)[6]
     print("Accuracy Score:", round(accuracy_sc, 5))
     print("Precision Score:", round(precision_sc, 5))
     print("Recall Score:", round(recall_sc, 5))
     print("F1 Score:", round(f1_sc, 5))
    
    
     # Gets the raw dictionary of the inputted song link
     user_input_dict = user_input()
     
     # Cleans the raw dictionary and only gets the dictionary of the features
     input_dct = get_input_feat(user_input_dict)
     
     
     # Normalizes the inputted dictionary value features and converts to list
     normalized_lst = normalize_dictionary(input_dct, DF)
     
     # Converts the inputted song list to a dataframe
     df_input = lst_to_df(normalized_lst)
     
     
     # Running KNN classifier to see if person would like reccomended song
     predicted_label = predict_song(df_input, knn_classifier)
     print(f"Would this person like the song you reccomended?\
: {predicted_label[0]}")


     # Getting the tested and predicted labels to generate heatmap
     tested_labels = knn_classifier = classifier(DF, FEATURES, LABEL,
optimal_k)[1]
     predicted_labels = knn_classifier = classifier(DF, FEATURES, LABEL, 
optimal_k)[2]
     
     # Generating heatmap 
     create_heatmap(tested_labels, predicted_labels, ["Yes", "No"]
  , ["Predicted Yes", "Predicted No"], ["Actual Yes", "Actual No"], 
  "Heatmap of Confusion Matrix for Song Classifier")

    # Generating lineplot of K-Values and their mean accuracy scores
     k_linechart(k_dct, "K-Values", "Mean Accuracy Score", "K-Values\
 vs Mean Accuracy Score for different values of K")
     

  # Song ulrs to put in classifier for testing

  # https://open.spotify.com/track/2Ey6y1MP7KH78m4CHXqZU9?si=d19d3ccf443b48cb
  
  # https://open.spotify.com/track/58ge6dfP91o9oXMzq3XkIS?si=9063099f18cf4678


     


if __name__ == "__main__":
    main()







