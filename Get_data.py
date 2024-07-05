"""
DS25000
Jeff, Michael, Rishi

This code gets all of our raw data into a CSV
"""

import ast
from typing import List
from os import listdir
import spotipy as spoti 
import requests
import pandas as pd
import csv
import json

# Creating constants used for the api

USERNAME = '31bzljpdqfebkgflpbfan5kcevqu'
CLIENT_ID ='d66545eaa173409abb29b0fd69cc9c13'
CLIENT_SECRET = 'e1624bd5e6104e45a426b7a0a5abba69'
REDIRECT_URI = 'http://localhost:7777/callback'
SCOPE = 'user-read-recently-played'
FIND_GATHERED_DATA = False
CSV = False
ALL_DATA = True
FIND_DATA = False


TOKEN = spoti.prompt_for_user_token(username=USERNAME, 
                                 scope=SCOPE, 
                                 client_id=CLIENT_ID,   
                                 client_secret=CLIENT_SECRET,     
                                 redirect_uri=REDIRECT_URI)



def find_files(file_indicator, path = 'MyData/'):
    """GIven a path and a str of filenames return a lst of filenames"""
    files = [path + x for x in listdir(path)
           if file_indicator in x]
    return files



def merge_pandas(lst_of_csvs):
    """Given a list of csv's containing identical rows and columns,
    but not values, merge the list of csv's together and return the file"""
    df = pd.read_csv(lst_of_csvs[0])
    for i in range(1, (len(lst_of_csvs))):
        dff = pd.read_csv(lst_of_csvs[i])
        lst = [df, dff]
        df = pd.concat(lst)
    return df

def read_csv(filename):
    """ given the name of a csv file, return its contents as a 2d list,
        including the header."""
    data = []
    with open(filename, "r") as infile:
        csvfile = csv.reader(infile)
        for row in csvfile:
            data.append(row)
    return data

def find_correct_ms_played(json_files, dct_key, dct_value):
    """Given a list of Json files set up a a list containing identically set up
    dictionary's, dictionary key for keys, and dictionary key for values,
    read the json's and create a dictionary with key's and
    values. Return the dictionary and the list of keys"""
    song_names = {}
    for file in json_files:
        json_read = open(file, "r")
 

        lst = json.loads(json_read.read())
        for dct in lst:
            if dct[dct_key] in song_names.keys():
                song_names[dct[dct_key]] += dct[dct_value]
            else: 
                song_names[dct[dct_key]] = dct[dct_value]
    total_dct = {"name": list(song_names.keys()),
                 "msPlayed": list(song_names.values())}
    
    return total_dct, list(song_names.keys())
        

def lst_to_dct(lst):
    """ given a 2d list, create and return a dictionary.
        keys of the dictionary come from the header (first
                                                     row)
        , values are corresponding columns, saved as lists
        Ex: [[1, 2, 3], [x, y, z], [a, b, c]]
        should return {1 : [x, a], 2 : [y, b], 3 : [z, c]}
    """

    dct = {h : [] for h in lst[0]}
    for row in lst[1:]:
        for i in range(len(row)):
            dct[lst[0][i]].append(row[i])
    return dct

def big_list(dct, key, big_lst):
    """Given a dictionary containing a list as it's value, create a big list
    containing that data return the list"""
    # check if theres a merge lst 
    for item in dct[key]:
        big_lst.append(item)
    return big_lst

def filter_lst(lst_of_total, lst_of_old):
    """Given a list of names and a lst of old names return only the new names"""
    new = []
    for name in lst_of_total:
        if name not in lst_of_old:
            new.append(name)
    return new


def get_id(track_info: str, token: str,
                artist: str = None) -> str:
    """Given a track's name, a spotify api token, and an artist, this function
    performs a request to gather a tracks spotify id, this function will then
    return the id
    Function taken from: https://github.com/vlad-ds/spoty-records
    """
    
    '''Performs a query on Spotify API to get a track ID.
    See https://curl.trillworks.com/'''

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    track_name = track_info.split("___")[0]
    params = [
    ('q', track_name),
    ('type', 'track'),
    ]
    artist = track_info.split("___")[-1]
    if artist:
        params.append(('artist', artist))
        
    try:
        response = requests.get('https://api.spotify.com/v1/search', 
                    headers = headers, params = params, timeout = 5)
        json = response.json()
        results = json['tracks']['items']
        first_result = json['tracks']['items'][0]
        # Check if searched artist is in response as the first one isn't
        # necessarily the right one
        if artist:
            for result in results:
                if artist.strip() == result['artists'][0]['name'].strip():
                    track_id = result['id']
                    return track_id
        # If specific artist is not found from results, use the first one
        track_id = first_result['id']
        return track_id
    except:
        return None

def get_features(track_id, token):
    """
    Given a track id and the api token, utilizes spotipy api to access 
    a song's features, return the features'
    """
    sp = spoti.Spotify(auth=token)
    try:
        features = sp.audio_features([track_id])
        return features[0]
    except:
        return None
    

def main():

    # creating a big list of song names that the features have already 
    # been found as spotify api limits 2000 requests a day
    # big_lst allows us too not repeat song names allowing us to collect
    # as many as possible
    if FIND_GATHERED_DATA == True:
        big_lst = []
        lst_of_csv = find_files("csv_features/", 'streaming_history')
        for file in lst_of_csv:
            lst = read_csv(file)
            dct = lst_to_dct(lst)
            big_lst = big_list(dct, "name", big_lst)
    # finding the files
    files = find_files("StreamingHistory_music")
    

    time_played, unique_tracks = find_correct_ms_played(files, "trackName",
                                                        "msPlayed")
    # Filtering out all of the songs whose features have been found before
    if FIND_GATHERED_DATA == True:
        unique_tracks = filter_lst(unique_tracks, big_lst)

    all_features = {}
    # gathering track id's and features
    for track in unique_tracks:
        track_id = get_id(track, TOKEN)
        features = get_features(track_id, TOKEN)
        if features:
            all_features[track] = features
        
        with_features = []
        # priming the dictionary to become a data set
        for track_name, features in all_features.items():
            with_features.append({'name': track_name, **features})
            
            print(with_features)
    # creating data frame 
    df_features = pd.DataFrame(with_features)
    # exporting the files
    if FIND_GATHERED_DATA == True:
    # finding files containing gathered data + features
        csv_files = find_files(".csv", "csv_features/")
        # merging the old data to one csv
        old_df = merge_pandas(csv_files)
        # merging all of the data together
        df = pd.concat(df_features, old_df)
        
    if CSV == True:
        # if exporting data for future use
        df_features.to_csv('streaming_history.csv')
        
    if ALL_DATA == True:
        # if all data is gathered and creates a dataframe with correct times
        # and features
        time = (pd.DataFrame(time_played)).sort_values(by=['name'])
        # Sort both dataframes by name
        df_features = df_features.sort_values(by=['name'])       	
        # Merge time dataframe with feature dataframe
        merged = time.merge(df_features, left_on="name", right_on="name")
        df = merged.drop("msPlayed_y", axis= 1)
        # Drop any songs that have zero ms listened to
        df = df[(df[["msPlayed_x"]] != 0).all(axis=1)]
    
main()
    
    
    
    