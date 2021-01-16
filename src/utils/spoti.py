import pandas as pd
import spotipy

from spotipy.oauth2 import SpotifyClientCredentials
from typing import Tuple


def authenticate() -> spotipy.client.Spotify:
    """
    Authenticate to spotify api, SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables needs to be set

    :return: spotify client object
    """
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    return spotify


def get_playlist_info(sp: spotipy.client.Spotify, playlist_uri: str) -> Tuple[str, list, list, list]:
    """
    Extract track names and URIs from a playlist

    :param sp: spotify client object
    :param playlist_uri: playlist uri
    :return: playlist name, track names, artist names, song uris
    """

    # Initialize vars
    offset = 0
    tracks, uris, names, artists = [], [], [], []

    # Get playlist id and name from URI
    playlist_id = playlist_uri.split(':')[2]
    playlist_name = sp.playlist(playlist_id)['name']

    # Get all tracks in given playlist (max limit is 100 at a time --> use offset)
    while True:
        results = sp.playlist_items(playlist_id, offset=offset)
        tracks += results['items']
        if results['next'] is not None:
            offset += 100
        else:
            break

    # Get track metadata
    for track in tracks:
        names.append(track['track']['name'])
        artists.append(track['track']['artists'][0]['name'])
        uris.append(track['track']['uri'])

    return playlist_name, names, artists, uris


def get_features_for_playlist(sp: spotipy.client.Spotify, uri: str) -> pd.DataFrame:
    """
    Extract features from each track in a playlist

    :param sp: spotify client object
    :param uri: playlist uri
    :return: playlist with audio features of each song
    """
    # Get all track metadata from given playlist
    playlist_name, names, artists, uris = get_playlist_info(sp, uri)

    # Set column names and empty dataframe
    column_names = ['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy',
                    'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                    'playlist']
    df = pd.DataFrame(columns=column_names)

    # Iterate through each track to get audio features and save data into dataframe
    for name, artist, track_uri in zip(names, artists, uris):

        # Access audio features for given track URI via spotipy
        audio_features = sp.audio_features(track_uri)

        # Get relevant audio features
        feature_subset = [audio_features[0][col] for col in column_names
                          if col not in ["name", "artist", "track_URI", "playlist"]]

        # Compose a row of the dataframe by flattening the list of audio features
        row = [name, artist, track_uri, *feature_subset, playlist_name]
        df.loc[len(df.index)] = row

    return df
