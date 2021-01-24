from utils import spoti

# Get data from playlist
sp = spoti.authenticate()
my_top_2017_uri = 'spotify:playlist:37i9dQZF1E9UxNJQfpBbMh'
df = spoti.get_features_for_playlist(sp, my_top_2017_uri)

# Save data to a folder
df.to_csv('./data/playlist_songs.csv', index=False)
