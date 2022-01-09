import spotipy
import pylast
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
from sklearn import svm
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
# from mpl_toolkits.mplot3d import Axes3D

client_id = os.environ['CLIENT_ID']
client_secret = os.environ['CLIENT_SECRET']
username = os.environ['USER_NAME']
redirect_uri = os.environ['REDIRECT_URI']

API_KEY = os.environ['LASTFM_KEY']
API_SECRET = os.environ['LASTFM_SECRET']

scope = "user-library-read user-follow-read user-top-read playlist-read-private playlist-read-collaborative"

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

sp = spotipy.Spotify(client_credentials_manager = auth_manager)

network = pylast.LastFMNetwork(
	api_key=API_KEY,
	api_secret=API_SECRET
)

## Begin the recommendation process
def start(username):
	user = pylast.User(username, network)
	top_songs = favourites(user)

	## Retrieve features
	features, missing = fetch_audio_features(top_songs)
	top_songs.drop(top_songs.index[missing], inplace=True)
	
	df = features.assign(weight=top_songs["weight"])
	y = top_songs["weight"].values
	
	## Remove outliers
	features, y = remove_outliers(df, y)
	
	## Standardize features
	scaler = StandardScaler().fit(features)
	X = scaler.transform(features)
	
	## PCA reduce number of features due to small data set
	pca = PCA(n_components=8).fit(X)
	X_pc = pca.transform(X)
	
	regression = LinearRegression().fit(X_pc, y)
	
	## Evaluate the model
	# evaluate_model(X_pc, y)
	
	# plot3D(X, y)

	return get_recs(sp, top_songs, scaler, regression, pca)

## Find coefficient of determination using cross validation on all data
def evaluate_model(X_pc, y):
	kf = KFold(n_splits=5, shuffle=True)
	y_true, y_pred = [], []
	for train_i, test_i in kf.split(X_pc):
		X_train, X_test = X_pc[train_i], X_pc[test_i]
		y_train, y_test = y[train_i], y[test_i]
		
		reg = LinearRegression().fit(X_train, y_train)
		y_hat = reg.predict(X_test)
		
		y_true.extend(y_test.tolist())
		y_pred.extend(y_hat.tolist())
		# print(y_test, y_hat)
		
	r2 = r2_score(y_true, list(map(int, y_pred)))
	print("Coefficient of Determination:", r2)

## Find standard deviation and mean to remove outliers for y
def remove_outliers(df, y):
	std = np.std(y)
	mean = np.mean(y)
	
	upper = mean + 3 * std
	lower = mean - 3 * std
	
	outlierless = df[(df["weight"] < upper) & (df["weight"] > lower)]
	features = outlierless[outlierless.columns.difference(["weight"])].to_numpy()
	y = outlierless["weight"].values
	
	return features, y

## Plot the linear regression (3D hyperplane) of the dimension reduced features using PCA
"""
def plot3D(X, y):
	## PCA setup and regression
	pca_audio = PCA(n_components=2)
	pc_audio = pca_audio.fit_transform(X)
	regression = LinearRegression().fit(pc_audio, y)
	prediction = regression.predict(pc_audio)
	
	## Form the plane
	coefs = regression.coef_
	intercept = regression.intercept_
	xs = np.tile(np.arange(-4, 8), (12,1))
	ys = np.tile(np.arange(-4, 8), (12,1)).transpose()
	zs = xs * coefs[0] + ys * coefs[1] + intercept
	
	## Begin plotting
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	
	ax.set_xlabel('Principal Component - 1', fontsize=16)
	ax.set_ylabel('Principal Component - 2', fontsize=16)
	ax.set_zlabel('Target', fontsize=16)
	ax.set_title("Principal Component Analysis of Audio Features",fontsize=24)
	ax.scatter(pc_audio[:, 0], pc_audio[:, 1], y, color='red')
	
	ax.plot_surface(xs, ys, zs, alpha=0.5)
	plt.show()
	
## Plot the linear regression (3D hyperplane) of the dimension reduced features using PCA
def plot2D(X, y):
	## PCA setup and regression
	pca_audio = PCA(n_components=1)
	pc_audio = pca_audio.fit_transform(X)
	regression = LinearRegression().fit(pc_audio, y)
	prediction = regression.predict(pc_audio)
	
	## Form the plane
	coefs = regression.coef_
	intercept = regression.intercept_
	
	## Begin plotting
	fig = plt.figure()
	ax = fig.add_subplot()
	
	ax.set_xlabel('Principal Component', fontsize=16)
	ax.set_ylabel('Target', fontsize=16)
	ax.set_title("Principal Component Analysis of Audio Features",fontsize=24)
	ax.scatter(pc_audio, y, color='red')
	ax.plot(pc_audio, prediction)
	
	plt.show()
"""
## Return a dataframe of the user's top 50 tracks
def favourites(user):
	raw_fm = user.get_top_tracks(period="12months", limit=300)
	
	missing = 0
	track_ids, playcount = [], []
	for song in raw_fm:
		try: 
			track_ids.append(sp.search(q='track: ' + song.item.title + ' artist: ' + str(song.item.artist), type='track')['tracks']['items'][0]['id'])
			playcount.append(song.weight)
		except IndexError:
			print(song.item.title + " - " + str(song.item.artist) + " not found.")
			missing += 1
		except Exception as e:
			print(e)
	
	print("Missing:", missing, "out of", len(raw_fm))
	
	top_fm = pd.DataFrame({"track_id": track_ids, "weight": playcount})
	
	return top_fm

## Converts the Spotify API's response to a dataframe
def convert_df(sp_result):
	track_name, track_id, artist, album, duration, popularity = ([] for i in range(6))

	for i, items in enumerate(sp_result):
		track_name.append(items['name'])
		track_id.append(items['id'])
		artist.append(items["artists"][0]["name"])
		duration.append(items["duration_ms"])
		album.append(items["album"]["name"])
		popularity.append(items["popularity"])

	df = pd.DataFrame({
		"track_name": track_name, 
		"album": album, 
		"track_id": track_id,
		"artist": artist,
		"popularity": popularity})
	
	return df

## Requests audio features for tracks from Spotify's API
def fetch_audio_features(df):
	playlist = df[['track_id']]
	audio_features = []

	## Search for the audio features of the top 50 songs
	for i in range(0, playlist.shape[0], 100):
		audio_features.extend(sp.audio_features(playlist.iloc[i:i + 100, 0]))
		
	## Make a list of dictionaries of audio features of each song
	features_list, missing = [], []
	wanted_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']
	for i, features in enumerate(audio_features):
		try:
			features_list.append({key: features[key] for key in wanted_features})
		except:
			missing.append(i)
	
	feature_df = pd.DataFrame(features_list)
	
	return feature_df, missing

## Request songs from Spotify's API and filter through them to get the top 25 recommendations
def get_recs(sp, top_songs, scaler, regression, pca):
	seed_tracks = top_songs["track_id"].tolist()
	
	sp_list = []
	for i in range(0, len(seed_tracks), 5):
		recs = sp.recommendations(seed_tracks=seed_tracks[i:i+5], limit=50)
		sp_list += recs['tracks']
	
	## Remove duplicates and songs from top 50
	df = convert_df(sp_list).drop_duplicates()
	common_df = df.merge(top_songs, how='inner', indicator=False)
	df = pd.concat([df, common_df]).drop_duplicates(subset=['track_id'], keep=False)
	
	## Predict on the new songs' audio features using the trained linear regression and PCA
	features, missing = fetch_audio_features(df)
	df.drop(df.index[missing], inplace=True)
	
	X = scaler.transform(np.array(features))
	X_pc = pca.transform(X)
	predictions = regression.predict(X_pc)
	
	recs_df = pd.DataFrame({
		"Title": df['track_name'],
		"Artist": df['artist'],
		"Recommendation Strength": predictions.tolist()
	})
	recs_df.sort_values(by=["Recommendation Strength"], ascending=False, inplace=True)
	
	return recs_df.head(25)[["Title", "Artist"]]

# start()