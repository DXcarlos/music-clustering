from utils import spoti
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Get data from playlist
sp = spoti.authenticate()
my_top_2017_uri = 'spotify:playlist:37i9dQZF1E9UxNJQfpBbMh'
df = spoti.get_features_for_playlist(sp, my_top_2017_uri)

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(['name', 'artist', 'track_URI', 'playlist'], axis=1))

# Apply PCA
pca = PCA()
pca.fit(X_std)

# Variance explained
evr = pca.explained_variance_ratio_

# Plot cumulated variance ratio
fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, X_std.shape[1] + 1), evr.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components', fontsize=18)
plt.ylabel('Cumulative Explained Variance', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# Choose number of components
n_comps = evr.shape[0]
for i, exp_var in enumerate(evr.cumsum()):
    if exp_var >= 0.8:
        n_comps = i + 1
        break
print("Number of components:", n_comps)
pca = PCA(n_components=n_comps)
pca.fit(X_std)
scores_pca = pca.transform(X_std)

# Elbow method
visualizer = KElbowVisualizer(KMeans(init='k-means++', random_state=42), k=(1, 21), timings=True)
visualizer.fit(scores_pca)
visualizer.show()
n_clusters = visualizer.elbow_value_
print("Optimal number of clusters:", n_clusters)

# Train kmeans
kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

# Analysis and visualization
df_seg_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_seg_pca_kmeans.columns.values[(-1*n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_

# Plot cluster in 2d
x = df_seg_pca_kmeans['Component 2']
y = df_seg_pca_kmeans['Component 1']
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(x, y, hue=df_seg_pca_kmeans['Cluster'], palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                                                 'tab:purple', 'goldenrod'])
plt.title('Clusters by PCA Components', fontsize=20)
plt.xlabel("Component 2", fontsize=18)
plt.ylabel("Component 1", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# Original features with cluster
train_features = df.drop(['name', 'artist', 'track_URI', 'playlist'], axis=1)
train_features['cluster'] = kmeans_pca.labels_
