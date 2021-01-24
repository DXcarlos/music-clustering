import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

from utils import plot_figures

from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Get data from playlist
df = pd.read_csv('./data/playlist_songs.csv')

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
fig.savefig('./figures/cumulative_explained_variance.png')
plt.clf()

# Choose number of components
n_comps = evr.shape[0]
for i, exp_var in enumerate(evr.cumsum()):
    if exp_var >= 0.9:
        n_comps = i + 1
        break

# Get PCA values
pca = PCA(n_components=n_comps, random_state=42)
pca.fit(X_std)
scores_pca = pca.transform(X_std)

# Elbow method
visualizer = KElbowVisualizer(KMeans(init='k-means++', random_state=42), k=(1, 21), timings=False)
visualizer.fit(scores_pca)
visualizer.show('./figures/elbow_method', clear_figure=True)
n_clusters = visualizer.elbow_value_

# Write metadata to a file
with open("./metadata/experiment.txt", 'w') as outfile:
    outfile.write(f"Number of components needed to have a cumulative variance of 80% : {n_comps} \n")
    outfile.write(f"Optimal number of clusters: {n_clusters} \n")

# Train kmeans
kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

# Save the model
filename = './models/clustering_model.pkl'
pickle.dump(kmeans_pca, open(filename, 'wb'))

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
fig.savefig('./figures/cluster_pca_2d.png')
plt.clf()

# Create dataframe with original train features with cluster labels
train_features = df.drop(['name', 'artist', 'track_URI', 'playlist'], axis=1)
train_features['cluster'] = kmeans_pca.labels_
polar_fig = plot_figures.plot_cluster_polar_figure(train_features, 'single', n_cols=2)
polar_fig.write_image('./figures/single_polar_cluster.png', format='png', scale=2,
                      height=850, width=750, engine='kaleido')
