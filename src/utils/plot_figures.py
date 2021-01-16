import pandas as pd
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler


def plot_cluster_polar_figure(df):
    """
    Plot polar graph for cluster description

    :param df: original features with a cluster label named cluster
    :return:
    """

    # Separate cluster label from train data to normalize features
    # MinMax scaler transform features to a scale from 0 to 1
    train_features = df.drop(['cluster'], axis=1)
    normalized_features = MinMaxScaler().fit_transform(train_features)
    normalized_df = pd.DataFrame(normalized_features, columns=train_features.columns)
    normalized_df['cluster'] = df['cluster']

    # Get mean of each normalized feature
    grouped_df = normalized_df.groupby(['cluster']).mean().reset_index()
    mean_df = grouped_df.melt(id_vars='cluster', var_name='audio_feature', value_name='mean')

    # Make polar plot
    fig = px.line_polar(mean_df, r="mean", theta="audio_feature", color="cluster", line_close=True)

    return fig
