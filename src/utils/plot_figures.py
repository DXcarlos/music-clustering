import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler


def plot_cluster_polar_figure(df: pd.DataFrame, plot_type: str) -> go.Figure:
    """
    Plot polar graph for cluster description

    :param df: original features with a cluster label named cluster
    :param plot_type: single or join graph
    :return: figure object
    """

    # Separate cluster label from train data to normalize features
    # MinMax scaler transform features to a scale from 0 to 1
    train_features = df.drop(['cluster'], axis=1)
    normalized_features = MinMaxScaler().fit_transform(train_features)
    normalized_df = pd.DataFrame(normalized_features, columns=train_features.columns)
    normalized_df['cluster'] = df['cluster']

    # Get mean of each normalized feature
    grouped_df = normalized_df.groupby(['cluster']).mean().reset_index()

    # Create diferent graphs depending on type plot
    if plot_type == 'join':

        # Create melt dataframe
        mean_df = grouped_df.melt(id_vars='cluster', var_name='audio_feature', value_name='mean')
        fig = px.line_polar(mean_df, r="mean", theta="audio_feature", color="cluster", line_close=True,
                            title='Cluster Composition')
        fig.update_traces(fill='toself')

    elif plot_type == 'single':

        # Define number of columns and rows
        n_clusters = df['cluster'].nunique()
        cols = 3
        rows = math.ceil(n_clusters / cols)

        # Define figure
        fig = make_subplots(rows=rows, cols=3, specs=[[{'type': 'polar'}] * 3] * rows)

        for i in range(n_clusters):
            fig.add_trace(
                go.Scatterpolar(
                    name=f"Cluster {i}",
                    r=grouped_df[grouped_df['cluster'] == i].drop('cluster', axis=1).values[0],
                    theta=list(train_features.columns),
                ), i // cols + 1, i % cols + 1)

        fig.update_traces(fill='toself')
        fig.update_layout(showlegend=True)

    else:
        raise Exception('Plot type supporting by now : single and join')

    return fig
