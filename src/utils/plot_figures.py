import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots


def plot_cluster_polar_figure(df: pd.DataFrame, plot_type: str, n_cols: int = 3) -> go.Figure:
    """
    Plot polar graph for cluster description

    :param df: original features with a cluster label named cluster
    :param plot_type: single or join graph
    :param n_cols: number of subplots per row used only in single plot type
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
        clusters = list(df['cluster'].unique())
        clusters.sort()
        rows = math.ceil(len(clusters) / n_cols)

        # Define figure
        fig = make_subplots(rows=rows, cols=n_cols, specs=[[{'type': 'polar'}] * n_cols] * rows)

        for i, cluster_name in enumerate(clusters):
            fig.add_trace(
                go.Scatterpolar(
                    name=cluster_name,
                    r=grouped_df[grouped_df['cluster'] == cluster_name].drop('cluster', axis=1).values[0],
                    theta=list(train_features.columns),
                ), i // n_cols + 1, i % n_cols + 1)

        fig.update_traces(fill='toself')
        fig.update_layout(showlegend=True)

    else:
        raise Exception('Plot type supporting by now : single and join')

    return fig
