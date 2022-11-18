#!/usr/bin/env python3
'Clustering'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans  # clustering algorithms
from sklearn.preprocessing import StandardScaler as SS  # z-score standardization
from sklearn.decomposition import PCA  # dimensionality reduction
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from common import describe_data, test_env


def progressive_feature_selection(data, n_clusters, max_features):
    '''
    We use this function to select the features based on sillhoutte
    score and that way being able to get consistent clusters.;
    inspired by this post:
    https://datascience.stackexchange.com/questions/67040/
    how-to-do-feature-selection-for-clustering-and-implement-it-in-python.
    '''
    feature_list = list(data.columns)
    selected_features = list()
    # select starting feature
    initial_feature = " "
    high_score = 0
    for feature in feature_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data_ = data[feature]
        labels = kmeans.fit_predict(data_.to_frame())
        score_ = silhouette_score(data_.to_frame(), labels)
        print("Proposed new feature {} with score {}". format(feature, score_))
        if score_ >= high_score:
            initial_feature = feature
            high_score = score_
    print("The initial feature is {} with a silhouette score of {}.".format(
        initial_feature, high_score))
    feature_list.remove(initial_feature)
    selected_features.append(initial_feature)
    for _ in range(max_features-1):
        high_score = 0
        selected_feature = ""
        print("Starting selection {}...".format(_))
        for feature in feature_list:
            selection_ = selected_features.copy()
            selection_.append(feature)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_ = data[selection_]
            labels = kmeans.fit_predict(data_)
            score_ = silhouette_score(data_, labels)
            print("Proposed new feature {} with score {}". format(feature, score_))
            if score_ > high_score:
                selected_feature = feature
                high_score = score_
        selected_features.append(selected_feature)
        feature_list.remove(selected_feature)
        print("Selected new feature {} with score {}". format(
            selected_feature, high_score))
    return selected_features


def elbow(data):
    '''
    Elbow method function to find the suitable number of clusters.
    '''
    wcss = []
    max_clusters = 14

    for i in range(1, max_clusters):
        k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
        k_means.fit(data)
        wcss.append(k_means.inertia_)

    plt.xticks(np.arange(1, 14, 1.0))
    plt.plot(range(1, max_clusters), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("./results/elbow.png")


def silhouette_plot(range_, data):
    '''
    we will use this function to plot a silhouette plot that
    helps us to evaluate the cohesion in clusters (k-means only)
    '''
    half_length = int(len(range_)/2)
    range_list = list(range_)
    fig, axis = plt.subplots(half_length, 2, figsize=(15, 8))
    for _ in range_:
        kmeans = KMeans(n_clusters=_, init='k-means++', random_state=42)
        qus, mod = divmod(_ - range_list[0], 2)
        sil_vis = SilhouetteVisualizer(
            kmeans, colors="yellowbrick", ax=axis[qus][mod])
        axis[qus][mod].set_title("Silhouette Plot with n={} Cluster".format(_))
        sil_vis.fit(data)
    fig.tight_layout()
    fig.show()
    fig.savefig("./results/silhouette_plot.png")


def plot_clusters(x_feat, y_feat, figure, file=''):
    '''
    we will use this function to plot the different clusters with different colors and shapes
    '''
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    plt.figure(figure)

    for cluster in range(0, len(set(y_feat))):
        plt.scatter(x_feat[y_feat == cluster, 0], x_feat[y_feat == cluster, 1],
                    s=5, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    plt.title(figure)
    # Remove axes numbers because those are not relevant for visualisation
    plt.xticks([])
    plt.yticks([])

    if file:
        plt.savefig(file)

    plt.show()


def pca_visual(data, n_clusters):
    '''
     we will use this function to visualize the clusters
    '''
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_kmeans = k_means.fit_predict(data)
    x_pca = PCA(n_components=2, random_state=0).fit_transform(data)
    plot_clusters(x_pca, np.full(x_pca.shape[0], 0),
                  'PCA visualisation without clusters',
                  file='./results/PCA_visualisation_without_clusters')
    plot_clusters(x_pca, y_kmeans,
                  'PCA visualisation with clusters',
                  file='./results/PCA_visualisation_with_clusters')

    # Add cluster to data frame as last column and plot with pairplot
    data = data.copy()
    data.loc[:, 'cluster'] = y_kmeans
    sns.pairplot(data[list(data.columns)], hue='cluster')
    plt.savefig('./results/PCA_selected_features_clusters')
    plt.show()


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    # We get rid of the unnamed column and use the column author that has unique values as index.
    df_authors = pd.read_csv(
        "data/DNP_ancient_authors.csv").drop(columns=["Unnamed: 0"])
    # description of data
    # def print_overview(data_frame, file='')
    describe_data.print_overview(
        df_authors, file='results/ancient_authors_overview.txt')
    describe_data.print_categorical(df_authors, columns=['authors',
                                                         'word_count',
                                                         'modern_translations',
                                                         'known_works',
                                                         'manuscripts',
                                                         'early_editions',
                                                         'early_translations',
                                                         'modern_editions',
                                                         'commentaries'],
                                    file='results/ancient_authors_categorical_features.txt')

    # As we can see in the data description the mean is very distant from maximum value
    # so it is advisable to discard the data above 90% at least
    ninety_quantile = df_authors["word_count"].quantile(0.9)
    df_authors = df_authors[df_authors["word_count"] <= ninety_quantile]

    # don't need to preprocess data since we have only numeric values and
    # there is no missing data to be handled.
    df_authors = df_authors.set_index('authors')

    # Scaling the data
    scaler = SS()
    DNP_authors_standardized = scaler.fit_transform(df_authors)
    df_authors_standardized = pd.DataFrame(DNP_authors_standardized,
                                           columns=["word_count_standardized",
                                                    "modern_translations_standardized",
                                                    "known_works_standardized",
                                                    "manuscripts_standardized",
                                                    "early_editions_standardized",
                                                    "early_translations_standardized",
                                                    "modern_editions_standardized",
                                                    "commentaries_standardized"])

    # With this function we can try different feature number and cluster for suitable set up.
    selected_feat = progressive_feature_selection(
        df_authors_standardized, max_features=3, n_clusters=3)

    # The data with selected features
    df_standardized_sliced = df_authors_standardized[selected_feat]

    # Applying elbow method to see which number of cluster are appropriate
    elbow(df_standardized_sliced)

    # Silhoutte plot.
    silhouette_plot(range(3, 9), df_standardized_sliced)

    # PCA visualization.
    pca_visual(df_standardized_sliced, 5)

    print('Done')
