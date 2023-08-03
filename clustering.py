NUM_OF_KEYWORDS_PER_CLUSTER = 15
PATH_TO_FILE = '/Users/lukamedvidovic/Downloads/Call Centre Enquiry Data.csv'




# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(PATH_TO_FILE)

# Define custom stopwords
custom_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Initialize TF-IDF Vectorizer with custom stopwords
vectorizer = TfidfVectorizer(stop_words=custom_stopwords)

# Define function to get top issues for a given column
def get_top_issues(column, max_clusters=15):
    # Extract subjects for the given column
    column_data = df[column].dropna()

    # Vectorize the subjects
    X = vectorizer.fit_transform(column_data)

    # Find the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    elbow_points = np.where(np.gradient(wcss) > np.max(wcss)*0.1)[0]
    elbow_point = elbow_points[0] if len(elbow_points) > 0 else 15

    # Initialize KMeans with the optimal number of clusters
    km = KMeans(n_clusters=elbow_point, random_state=42)

    # Fit KMeans on the data
    km.fit(X)

    # Predict the clusters
    clusters = km.predict(X)

    # Count the frequency of each cluster
    cluster_counts = pd.Series(clusters).value_counts()

    # Get the cluster centers
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    # Generate descriptions for each cluster
    cluster_descriptions = {}
    for i in range(elbow_point):
        cluster_terms = [terms[ind] for ind in order_centroids[i, :NUM_OF_KEYWORDS_PER_CLUSTER]]
        cluster_descriptions[i] = ' '.join(cluster_terms)

    # Calculate Silhouette score and Davies-Bouldin index
    silhouette = silhouette_score(X, clusters)
    dbi = davies_bouldin_score(X.toarray(), clusters)

    print('Silhouette score ranges from -1 to 1. Higher values indicate better clustering.')
    print('Davies-Bouldin index values start from 0. Lower values indicate better clustering.')
    # Return the cluster counts, descriptions, and quality metrics
    return cluster_counts, cluster_descriptions, silhouette, dbi

# Get top issues for each column and plot
for column in df.columns:
    top_issues, cluster_descriptions, silhouette, dbi = get_top_issues(column)
    print(f"\nCluster descriptions for {column} ranked by frequency (Silhouette score: {silhouette:.2f}, Davies-Bouldin index: {dbi:.2f}):\n")
    for cluster, count in top_issues.items():
        print(f"Cluster {cluster}: {cluster_descriptions[cluster]} (Frequency: {count})")
    top_issues.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Top Issues for {column}")
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.show()

