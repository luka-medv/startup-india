PATH_TO_FILE = 'path-to-file'
# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

# Load the data
df = pd.read_csv(PATH_TO_FILE)

# Define the clusters with their keywords
clusters = {
    'Technical Issue': ['difficulty', 'dpiit form', 'logging in', 'startup india portal', 'upload', 'documents', 'technical issues', 'recognition form', 'accessing', 'unable to', 'technical', 'issue', 'problem', 'submit', 'access'],
    'Recognition': ['problems', 'startup registration', 'updating dpiit certificate', 'delay', 'dpiit certificate', 'registration process', 'startup scheme', 'benefits', 'application', 'information'],
    'Startup India Scheme': ['clarification', 'startup benefits', 'loan', 'assistance', 'grants'],
    'Startup Details': ['startup proposal', 'wrong cin number', 'change of company name'],
    'External Engagements': ['resource partnerships', 'corporate partnership', 'event', 'challenge', 'award', 'awards', '2020', '2021', '2022', 'event', 'national awards'],
    'Tax Exemption': ['tax exemption', 'section 56', 'application for exemption', '80 iac', 'tax', 'exemption'],
    'Startup India Seed Fund Scheme': ['fund', 'funding', 'seed fund', 'sisfs', 'investment', 'incubator', 'mentor']
}


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the data
for column in df.columns:
    vectorizer.fit(df[column].dropna())

# Calculate TF-IDF scores for all keywords
cluster_centers = {cluster: np.mean(vectorizer.transform(keywords).toarray(), axis=0) for cluster, keywords in clusters.items()}

# Define function to assign queries to clusters
def assign_to_clusters(column):
    # Extract queries for the given column
    column_data = df[column].dropna()

    # Initialize a dictionary to count the frequency of each cluster and store the queries for silhouette score
    cluster_counts = {cluster: 0 for cluster in clusters}
    cluster_queries = {cluster: [] for cluster in clusters}

    # Assign each query to the cluster it matches the most
    for query in column_data:
        query_vector = vectorizer.transform([query]).toarray()[0]  # Reshape the query vector into a 1D array
        best_cluster = max(cluster_centers, key=lambda cluster: np.dot(query_vector, cluster_centers[cluster]))
        cluster_counts[best_cluster] += 1
        cluster_queries[best_cluster].append(query_vector)

    # Calculate silhouette score and Davies-Bouldin index
    all_queries = np.concatenate(list(cluster_queries.values()))
    labels = np.concatenate([[i]*len(queries) for i, queries in enumerate(cluster_queries.values())])
    silhouette = silhouette_score(all_queries, labels)
    dbi = davies_bouldin_score(all_queries, labels)
    print('Silhouette score: ', silhouette)
    print('Davies-Bouldin index: ', dbi)

    # Return the cluster counts
    return pd.Series(cluster_counts)

# Assign queries to clusters for each column and plot
for column in df.columns:
    cluster_counts = assign_to_clusters(column)
    print(f"\nCluster descriptions for {column} ranked by frequency:\n")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {clusters[cluster]} (Frequency: {count})")
    cluster_counts.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Top Issues for {column}")
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.show()
