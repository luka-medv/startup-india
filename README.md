# Automated Text Clustering and Quality Analysis


This Python script performs automated clustering of text data from a CSV file. The script is specifically tailored for analyzing customer complaint data but can be adapted for other types of text data as well. 

Features of the script include:
- Capability to handle CSV files with one or multiple text columns. While the script can process multiple columns, for optimal results, it is recommended to use a file with a single text column that needs analysis.
- Use of TF-IDF Vectorizer for converting the text into vectors.
- Identification of optimal cluster numbers using the Elbow method.
- Implementation of KMeans algorithm for clustering.
- Generation of cluster descriptions based on the most relevant terms.
- Calculation and display of Silhouette score and Davies-Bouldin index to evaluate the quality of the clustering.
- Plotting of clusters for visual interpretation.

This script is beneficial for understanding and summarizing large amounts of text data, finding patterns, and improving customer service by identifying common issues.
