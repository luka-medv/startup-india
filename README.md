# Text Clustering for Startup India

## Description:

This Python script performs unsupervised text clustering on a dataset of call centre data. The script uses the TF-IDF Vectorizer and KMeans algorithm from the Scikit-learn library to cluster the text data. It determines the optimal number of clusters using the elbow method. The script then generates and prints descriptions for each cluster, the silhouette score, and the Davies-Bouldin index. The descriptions are formed by the top N (default is 15, can be modified) keywords in each cluster. It also plots a bar chart of the frequency of issues in each cluster.

The input data can be any CSV file with text data in its columns. The path to the file is defined by the variable `PATH_TO_FILE`.

## Input:

The input file should be a CSV file with text data. Each column of the CSV file is treated as a separate category and clustering is performed on each of these categories. The path to the input file is defined by the variable `PATH_TO_FILE` in the script.

## Setup:

1. Clone the repository to your local machine.
2. Ensure that you have the necessary Python libraries installed. You can install them using pip:
   ```
   pip install pandas scikit-learn matplotlib numpy
   ```
3. Modify the `PATH_TO_FILE` variable in the script to point to your input CSV file.
4. Modify the `NUM_OF_KEYWORDS_PER_CLUSTER` variable to control the number of keywords in the cluster descriptions.
5. Run the script:
   ```
   python text_clustering.py
   ```
6. The script will print the cluster descriptions, silhouette score, and Davies-Bouldin index for each column in the CSV file. It will also display a bar chart showing the frequency of issues in each cluster.
