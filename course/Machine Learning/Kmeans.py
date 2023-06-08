import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

# # upload file
# uploaded_file = st.file_uploader("Choose a file")


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Create a Streamlit application
st.title("K-means Clustering on Iris Dataset")

# Add a slider for choosing the number of clusters (k)
k = st.slider("Select the number of clusters (k)", 2, 10)

# Train the K-means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Predict the clusters for each data point
y_pred = kmeans.predict(X)

# Create a DataFrame with the original features and predicted clusters
df = pd.DataFrame(np.column_stack((X, y_pred)), columns=iris.feature_names + ["Cluster"])

# Display the DataFrame
st.write("Iris dataset with predicted clusters:")
st.dataframe(df)

# Plot the clusters
st.write("Cluster visualization:")
fig,ax = plt.subplots(1,1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
st.pyplot(fig)
