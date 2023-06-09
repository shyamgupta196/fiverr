import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

# # upload file
st.title("""K-means Clustering On any dataset""")

# uploaded_file = st.file_uploader("Choose a file")

iris = datasets.load_iris()
df_ori = iris.data


# Load the Iris dataset
# df_ori = pd.read_csv(uploaded_file)

from sklearn.decomposition import PCA
print(df_ori)

target = st.selectbox('select targets' ,iris.feature_names)

pca = PCA(2)

#Transform the data
df = pca.fit_transform(iris)
pca_diag = np.c_[df, df_ori[target]]
dframe = pd.DataFrame(pca_diag)

df = pd.DataFrame(pca_diag)


X = st.selectbox('select columns' ,df.drop(target,axis=1).columns)

# Create a Streamlit application

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
