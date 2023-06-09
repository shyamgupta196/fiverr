import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import streamlit as st

st.title('3-D Visualisation of MNIST using t-SNE') 

st.markdown('Kindly wait 2 mins for t-SNE to reduce dimensions')
# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Apply t-SNE to reduce the dimensionality to 3
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for Plotly Express
df = pd.DataFrame()
df["tsne_dim_1"] = X_tsne[:, 0]
df["tsne_dim_2"] = X_tsne[:, 1]
df["tsne_dim_3"] = X_tsne[:, 2]
df["digit_label"] = y

# Define the vibrant color tones
color_scale = px.colors.qualitative.Vivid

# Create a 3D scatter plot with Plotly Express
fig = px.scatter_3d(
    df,
    x="tsne_dim_1",
    y="tsne_dim_2",
    z="tsne_dim_3",
    color="digit_label",
    color_discrete_sequence=color_scale,
    symbol="digit_label",
    title="t-SNE Clustering on MNIST Dataset"
)

fig.update_layout(scene=dict(
    xaxis_title='t-SNE Dimension 1',
    yaxis_title='t-SNE Dimension 2',
    zaxis_title='t-SNE Dimension 3'
))
fig.update_layout(
    autosize=False,
    width=800,
    height=800,)

st.plotly_chart(fig, use_container_width=True)
