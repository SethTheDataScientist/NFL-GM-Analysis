import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.session_state.target_column = None
st.title("NFL GM Analysis")
st.write("Here is my model output from an analysis of NFL GMs. The data used for training is the draft picks from 2003 to 2024 drafts along with features about the prospects including athletic testing, consensus big board rank, and NGS scores. The model was trained to predict the class label of the GM most likely to draft that player. Given all the biases inherent with the draft and any post-hoc analysis, the model is not very accurate but it is at least interesting.")
st.write("The model was trained using a XGBoost Classifier with the following hyperparameters and model performance metrics:")
st.write("Hyperparameters: 'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8")
st.write("Model Performance: log_loss: 4.4444, roc_auc_ovr_macro: 0.7058, top_1_accuracy: 0.0271, top_5_accuracy: 0.1470")
st.image("feature importance.png", caption="XGBoost Feature Importances")


# Read in the PCA means table
st.session_state.pca_means = pd.read_csv("GM_PCA.csv")

# Select target column as the GM label
st.session_state.target_column = st.selectbox("Select the specific GM", st.session_state.pca_means.label.unique())

st.write("The similarity plot is derived by the principal components of the average features for each GM. The closer the GMs are in the plot, the more similar their average features are.")
st.write("The actual values are not relevant here, just the relative distances and locations near each other. The first two principal components explain 70% of the variance in the data.")

# Filter the PCA means table to get the selected GM
selected = st.session_state.pca_means[st.session_state.pca_means['label'] == st.session_state.target_column]

pca_means = st.session_state.pca_means[st.session_state.pca_means['label'] != st.session_state.target_column]


# Plot button
if st.button("Similarity Plot"):
    plt.figure(figsize=(10, 7))

    # Create scatterplot
    sns.scatterplot(data=pca_means, x='PC1', y='PC2', hue='label', palette='Set2', s=70, alpha=0.8)

    # Add jittered text annotations for each point
    for i in range(pca_means.shape[0]):
        jitter_x = np.random.uniform(-1, 1)  # Jitter in x direction
        jitter_y = np.random.uniform(-1, 1)  # Jitter in y direction
        
        plt.text(pca_means['PC1'].iloc[i] + jitter_x, 
                pca_means['PC2'].iloc[i] + jitter_y, 
                pca_means['label'].iloc[i], 
                fontsize=9, alpha=0.7, color='black', ha='center', va='center')

    # Title and labels
    plt.title("PCA - First Two Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Put dashed lines at every 5 units for both axes
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)


    sns.scatterplot(data=selected, x='PC1', y='PC2', s=100, color = 'red', edgecolor = 'black')
    plt.text(selected['PC1'], 
                selected['PC2'] - 2, 
                selected['label'].values[0],
                fontsize=10, alpha=1, color='black', ha='center', va='center')


    # Remove the legend
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left').set_visible(False)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Show the plot
    st.pyplot(plt)

# Read in the averages table
st.session_state.averages = pd.read_csv("averages.csv")
st.write("Below are some tables of outputs of the model by GM or prospect.")
st.write("This first table is the averages of the features for each GM for all players drafted by that GM.")

st.dataframe(st.session_state.averages)


# Read in the players and their top 5 GMs
st.session_state.top_5_labels = pd.read_csv("top_5_labels.csv")
st.write("Next is a list of all the players in the sample by the five most likely GMs to draft them.")

st.dataframe(st.session_state.top_5_labels)

# Read in the players by if they had a specific GM within their top 5

st.session_state.all_players_top5 = pd.read_csv("all_players_top5.csv")

st.session_state.all_players_top5 = st.session_state.all_players_top5[st.session_state.all_players_top5['GM'] == st.session_state.target_column]

st.write("Finally is a list of the players by if they had the selected GM within their top 5")

st.dataframe(st.session_state.all_players_top5)