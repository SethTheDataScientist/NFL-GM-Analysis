import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", 
    page_title="NFL GM Analysis",
    page_icon="🧊",
    menu_items={
        'About': "Check out my [Portfolio](https://sites.google.com/view/seth-lanza-portfolio/home), Get in touch at Seth.Lanza@gmail.com, Connect on [Linkedin](https://www.linkedin.com/in/sethlanza/), Check my [Twitter](https://x.com/SethDataScience), or Check out my [Github](https://github.com/SethTheDataScientist?tab=repositories)"
    })
st.session_state.target_column = None
st.title("NFL GM Analysis")
st.write("Here is my model output from an analysis of NFL GMs. The data used for training is the draft picks from 2003 to 2024 drafts along with features about the prospects including athletic testing, consensus big board rank, and NGS scores. The model was trained to predict the class label of the GM most likely to draft that player. The model was trained with recency weighting by an exponential function centered on the 2021 draft with an alpha of 0.5 (2024 = 4.48, 2021 = 1.0, and 2003 = 0.00012). Given all the biases inherent with the draft and any post-hoc analysis, the model is not very accurate but it is at least interesting.")
st.write("The model was trained using a XGBoost Classifier with the following hyperparameters and model performance metrics:")
st.write("Hyperparameters: 'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8")
st.write("Model Performance: log_loss: 4.2819, roc_auc_ovr_macro: 0.3894, top_1_accuracy: 0.0279, top_5_accuracy: 0.1540")
st.image("feature_importance.png", caption="XGBoost Feature Importances")


# Read in the upcoming draft table
st.session_state.current_prospects = pd.read_csv("current_year.csv")
st.subheader("Current Year Predictions")
st.write("Let's jump right to the good stuff and see the model output for the upcoming draft class. Below is the top 5 predictions and their probabilites for players in this upcoming draft sorted by most likely first label.")
st.dataframe(st.session_state.current_prospects, use_container_width=True)

st.subheader("Similarity Plot")
# Read in the PCA means table
st.session_state.pca_means = pd.read_csv("GM_PCA.csv")

# Select target column as the GM label
st.session_state.target_column = st.selectbox("Select the specific GM from either a team's most recent GM to make a draft pick or if that GM picked at least 50 draft picks.", st.session_state.pca_means.label.unique())

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


# Read in the SHAP values table
st.subheader("SHAP Analysis")
st.write("Here are some of the SHAP values from the analysis. SHAP values show an effective way of explaining how the model has learned to create the output. This is for the selected GM to give a sense of what the model found them to draft more consistently.")
with open("shap_full.pkl", "rb") as f:
    st.session_state.shap_values = pickle.load(f)
st.session_state.categories = pd.read_csv("categories.csv")['0']
st.session_state.categories = pd.Index(st.session_state.categories)
st.session_state.test_X = pd.read_csv("test_X.csv")
st.session_state.class_idx = st.session_state.categories.get_loc(st.session_state.target_column) 
# If they don't match, you can force alignment:
if list(st.session_state.shap_values.feature_names) != st.session_state.test_X.columns.tolist():
    # Reorder test_X to match SHAP feature names
    st.session_state.test_X = st.session_state.test_X[list(st.session_state.shap_values.feature_names)]
# Extract SHAP values for that class
st.session_state.shap_class_values = st.session_state.shap_values.values[:, :, st.session_state.class_idx]

# Then create a new SHAP Explanation object for that class
shap_class = shap.Explanation(
    values=st.session_state.shap_class_values,
    base_values=st.session_state.shap_values.base_values[:, st.session_state.class_idx],
    data=st.session_state.shap_values.data,
    feature_names=st.session_state.test_X.columns
)

st.subheader("SHAP Beeswarm Plot")
fig, ax = plt.subplots(figsize=(10, 8))  # Create figure with explicit size
shap.plots.beeswarm(shap_class, max_display=20, show=False)  # Add show=False
plt.tight_layout()  # Ensure good spacing
st.pyplot(fig)  # This should now work

st.subheader("SHAP Bar Plot")
fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.plots.bar(shap_class, max_display=20, show=False)
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Waterfall for First Prediction in dataset")
st.write("This shows how the model arrived at the output for that player. The features are sorted by their SHAP value and the red bars are pushing the prediction higher while the blue bars are pushing it lower. The base value is the average model output for all players in the dataset.")
fig3, ax3 = plt.subplots()
shap.plots.waterfall(shap_class[0])
st.pyplot(fig3)

st.session_state.shap_column = st.selectbox("Select the specific feature to see how the SHAP values change across that feature for this specific GM", set(shap_class.feature_names))

feature_index = list(shap_class.feature_names).index(st.session_state.shap_column)
fig4, ax4 = plt.subplots(figsize=(10, 6))

shap.dependence_plot(
    feature_index, 
    shap_class.values,
    shap_class.data,
    feature_names=shap_class.feature_names,
    show=False,
    ax=ax4
)
plt.tight_layout()
st.pyplot(fig4)


# Read in the averages table
st.session_state.averages = pd.read_csv("averages.csv")
st.subheader("Other Tables")
st.write("Below are some tables of outputs of the model by GM or prospect.")
st.write("This first table is the averages of the features for each GM for all players drafted by that GM.")

st.dataframe(st.session_state.averages, use_container_width=True)


# Read in the players and their top 5 GMs
st.session_state.top_5_labels = pd.read_csv("top_5_labels.csv")
st.write("Next is a list of all the players in the sample by the five most likely GMs to draft them.")

st.dataframe(st.session_state.top_5_labels, use_container_width=True)

# Read in the players by if they had a specific GM within their top 5

st.session_state.all_players_top5 = pd.read_csv("all_players_top5.csv")

st.write("Finally is a list of the players by if they had a GM within their top 5 most likely outcomes.")

st.dataframe(st.session_state.all_players_top5, use_container_width=True)

st.subheader("About Me")
st.write('If you want to know more about me and my work: Check out my [Portfolio](https://sites.google.com/view/seth-lanza-portfolio/home), Get in touch at Seth.Lanza@gmail.com, Connect on [Linkedin](https://www.linkedin.com/in/sethlanza/), Check my [Twitter](https://x.com/SethDataScience), or Check out my [Github](https://github.com/SethTheDataScientist?tab=repositories)')