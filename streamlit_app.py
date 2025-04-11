import streamlit as st
import pandas as pd
import os


st.title("NFL GM Analysis")
st.write("Here is my model output from an analysis of NFL GMs. The data used for training is the draft picks from 2003 to 2024 drafts along with features about the prospects including athletic testing, consensus big board rank, and NGS scores. The model was trained to predict the class label of the GM most likely to draft that player. Given all the biases inherent with the draft and any post-hoc analysis, the model is not very accurate but it is at least interesting.")
st.write("The model was trained using a XGBoost Classifier with the following hyperparameters and model performance metrics:")
st.write("Hyperparameters: Best Hyperparameters: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}")
st.write("Model Performance: log_loss: 4.4444, roc_auc_ovr_macro: 0.7058, top_1_accuracy: 0.0271, top_5_accuracy: 0.1470")
st.image("feature importance.png", caption="XGBoost Feature Importances")


# Select target column
# target_column = st.selectbox("Select the target column", st.session_state.train_data.columns)

# gm_scores = pd.read_csv("gm_scores.csv")  # hypothetical file
# st.bar_chart(gm_scores.set_index("GM")["Top5Accuracy"])
